"""Training function for TextING model

Implements inductive training loop with per-document graphs using mini-batch gradient descent.
"""

import torch
import torch.nn as nn
import numpy as np
from textgnn.models.texting.eval import eval as evaluate_model


def train_texting(model, dataloader, config):
    """
    Train TextING model using inductive learning (per-document graphs).

    Args:
        model: TextINGClassifier instance
        dataloader: Dict with 'train' and 'val' DataLoaders
        config: ModelConfig with training parameters

    Returns:
        Trained model
    """
    # Extract training parameters
    common_params = config.common_params
    model_params = config.model_specific_params
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = common_params.get("epochs", 15)
    batch_size = common_params.get("batch_size", 128)
    lr = model_params.get("lr", 0.005)
    patience = model_params.get("patience", 10)
    weight_decay = model_params.get("weight_decay", 0)

    print(f"Training TextING on {device}")
    print(f"Parameters: epochs={epochs}, batch_size={batch_size}, lr={lr}, patience={patience}")

    # Move model to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Get dataloaders
    if isinstance(dataloader, dict):
        train_loader = dataloader['train']
        val_loader = dataloader.get('val', None)
    else:
        train_loader = dataloader
        val_loader = None

    # Early stopping tracking
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # ===== Training =====
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in train_loader:
            # Move batch to device (adj is list of sparse tensors)
            adj = [sparse_tensor.to(device) for sparse_tensor in batch['adj']]
            features = batch['features'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            optimizer.zero_grad()
            logits, embeddings = model(adj, features, mask)

            # Compute loss (convert one-hot to class indices)
            labels_idx = torch.argmax(labels, dim=1)
            loss = criterion(logits, labels_idx)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Compute batch metrics
            batch_size = features.size(0)
            train_loss += loss.item() * batch_size
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels_idx).sum().item()
            train_total += batch_size

            # Free tensors immediately to reduce memory
            del adj, features, mask, labels, logits, embeddings, preds, labels_idx, loss
            if device.type == "cuda":
                torch.cuda.empty_cache()

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ===== Validation =====
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_preds_list = []
            val_true_list = []

            with torch.no_grad():
                for batch in val_loader:
                    adj = [sparse_tensor.to(device) for sparse_tensor in batch['adj']]
                    features = batch['features'].to(device)
                    mask = batch['mask'].to(device)
                    labels = batch['labels'].to(device)

                    # Forward pass
                    logits, embeddings = model(adj, features, mask)

                    # Compute loss
                    labels_idx = torch.argmax(labels, dim=1)
                    loss = criterion(logits, labels_idx)

                    batch_size = features.size(0)
                    val_loss += loss.item() * batch_size

                    # Store predictions
                    preds = torch.argmax(logits, dim=1)
                    val_preds_list.append(preds.cpu().numpy())
                    val_true_list.append(labels_idx.cpu().numpy())

                    # Free tensors immediately to reduce memory
                    del adj, features, mask, labels, logits, embeddings, preds, labels_idx, loss
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            val_loss /= len(val_loader.dataset)

            # Compute validation metrics
            val_preds = np.concatenate(val_preds_list)
            val_true = np.concatenate(val_true_list)
            val_metrics = evaluate_model(val_preds, val_true)
            val_acc = val_metrics["accuracy"]
        else:
            val_loss = 0.0
            val_acc = 0.0
            val_metrics = {}

        # ===== Logging =====
        # Print progress
        if (epoch + 1) % 1 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # MLflow logging
        try:
            import mlflow
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            if val_loader is not None:
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                # Log additional metrics
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != "accuracy":  # Already logged
                        mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
        except Exception:
            # MLflow might not be active, continue training
            pass

        # ===== Early Stopping =====
        if val_loader is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model state (move to CPU to avoid GPU memory overhead)
                if best_model_state is not None:
                    del best_model_state
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                best_model_state = {
                    key: value.cpu().clone() for key, value in model.state_dict().items()
                }

                print(f"  → New best validation accuracy: {best_val_acc:.4f}")
            else:
                epochs_without_improvement += 1

            # Check if we should stop
            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
                break

    # ===== Load Best Model =====
    if best_model_state is not None:
        print(f"\nLoading best model from epoch {best_epoch+1}")
        model.load_state_dict(best_model_state)
    else:
        print("\nUsing final model (no validation data)")

    return model
