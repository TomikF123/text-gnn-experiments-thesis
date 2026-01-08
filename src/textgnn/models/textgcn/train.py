"""Training function for TextGCN model

Implements training loop with patience-based early stopping and MLflow logging.
"""

import torch
import torch.nn as nn
import mlflow
from textgnn.models.textgcn.eval import eval as evaluate_model
from textgnn.logger import setup_logger, log_batch_info

logger = setup_logger(__name__)


def train_textgcn(model, dataloader, config):
    """
    Train TextGCN model using transductive learning.

    Args:
        model: TextGCNClassifier instance
        dataloader: Dict with 'train' and 'val' DataLoaders
        config: ModelConfig with training parameters

    Returns:
        Trained model
    """
    # Extract training parameters
    common_params = config.common_params
    model_params = config.model_specific_params
    device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    epochs = common_params.get("epochs", 200)
    lr = model_params.get("lr", 0.02)
    patience = model_params.get("patience", 10)

    print(f"Training TextGCN on {device}")
    print(f"Parameters: epochs={epochs}, lr={lr}, patience={patience}")

    # Move model to device
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Load graphs (transductive learning - single graph per split)
    # The dataloader dict should have 'train' and 'val' keys
    if isinstance(dataloader, dict):
        train_loader = dataloader['train']
        val_loader = dataloader.get('val', None)
    else:
        # Fallback if single dataloader provided
        train_loader = dataloader
        val_loader = None

    train_data = next(iter(train_loader))
    train_data = train_data.to(device)

    # Log graph structure info (transductive: single graph, no batch iteration)
    # Note: log_batch_every_n doesn't apply here - we always log the full graph once
    log_batch_info(train_data, logger=logger, device=device)

    if val_loader is not None:
        val_data = next(iter(val_loader))
        val_data = val_data.to(device)
    else:
        val_data = None
        print("Warning: No validation data provided")

    # Early stopping tracking
    best_val_acc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        # ===== Training =====
        model.train()
        optimizer.zero_grad()

        # Forward pass on full graph
        logits = model(train_data)

        # Compute loss only on training document nodes
        train_nodes = train_data.doc_mask & train_data.split_mask
        train_logits = logits[train_nodes]
        train_labels = train_data.y[train_nodes]

        loss = criterion(train_logits, train_labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Compute training metrics
        with torch.no_grad():
            train_preds = train_logits.argmax(dim=1).cpu().numpy()
            train_true = train_labels.cpu().numpy()
            train_metrics = evaluate_model(train_preds, train_true)
            train_acc = train_metrics["accuracy"]

        # Free training tensors to reduce memory (keep loss for logging)
        train_loss_value = loss.item()
        del logits, train_logits, train_labels

        # ===== Validation =====
        if val_data is not None:
            model.eval()
            with torch.no_grad():
                val_logits = model(val_data)

                # Compute loss only on validation document nodes
                val_nodes = val_data.doc_mask & val_data.split_mask
                val_logits_subset = val_logits[val_nodes]
                val_labels = val_data.y[val_nodes]

                val_loss = criterion(val_logits_subset, val_labels)

                # Compute validation metrics
                val_preds = val_logits_subset.argmax(dim=1).cpu().numpy()
                val_true = val_labels.cpu().numpy()
                val_metrics = evaluate_model(val_preds, val_true)
                val_acc = val_metrics["accuracy"]

                # Free validation tensors immediately to reduce memory
                del val_logits, val_logits_subset
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        else:
            val_loss = 0.0
            val_acc = 0.0
            val_metrics = {}

        # ===== Logging =====
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss_value:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss.item() if val_data else 0:.4f} | "
                  f"Val Acc: {val_acc:.4f}")

        # MLflow logging
        try:
            mlflow.log_metric("train_loss", train_loss_value, step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            if val_data is not None:
                mlflow.log_metric("val_loss", val_loss.item(), step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
                # Log additional metrics
                for metric_name, metric_value in val_metrics.items():
                    if metric_name != "accuracy":  # Already logged
                        mlflow.log_metric(f"val_{metric_name}", metric_value, step=epoch)
        except Exception as e:
            # MLflow might not be active, continue training
            pass

        # ===== Early Stopping =====
        if val_data is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                epochs_without_improvement = 0

                # Save best model state (move to CPU to avoid GPU memory overhead)
                # Delete old best_model_state first to free memory before cloning
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
