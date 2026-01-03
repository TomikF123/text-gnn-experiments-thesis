# models/lstm/train.py
import torch
from torch import nn
from textgnn.config_class import ModelConfig
from textgnn.models.base_text_classifier import BaseTextClassifier


def train(model: BaseTextClassifier, dataloader, config: ModelConfig):
    """
    Train LSTM model with optional validation and early stopping.

    Args:
        model: LSTM model instance
        dataloader: Either a single DataLoader (train only) or dict with 'train' and 'val' keys
        config: ModelConfig with training parameters

    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.model_specific_params.get("lr", 1e-3)
    )
    criterion = nn.CrossEntropyLoss()
    num_epochs = config.common_params.get("epochs", 10)

    # Check if validation dataloader is provided
    if isinstance(dataloader, dict):
        train_loader = dataloader.get("train")
        val_loader = dataloader.get("val")
        has_validation = val_loader is not None
    else:
        train_loader = dataloader
        val_loader = None
        has_validation = False

    # Early stopping parameters (only used if validation is available)
    if has_validation:
        patience = config.model_specific_params.get("patience", 10)
        best_val_acc = 0.0
        epochs_without_improvement = 0
        best_model_state = None
        print(f"Training with validation | patience={patience}")
    else:
        print("Training without validation")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        train_loss = total_loss

        # Validation phase (if available)
        if has_validation:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total

            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

            # Log metrics to MLflow
            try:
                import mlflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)
            except Exception:
                pass  # MLflow context might not exist

            # Early stopping logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                epochs_without_improvement = 0
                best_model_state = model.state_dict().copy()
                print(f"  → New best validation accuracy: {best_val_acc:.4f}")
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.4f}")
                # Restore best model
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                    print("Loaded best model state")
                break
        else:
            # No validation - just print training metrics
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {train_loss:.4f} - Accuracy: {train_acc:.4f}"
            )

            # Log metrics to MLflow (train only)
            try:
                import mlflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            except Exception:
                pass  # MLflow context might not exist

    return model
