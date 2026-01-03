# models/fastText/train.py
import torch
from torch import nn
from textgnn.config_class import ModelConfig
from textgnn.models.base_text_classifier import BaseTextClassifier
from torch.utils.data import DataLoader


def train(model: BaseTextClassifier, dataloader: DataLoader, config: ModelConfig):
    """
    Train FastText model.

    Args:
        model: FastTextClassifier instance
        dataloader: DataLoader with training data
        config: Pydantic ModelConfig model

    Returns:
        Trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training FastText on device: {device}")
    model = model.to(device)

    # Get hyperparameters from config
    lr = config.model_specific_params.get("lr", 0.1)  # FastText default is higher LR
    num_epochs = config.common_params.get("epochs", 10)

    # FastText paper uses SGD, but Adam also works
    optimizer_type = config.model_specific_params.get("optimizer", "sgd")
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for texts, labels in dataloader:
            # texts is a list of strings, labels is a tensor
            # Only move labels to device (texts stay as strings)
            labels = labels.to(device)

            # Forward pass - model.forward() handles the text list
            outputs = model(texts)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        # Compute epoch metrics
        acc = correct / total
        avg_loss = total_loss / len(dataloader)

        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {avg_loss:.4f} - Accuracy: {acc:.4f}"
        )

    return model
