import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from textgnn.utils import get_device



def evaluate(model, data_loader, device=get_device(), return_preds=False):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
      for batch in tqdm(data_loader, desc="Evaluating"):
        # Handle different batch formats
        if isinstance(batch, dict):
            # Dictionary format (e.g., TextING with adj, features, mask)
            # Move all tensors to device, handling special case of adj (list of sparse tensors)
            batch_device = {}
            for k, v in batch.items():
                if k == 'adj' and isinstance(v, list):
                    # adj is a list of sparse tensors - move each to device
                    batch_device[k] = [sparse_tensor.to(device) for sparse_tensor in v]
                elif isinstance(v, torch.Tensor):
                    batch_device[k] = v.to(device)
                else:
                    batch_device[k] = v

            # Extract labels
            labels = batch_device['labels']

            # Forward pass - check if model returns tuple (logits, embeddings) or just logits
            model_args = {k: v for k, v in batch_device.items() if k not in ['labels', 'num_graphs']}
            model_output = model(**model_args)

            if isinstance(model_output, tuple):
                outputs = model_output[0]  # Get logits from (logits, embeddings)
            else:
                outputs = model_output
        else:
            # Tuple format (e.g., LSTM, FastText with inputs, labels)
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)

        # Convert one-hot labels to class indices if needed
        if labels.dim() > 1 and labels.size(1) > 1:
            labels = torch.argmax(labels, dim=1)

        # Get predictions
        if outputs.dim() > 1 and outputs.size(1) > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = (outputs > 0.5).long()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        # Collect probabilities for ROC/PR curves
        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_probs = torch.cat(all_probs).detach().numpy()


    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
    }

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    if return_preds:
        return metrics, y_true, y_pred, y_probs
    return metrics, y_probs
