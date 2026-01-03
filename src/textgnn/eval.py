import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
from textgnn.utils import get_device



@torch.no_grad()
def evaluate(model, data_loader, device=get_device(), return_preds=False):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        if outputs.dim() > 1 and outputs.size(1) > 1:
            preds = torch.argmax(outputs, dim=1)
        else:
            preds = (outputs > 0.5).long()

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_labels).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
    }

    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))

    if return_preds:
        return metrics, y_true, y_pred
    return metrics
