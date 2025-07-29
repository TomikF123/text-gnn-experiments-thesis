TRAINING_LOOPS = {
    "lstm": "models.lstm.train.train_lstm",
    "text_gcn": "models.text_gcn.train.train_text_gcn",  # TODO
}

from utils import get_function_from_path


def train_model(model, dataloaders, config):
    model_type = config.get("model_type")
    if model_type not in TRAINING_LOOPS:
        raise ValueError(f"Unsupported model type: {model_type}")

    train_fn = get_function_from_path(TRAINING_LOOPS[model_type])
    return train_fn(model, dataloaders, config)
