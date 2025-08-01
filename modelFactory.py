from utils import get_function_from_path

MODEL_CREATORS = {
    "lstm": "models.lstm.model.create_lstm_model",
    "text_gcn": "models.text_gcn.model.create_text_gcn_model",  # future
    "fastText": "models.fastText.model.create_fasttext_model",  # future
}


def create_model(model_config: dict, dataset_config: dict):
    model_type = model_config["model_type"]
    if model_type not in MODEL_CREATORS:
        raise ValueError(f"Unsupported model type: {model_type}")

    create_fn = get_function_from_path(MODEL_CREATORS[model_type])
    return create_fn(model_config, dataset_config)
