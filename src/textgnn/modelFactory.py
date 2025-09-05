from .utils import get_function_from_path

MODEL_CREATORS = {
    "lstm": "textgnn.models.lstm.model.create_lstm_model",
    "text_gcn": "textgnn.models.text_gcn.model.get_gnn_model_object",
    "fastText": "textgnn.models.fastText.model.create_fasttext_model",
}


def create_model(model_config: dict, dataset_config: dict):
    model_type = model_config["model_type"]
    if model_type not in MODEL_CREATORS:
        raise ValueError(f"Unsupported model type: {model_type}")

    create_fn = get_function_from_path(MODEL_CREATORS[model_type])
    return create_fn(model_config=model_config, dataset_config=dataset_config)
