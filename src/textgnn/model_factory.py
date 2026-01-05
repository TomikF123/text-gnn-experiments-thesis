from .utils import get_function_from_path
from .config_class import ModelConfig, DatasetConfig

MODEL_CREATORS = {
    "lstm": "textgnn.models.lstm.model.create_lstm_model",
    "text_gcn": "textgnn.models.textgcn.model.create_textgcn_model",
    "texting": "textgnn.models.texting.model.create_texting_model",
    "fastText": "textgnn.models.fastText.model.create_fasttext_model",
}


def create_model(model_config: ModelConfig, dataset_config: DatasetConfig):
    """
    Create model based on configuration.

    Args:
        model_config: Pydantic ModelConfig model
        dataset_config: Pydantic DatasetConfig model

    Returns:
        Model instance
    """
    model_type = model_config.model_type
    if model_type not in MODEL_CREATORS:
        raise ValueError(f"Unsupported model type: {model_type}")

    create_fn = get_function_from_path(MODEL_CREATORS[model_type])
    return create_fn(model_config=model_config, dataset_config=dataset_config)
