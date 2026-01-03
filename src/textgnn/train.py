TRAINING_LOOPS = {
    "lstm": "models.lstm.train.train_lstm",
    "fastText": "models.fastText.train.train",
    "text_gcn": "models.text_gcn.train.train_text_gcn",  # TODO
}

# Pipeline type registries
PIPELINE_RUNNERS = {
    "inductive": "textgnn.pipelines.inductive.run_inductive_pipeline",
    "transductive": "textgnn.pipelines.transductive.run_transductive_pipeline",
}

MODEL_PIPELINE_TYPES = {
    "lstm": "inductive",
    "fastText": "inductive",
    "text_gcn": "transductive",
}

from .utils import get_function_from_path
from .config_class import ModelConfig

def basic_inductive_training_loop(dataloader, model, config): #TODO?
    ...

def basic_transductive_training_loop(dataloader, model, config): #TODO?
    ...

def train_model(model, dataloaders, config: ModelConfig):
    """
    Train model using model-specific training loop.

    Args:
        model: Model instance
        dataloaders: DataLoader instance
        config: Pydantic ModelConfig model

    Returns:
        Trained model
    """
    model_type = config.model_type
    assert model_type is not None, "model_type must be specified in config"

    if model_type not in TRAINING_LOOPS:
        raise ValueError(f"Unsupported model type: {model_type}")

    train_fn = model.train_func
    return train_fn(dataloader=dataloaders, model=model, config=config)


def get_pipeline_runner(model_type: str):
    """
    Get pipeline runner function for the given model type.

    Args:
        model_type: Type of model (lstm, text_gcn, etc.)

    Returns:
        Pipeline runner function

    Raises:
        ValueError: If model_type is unknown
    """
    if model_type not in MODEL_PIPELINE_TYPES:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline_type = MODEL_PIPELINE_TYPES[model_type]
    runner_path = PIPELINE_RUNNERS[pipeline_type]

    return get_function_from_path(runner_path)
