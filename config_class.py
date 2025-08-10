from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional
from typing import Union, Literal


class PreprocessingConfig(BaseModel):
    remove_stopwords: bool = False
    remove_rare_words: int = 0  # minimum word frequency to keep in the vocabulary
    vocab_size: Optional[int] = (
        None  # restircts to vocabulary to n most frequent words, is applied after removing stopwords and rare words
    )


# Base shared fields (optional)
class BaseEncodingConfig(BaseModel):
    embedding_dim: Optional[int] = None


class ExternalEncodingConfig(BaseEncodingConfig):
    encode_token_type: Literal["glove", "bert"]
    tokens_trained_on: int
    embedding_dim: int


class InternalEncodingConfig(BaseEncodingConfig):
    encode_token_type: Literal["index", "onehot"]


class DatasetConfig(BaseModel):
    name: str
    tvt_split: List[float] = Field(..., min_items=3, max_items=3)  # train/val/test
    shuffle: bool = True
    random_seed: int = 42
    preprocess: Optional[PreprocessingConfig] = None
    encoding: Union[ExternalEncodingConfig, InternalEncodingConfig]

    def __repr__(self):
        base = super().__repr__()
        return base  # , f"DatasetConfig(name={self.name}, tvt_split={self.tvt_split})"


class ModelConfig(BaseModel):
    model_type: str  # e.g., "TextGCN", "LSTM"
    common_params: dict
    model_specific_params: dict


class loggingConfig(BaseModel):
    log_dir: str = "logs"
    log_level: str = "INFO"  # e.g., "DEBUG", "INFO", "WARNING", "ERROR"
    save_model: bool = True
    save_interval: int = 1  # epochs or steps, depending on the model
    tensorboard_enabled: bool = True  # if using TensorBoard for logging


class Config(BaseModel):
    experiment_name: str
    run_name: str
    dataset: DatasetConfig
    # encoding: Union[ExternalEncodingConfig, InternalEncodingConfig]
    model_conf: ModelConfig

    @model_validator(mode="after")
    def set_emb_dim(cls, values):
        if values.model_conf.common_params.get("embedding_dim") is None:
            values.model_conf.common_params["embedding_dim"] = (
                values.dataset.encoding.embedding_dim
            )
        return values

    # Ensure embedding_dim is set from dataset encoding
    # @field_validator
    # def val_embedings(cls, values):

    class Config:
        extra = "forbid"  # Disallow extra fields not defined in the model


if __name__ == "__main__":
    dataset_config = {
        "name": "mr",
        "tvt_split": [0.9, 0, 0.1],
        "random_seed": 42,
        "vocab_size": 10000,
        "preprocess": {"remove_stopwords": False, "remove_rare_words": 0},
        "encoding": {
            "embedding_dim": 300,
            "encode_token_type": "glove",
            "tokens_trained_on": 6,
        },
    }
    model_config = {
        "model_type": "fastText",
        "embedding_dim": 60,
        "common_params": {"batch_size": 128, "num_epochs": 25, "device": "cuda"},
        "model_specific_params": {
            "output_size": 2,
            "hidden_size": 256,
            "num_layers": 2,
            "dropout": 0.95,
            "bidirectional": True,
            "output_size": 2,
            "learning_rate": 0.001,
            "freeze_embeddings": False,
        },
    }

    dataset_config = DatasetConfig(**dataset_config)
    model_config = ModelConfig(**model_config)
    print(dataset_config)
    conf = Config(
        experiment_name="text_classification_experiment",
        run_name="run_1",
        dataset=dataset_config,
        model_config=model_config,
    )
