from pydantic import BaseModel, Field, validator,root_validator
from typing import List, Optional

class PreprocessingConfig(BaseModel):
    remove_stopwords: bool = False
    remove_rare_words: int = 0  # could also be float if you allow frequency thresholds

class EncodingConfig(BaseModel):
    type: str  # e.g., "glove", "onehot", "bert"
    embedding_dim: Optional[int] = None  # not always needed, e.g., for "bert"

class DatasetConfig(BaseModel):
    name: str
    tvt_split: List[float] = Field(..., min_items=3, max_items=3)  # train/val/test
    shuffle: bool = True
    random_seed: int = 42
    preprocessing: Optional[PreprocessingConfig] = None
    encoding: Optional[EncodingConfig] = None
    def __repr__(self):
        base = super().__repr__()
        return base #, f"DatasetConfig(name={self.name}, tvt_split={self.tvt_split})"
class ModelConfig(BaseModel):
    type: str  # e.g., "TextGCN", "LSTM"
    parameters: Optional[dict] = None  # additional model-specific parameters

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
    model_config: ModelConfig
    class Config:
        extra = "forbid"  # Disallow extra fields not defined in the model
