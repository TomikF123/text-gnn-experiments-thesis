import json
import os
import argparse
from textgnn.utils import get_project_root
from os.path import join
from textgnn.logger import setup_logger

import mlflow
from textgnn.config_class import Config, DatasetConfig, ModelConfig

logger = setup_logger(__name__)


# Set up MLflow tracking URI
mlflow.set_tracking_uri(join(get_project_root(), "mlruns"))


parser = argparse.ArgumentParser(description="Run model from JSON config")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to JSON config file (relative to project root, e.g., runConfigs/experiments/baseline/lstm_20ng_baseline.json)",
)
args = parser.parse_args()
config = args.config


# print(parser.print_help())
def parse_json(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The file '{config_path}' does not exist.")

    with open(config_path, "r") as file:
        data = json.load(file)

    return data


# Resolve config path - support both relative to project root and absolute paths
if os.path.isabs(config):
    config_path = config
elif os.path.exists(os.path.join(get_project_root(), config)):
    # Path relative to project root (e.g., runConfigs/experiments/baseline/...)
    config_path = os.path.join(get_project_root(), config)
else:
    # Legacy: path relative to runConfigs/ folder
    from textgnn.utils import get_configs_path
    config_path = os.path.join(get_configs_path(), config)

parsed_config = parse_json(config_path)

dataset_config = DatasetConfig(**parsed_config["dataset"])
model_config = ModelConfig(**parsed_config["model_config"])

logger.info("Validating config...")
config = Config(
    run_name=parsed_config["run_name"],
    experiment_name=parsed_config["experiment_name"],
    dataset=dataset_config,
    model_conf=model_config,
)
logger.info(f"Parsed config: {config.dataset.name}")
logger.info(f"Model type: {config.model_conf.model_type}")

# Get pipeline runner based on model type
from textgnn.train import get_pipeline_runner

pipeline_runner = get_pipeline_runner(config.model_conf.model_type)

# Run entire pipeline (data loading → training → evaluation)
trained_model = pipeline_runner(config)

logger.info(f"Training complete!")
if __name__ == "__main__":
    pass
