import json
import os
import argparse
import utils
from loadData import load_data
import torch
from torch.utils.data import DataLoader
from utils import get_root_path
from os.path import join
from modelFactory import create_model
from train import train_model
from eval import evaluate

# from config_class import Config, DatasetConfig, ModelConfig
import mlflow
from config_class import Config, DatasetConfig, ModelConfig


# Set up MLflow tracking URI
mlflow.set_tracking_uri(join(get_root_path(), "mlruns"))


parser = argparse.ArgumentParser(description="Run model from JSON config")
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Name of the JSON config file inside ./saved (e.g. runConfigTextGCN.json)",
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


configs_path = utils.get_configs_path()
config_path = os.path.join(configs_path, config)
parsed_config = parse_json(config_path)

dataset_config = DatasetConfig(**parsed_config["dataset"])
model_config = ModelConfig(**parsed_config["model_config"])

print("validating config")
config = Config(
    run_name=parsed_config["run_name"],
    experiment_name=parsed_config["experiment_name"],
    dataset=dataset_config,
    model_conf=model_config,
)
print("Parsed config:", config.dataset.name)
print("Loading Data")
train_dataset = load_data(
    dataset_config=config.dataset.model_dump(),
    model_type=config.model_conf.model_type,
    split =None
)

model = create_model(
    model_config=config.model_conf.model_dump(),
    dataset_config=config.dataset.model_dump(),
)


if __name__ == "__main__":
    pass
