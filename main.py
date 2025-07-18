import json
import os
import argparse
import utils
from loadData import load_data
import torch
from torch.utils.data import DataLoader

def parse_json(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The file '{config_path}' does not exist.")
    
    with open(config_path, 'r') as file:
        data = json.load(file)
    
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model from JSON config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file (e.g. runConfigTextGCN.json)"
    )
    
    args = parser.parse_args()
    config = args.config
    path_to_configs = utils.get_configs_path()
    parsed = parse_json(path_to_configs +"/"+ config)
    print("Model type:", parsed["model_type"])  # e.g., "TextGCN"

    model_type = parsed["model_type"]
    dataset = parsed["dataset"]
    common_params = parsed["common_params"]
    print(type(model_type), type(dataset), type(common_params))
    print(dataset["tvt_split"][0])


    # train_dataset, val_dataset, test_dataset = load_data(model_type, dataset, common_params)

    # train_loader = DataLoader(train_dataset, batch_size=common_params["batch_size"], shuffle=True)

    #from loaders import lstmLoader
    

