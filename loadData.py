from dataset import TextDataset
from prepData import clean_data
from utils import get_data_path, get_saved_path
import pandas as pd
import torch
import os
#from loaders.lstmLoader import LSTMDataset
from utils import get_tensors_tvt_split
from torch.utils.data import DataLoader 


import importlib

DATASET_CREATORS = {
    "lstm": "loaders.lstmLoader.create_lstm_dataset",
    "text_gcn": "loaders.textGCNLoader.create_textgcn_dataset", #future
}

FILENAME_CREATORS = {
    "lstm": "loaders.lstmLoader.create_lstm_filename",
    "text_gcn": "loaders.textGCNLoader.create_textgcn_filename",  # future
    "text_level_gnn": "loaders.textLevelGNNLoader.create_textlevelgnn_filename",  # future
}


def get_function_from_path(path: str):
    module_name, fn_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)

def create_dataset(dataset_config:dict, model_type:str, save_fn:str):
    if model_type not in DATASET_CREATORS:
        raise ValueError(f"Invalid model type{model_type}")
    create_fn = get_function_from_path(DATASET_CREATORS[model_type])
    create_fn(dataset_config, save_fn)

def create_file_name(dataset_config: dict, model_type: str) -> str:
    if model_type not in FILENAME_CREATORS:
        raise ValueError(f"Unsupported model type: {model_type}")
    fn_path = FILENAME_CREATORS[model_type]
    creator_fn = get_function_from_path(fn_path)
    return creator_fn(dataset_config)
#Create a saved folder at get_saved_path() if it does not exist
if not os.path.exists(get_saved_path()):
    os.makedirs(get_saved_path())

def load_data(dataset_config: dict, model_type: str) -> dict[str,TextDataset]:
    save_fn = create_file_name(dataset_config, model_type)
    # name = dataset_config["name"]
    # train_ratio = int(dataset_config["tvt_split"][0] * 100)
    # val_ratio = int(dataset_config["tvt_split"][1] * 100)
    # test_ratio = 100 - train_ratio - val_ratio
    # preprocess_config = dataset_config["preprocess"]
    # remove_stopwords = preprocess_config["remove_stopwords"]
    # remove_rare_words = preprocess_config["remove_rare_words"]
    save_path = os.path.join(get_saved_path(), save_fn)
    if not os.path.exists(save_path):
        create_dataset(save_fn = save_fn, model_type=model_type, dataset_config=dataset_config)
        return load_data(dataset_config, model_type)
    else:
        #save_path = os.path.join(get_saved_path(), save_fn)
        dataset_dict = {}

        train_path = os.path.join(save_path, "train.pt")
        val_path = os.path.join(save_path, "val.pt")
        test_path = os.path.join(save_path, "test.pt")

        if os.path.exists(train_path):
            dataset_dict["train"] = torch.load(train_path,weights_only=False)

        if os.path.exists(val_path):
            dataset_dict["val"] = torch.load(val_path,weights_only=False)

        if os.path.exists(test_path):
            dataset_dict["test"] = torch.load(test_path,weights_only=False)

        if not dataset_dict:
            raise FileNotFoundError(f"No dataset splits found in {save_path}")
    return dataset_dict

if __name__ == "__main__":
    # Example usage
    dataset_config = {
        "name": "20ng",
        "tvt_split": [0.8,0, 0.1],
        "random_seed": 42,
        "vocab_size": None,
        "preprocess": {
            "remove_stopwords": False,
            "remove_rare_words": 0
        },
        "encoding": {
            "encode_token_type": "index"
        }
    }
    model_type = "lstm"
    print("Loading dataset...")
    dataset = load_data(dataset_config, model_type)
    print("Dataset loaded!")
    print(dataset["train"].data[0]) 
