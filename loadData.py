from dataset import TextDataset
from prepData import clean_data
from utils import get_data_path, get_saved_path
import pandas as pd
import torch
import os
from utils import get_function_from_path, filter_kwargs_for_class
from torch_geometric.nn import GCNConv, GATConv

DATASET_CREATORS = {
    "lstm": "loaders.lstmLoader.create_lstm_dataset",
    "text_gcn": "loaders.tempLoader.create_gnn_dataset",  # future
    "fastText": "loaders.fastTextLoader.create_fasttext_dataset",  # future
}

DATASETS = {
    "lstm": "loaders.lstmLoader.LSTMDataset",
    "fastText": "loaders.lstmLoader.LSTMDataset",
    "text_gcn": "loaders.tempLoader.GraphTextDataset",  # future
}

FILENAME_CREATORS = {
    "lstm": "loaders.lstmLoader.create_lstm_filename",
    "text_gcn": "loaders.tempLoader.create_gnn_filename",  # future
    "text_level_gnn": "loaders.textLevelGNNLoader.create_textlevelgnn_filename",  # future
    "fastText": "loaders.fastTextLoader.create_fasttext_filename",  # future
}

COLLATE_FN_CREATORS = {
    "lstm": "loaders.lstmLoader.lstm_collate_fn",
    "text_gcn": "loaders.textGCNLoader.textgcn_collate_fn",  # future
    "text_level_gnn": "loaders.textLevelGNNLoader.textlevelgnn_collate_fn",  # future
}


from utils import slugify


def create_dir_name_based_on_dataset_config(dataset_config: dict) -> str:
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]
    vocab_size = dataset_config.get("vocab_size", None)
    parts = [
        name,
        f"train_{train_ratio}",
        f"val_{val_ratio}",
        f"test_{test_ratio}",
        f"stop_words_remove_{remove_stopwords}",
        f"rare_words_remove_{remove_rare_words}",
        f"vocab_size_{vocab_size}",
    ]
    dir_name = "_".join(parts)
    return slugify(dir_name)


def create_dir_if_not_exists(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


create_dir_if_not_exists(get_saved_path())


def create_dataset(dataset_save_path:str,full_path:str, model_type: str, dataset_config: dict, missing_parrent: bool = False):
    #parent_dir_path = os.path.join(get_saved_path(), dataset_save_path)
    if model_type not in DATASET_CREATORS:
        raise ValueError(f"Invalid model type{model_type}")
    create_fn = get_function_from_path(DATASET_CREATORS[model_type])
    create_fn(dataset_save_path=dataset_save_path, full_path=full_path, dataset_config=dataset_config, missing_parrent=missing_parrent)


def create_file_name(dataset_config: dict, model_type: str) -> str:
    if model_type not in FILENAME_CREATORS:
        raise ValueError(f"Unsupported model type: {model_type}")
    fn_path = FILENAME_CREATORS[model_type]
    creator_fn = get_function_from_path(fn_path)
    return creator_fn(dataset_config)


def get_dataset_class(model_type: str) -> TextDataset:
    if model_type not in DATASETS:
        raise ValueError(f"Unsupported model type: {model_type}")
    dataset_class = get_function_from_path(DATASETS[model_type])
    return dataset_class

#def load_split():
#    pass

def load_data(dataset_config: dict, model_type: str, split: str) -> TextDataset:
    # save_fn = create_file_name(dataset_config, model_type)
    print(dataset_config.get("vocab_size"), "############VOCAB#############")
    dataset_dir_name = create_dir_name_based_on_dataset_config(dataset_config)
    dataset_save_path = os.path.join(get_saved_path(), dataset_dir_name) # path to the dataset save directory
    save_fn = create_file_name(dataset_config, model_type)  # dir name of the model/architecture specific enodings of the preprocess dataset
    full_path = os.path.join(dataset_save_path, save_fn) # full relative path of the model/architecture specific enodings of the preprocess dataset(parrent dir)

    if not os.path.exists(full_path):
        create_dataset(
            dataset_save_path=dataset_save_path,
            full_path = full_path,
            model_type=model_type,
            dataset_config=dataset_config,
            missing_parrent= not os.path.exists(dataset_save_path),
        )
        return load_data(dataset_config, model_type, split)
 
    else:  
        dataset_class = get_dataset_class(model_type)
        csv_path = os.path.join(dataset_save_path, f"{split}.csv")
        vocab_path = os.path.join(dataset_save_path, "vocab.pkl")
        embedding_matrix_path = os.path.join(full_path, "embedding_matrix.pt")
        print(dataset_config.keys(), "############DEBUG#############")
        dataset_conf = filter_kwargs_for_class(dataset_class, dataset_config)
        dataset_conf["csv_path"] = csv_path
        dataset_conf["vocab_path"] = vocab_path
        dataset_conf["embedding_matrix_path"] = (
            embedding_matrix_path
            if dataset_config["encoding"]["encode_token_type"] == "glove"
            else None
        )  # HARD CODED :/+
        print(dataset_conf.keys(), "############DEBUG#############")
        dataset = dataset_class(**dataset_conf)
        return dataset


if __name__ == "__main__":
    # Example usage
    dataset_config = {
        "name": "20ng",
        "tvt_split": [0.8, 0, 0.1],
        "random_seed": 42,
        "vocab_size": None,
        "preprocess": {"remove_stopwords": False, "remove_rare_words": 5},
        "encoding": {
            "encode_token_type": "glove",
            "embedding_dim": 50,
            "tokens_trained_on": 6,
        },
    }
    model_type = "lstm"
    print("Loading dataset...")
    dataset = load_data(dataset_config, model_type)
    print("Dataset loaded!")
    print(dataset)
