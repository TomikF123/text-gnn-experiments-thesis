from dataset import TextDataset
from prepData import clean_data
from utils import get_data_path, get_saved_path
import pandas as pd
import torch
import os
from utils import get_function_from_path, filter_kwargs_for_class
from torch_geometric.nn import GCNConv, GATConv
import memory_profiler
import line_profiler

ARTIFACT_CREATORS = {
    "lstm": "loaders.lstmLoader.create_lstm_artifacts",
    "text_gcn": "loaders.tempLoader.create_gnn_artifacts",  # future
    "fastText": "loaders.fastTextLoader.create_fasttext_artifacts",  # future
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
GET_DATASET_OBJECT_FUNCS = {
    "lstm": "loaders.lstmLoader.get_lstm_dataset_object",
    "text_gcn": "loaders.tempLoader.get_gnn_dataset_object",  # future
    "text_level_gnn": "loaders.textLevelGNNLoader.get_textlevelgnn_dataset_object",  # future
    "fastText": "loaders.fastTextLoader.get_fasttext_dataset_object",  # future
}
# COLLATE_FN_CREATORS = {
#     "lstm": "loaders.lstmLoader.lstm_collate_fn",
#     "text_gcn": "loaders.lstmLoader.LSTMDataset",  # future
#     "text_level_gnn": "loaders.textLevelGNNLoader.textlevelgnn_collate_fn",  # future
# }


from utils import slugify


def create_dir_name_based_on_dataset_config(dataset_config: dict) -> str:
    dataset_config["encoding"] = dataset_config["rnn_encoding"]
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


def create_dataset_artifacts(
    dataset_save_path: str,
    full_path: str,
    model_type: str,
    dataset_config: dict,
    missing_parrent: bool = False,
):
    # parent_dir_path = os.path.join(get_saved_path(), dataset_save_path)
    if model_type not in ARTIFACT_CREATORS:
        raise ValueError(f"Invalid model type{model_type}")
    create_fn = get_function_from_path(ARTIFACT_CREATORS[model_type])
    create_fn(
        dataset_save_path=dataset_save_path,
        full_path=full_path,
        dataset_config=dataset_config,
        missing_parrent=missing_parrent,
    )


def create_file_name(dataset_config: dict, model_type: str) -> str:
    if model_type not in FILENAME_CREATORS:
        raise ValueError(f"Unsupported model type: {model_type}")
    fn_path = FILENAME_CREATORS[model_type]
    creator_fn = get_function_from_path(fn_path)
    return creator_fn(dataset_config=dataset_config, model_type=model_type)


def get_dataset_class(model_type: str) -> TextDataset:
    if model_type not in DATASETS:
        raise ValueError(f"Unsupported model type: {model_type}")
    dataset_class = get_function_from_path(DATASETS[model_type])
    return dataset_class


def get_dataset_object_func(dataset_config: dict, model_type: str):
    if model_type not in GET_DATASET_OBJECT_FUNCS:
        raise ValueError(f"Unsupported model type: {model_type}")
    get_object_fn = get_function_from_path(GET_DATASET_OBJECT_FUNCS[model_type])
    return get_object_fn


# def load_split():
#    pass


# @memory_profiler.profile
#@line_profiler
def load_data(dataset_config: dict, model_type: str, split: str) -> TextDataset:

    """ Loads or creates and loads the dataset based on the provided configuration and model type."""
    # save_fn = create_file_name(dataset_config, model_type)
    dataset_dir_name = create_dir_name_based_on_dataset_config(dataset_config)
    dataset_save_path = os.path.join(
        get_saved_path(), dataset_dir_name
    )  # path to the dataset save directory
    save_fn = create_file_name(
        dataset_config, model_type
    )  # dir name of the model/architecture specific enodings of the preprocess dataset
    full_path = os.path.join(
        dataset_save_path, save_fn
    )  # full relative path of the model/architecture specific enodings of the preprocess dataset(parrent dir)

    if not os.path.exists(full_path):
        create_dataset_artifacts(
            dataset_save_path=dataset_save_path,
            full_path=full_path,
            model_type=model_type,
            dataset_config=dataset_config,
            missing_parrent=not os.path.exists(dataset_save_path),
        )
        return load_data(dataset_config, model_type, split)

    else:
        # dataset_class = get_dataset_class(model_type)

        # dataset_conf = filter_kwargs_for_class(dataset_class, dataset_config)

        # HARD CODED :/+

        # dataset = dataset_class(**dataset_conf)
        get_dataset_object = get_dataset_object_func(dataset_config, model_type)
        dataset = get_dataset_object(
            dataset_save_path=dataset_save_path,
            full_path=full_path,
            dataset_config=dataset_config,
            split=split,
            model_type=model_type,
        )
        print(dataset)
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
