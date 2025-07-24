from dataset import TextDataset
import loaders
import loaders.lstmLoader as lstmLoader
from prepData import clean_data
from utils import get_data_path, get_saved_path
import pandas as pd
import torch
import os
from prepData import encode_tokens, encode_labels, encode_dataset
from loaders.lstmLoader import LSTMDataset

#Create a saved folder at get_saved_path() if it does not exist
if not os.path.exists(get_saved_path()):
    os.makedirs(get_saved_path())

def create_file_name(dataset_config: dict, model_type: str) -> str:
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]
    vocab_size = dataset_config.get("vocab_size", None)

    if model_type == "lstm":
        save_fn = '{}_train_{}_val_{}_test_{}_seed_{}_stop_words_{}_rare_words_{}_vocab_size_{}'.format(name, train_ratio,
                                                              val_ratio, test_ratio,
                                                              dataset_config["random_seed"],remove_stopwords,remove_rare_words,vocab_size)
    elif model_type == "text_gcn":
        pass
    elif model_type == "text_level_gnn":
        pass
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return save_fn

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
            dataset_dict["train"] = torch.load(train_path)

        if os.path.exists(val_path):
            dataset_dict["val"] = torch.load(val_path)

        if os.path.exists(test_path):
            dataset_dict["test"] = torch.load(test_path)

        if not dataset_dict:
            raise FileNotFoundError(f"No dataset splits found in {save_path}")

    return dataset_dict

def get_tensors_tvt_split(tensors: dict, save_path: str,tvt_split: list,seed: int = 42)->dict[str, tuple[torch.Tensor, torch.Tensor]]:
    #os.makedirs(save_path, exist_ok=True)
    from sklearn.model_selection import train_test_split
    X = tensors['X']
    y = tensors['y']
    train_ratio = tvt_split[0]
    val_ratio = tvt_split[1]
    test_ratio = tvt_split[2]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y
    )
    if val_ratio > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=seed, stratify=y_temp
        )
    else:
        X_val, y_val = None, None
        X_test, y_test = X_temp, y_temp
    return {"train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test)}

def create_dataset(dataset_config: dict, model_type: str, save_fn: str):
    if model_type == "lstm":
        name = dataset_config["name"]
        preprocess_config = dataset_config["preprocess"]
        vocab_size = dataset_config.get("vocab_size", None)
        df= pd.read_csv(os.path.join(get_data_path(), f"{name}.csv"))
        df,vocab = clean_data(df, remove_stop_words=preprocess_config["remove_stopwords"], remove_rare_words=preprocess_config["remove_rare_words"],vocab_size=vocab_size) # remove the vocab..
        X_tensor,y_tensors = encode_dataset(df,model_type= model_type, encode_token_type=dataset_config["encoding"]["encode_token_type"],vocab=vocab)
        split_dict = get_tensors_tvt_split(tensors={"X": X_tensor, "y": y_tensors},tvt_split= dataset_config["tvt_split"],save_path=save_fn, seed=dataset_config["random_seed"])
        for split, (X, y) in split_dict.items():
            dataset = LSTMDataset(data=X, labels=y)
            save_dir = os.path.join(get_saved_path(), save_fn)
            os.makedirs(save_dir, exist_ok=True)
            torch.save(dataset, os.path.join(save_dir, f"{split}.pt"))
if __name__ == "__main__":
    # Example usage
    dataset_config = {
        "name": "mr",
        "tvt_split": [0.8,0, 0.2],
        "random_seed": 42,
        "vocab_size": 10000,
        "preprocess": {
            "remove_stopwords": False,
            "remove_rare_words": 0
        },
        "encoding": {
            "encode_token_type": "index"
        }
    }
    model_type = "lstm"
    dataset = load_data(dataset_config, model_type)
    print(dataset["train"].data) 
