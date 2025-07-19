from dataset import TextDataset
import loaders
import loaders.lstmLoader as lstmLoader
from prepData import clean_data
from utils import get_save_path
from utils import get_data_path
import pandas as pd
import torch
import os
from prepData import encode_tokens, encode_labels, encode_dataset

def create_file_name(dataset_config: dict, model_type: str) -> str:
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]

    if model_type == "lstm":
        save_fn = '{}_train_{}_val_{}_test_{}_seed_{}_stop_words_{}_rare_words_{}'.format(name, train_ratio,
                                                              val_ratio, test_ratio,
                                                              dataset_config["random_seed"],remove_stopwords,remove_rare_words)
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
    if save_fn not in get_save_path():
        create_dataset(save_fn,model_type,dataset_config)
        return load_data(dataset_config, model_type)
    else:
        save_path = os.path.join(get_save_path(), save_fn)
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

    

def create_dataset(dataset_config: dict, model_type: str):
    if model_type == "lstm":
        name = dataset_config["name"]
        preprocess_config = dataset_config["preprocessing"]
        df= pd.read_csv(os.path.join(get_data_path(), f"{name}.csv"))
        df = clean_data(df, remove_stop_words=preprocess_config["remove_stopwords"], remove_rare_words=preprocess_config["remove_rare_words"])
        text_tensors,label_tensors = encode_dataset(df,model_type= model_type, encode_token_type=dataset_config["encoding"]["encode_token_type"])


    #if model_type == "lstm":
    # df = pd.read_csv(get_data_path + "/" + name + ".csv")
            # df = clean_data(df, remove_stop_words=remove_stopwords, remove_rare_words=remove_rare_words)
            # dataset = lstmLoader.LSTMDataset(df["text"].values, df["label"].values)
        #     # torch.save(dataset, get_save_path() + "/" + save_fn + ".pt")
        #     return lstmLoader.LSTMDataset(df["text"].values, df["label"].values)
        # elif model_type == "text_gcn":
        #     pass
        # else:
        #     raise ValueError(f"Unsupported model type: {model_type}")