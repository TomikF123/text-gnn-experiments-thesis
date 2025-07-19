from dataset import TextDataset
import loaders
import loaders.lstmLoader as lstmLoader
from prepData import clean_data
from utils import get_save_path
from utils import get_data_path
import pandas as pd
import torch


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

    return save_fn

def load_data(dataset_config: dict, model_type: str) -> TextDataset:
    save_fn = create_file_name(dataset_config, model_type)
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]
    if save_fn not in get_save_path():
        name = dataset_config["name"]

        if model_type == "lstm":
            df = pd.read_csv(get_data_path + "/" + name + ".csv")
            df = clean_data(df, remove_stop_words=remove_stopwords, remove_rare_words=remove_rare_words)
            dataset = lstmLoader.LSTMDataset(df["text"].values, df["label"].values)
            torch.save(dataset, get_save_path() + "/" + save_fn + ".pt")
            return lstmLoader.LSTMDataset(df["text"].values, df["label"].values)
        elif model_type == "text_gcn":
            pass
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    else:
        return torch.load(get_save_path() + "/" + save_fn + ".pt")

    
    pass
