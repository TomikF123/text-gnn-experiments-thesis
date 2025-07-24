import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset import TextDataset
from prepData import clean_data, encode_tokens, encode_labels
from utils import get_data_path, get_saved_path
from utils import get_tensors_tvt_split  # You may later refactor this to utils if needed


def encode_lstm_dataset(df, encode_token_type, vocab):
    X = encode_tokens(encode_token_type, df=df["text"], vocab=vocab)
    y = encode_labels(df["label"])
    return X, y

def create_lstm_dataset(dataset_config: dict, save_fn: str):
    name = dataset_config["name"]
    preprocess_config = dataset_config["preprocess"]
    vocab_size = dataset_config.get("vocab_size", None)
    df= pd.read_csv(os.path.join(get_data_path(), f"{name}.csv"))
    df,vocab = clean_data(df, remove_stop_words=preprocess_config["remove_stopwords"], remove_rare_words=preprocess_config["remove_rare_words"],vocab_size=vocab_size) # remove the vocab..
    X_tensor,y_tensors = encode_lstm_dataset(df, encode_token_type=dataset_config["encoding"]["encode_token_type"],vocab=vocab)
    split_dict = get_tensors_tvt_split(tensors={"X": X_tensor, "y": y_tensors},tvt_split= dataset_config["tvt_split"],save_path=save_fn, seed=dataset_config["random_seed"])
    for split, (X, y) in split_dict.items():
        dataset = LSTMDataset(data=X, labels=y)
        save_dir = os.path.join(get_saved_path(), save_fn)
        os.makedirs(save_dir, exist_ok=True)
        torch.save(dataset, os.path.join(save_dir, f"{split}.pt"))

def create_lstm_filename(dataset_config: dict) -> str:
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]
    vocab_size = dataset_config.get("vocab_size", None)
    
    return f"{name}_train_{train_ratio}_val_{val_ratio}_test_{test_ratio}_seed_{dataset_config['random_seed']}_stop_words_{remove_stopwords}_rare_words_{remove_rare_words}_vocab_size_{vocab_size}"

class LSTMDataset(TextDataset):
    def __init__(self, data:np.array, labels:np.array,pad_token:int=0, max_len:int=None, embeddings_path:str = None):
        super().__init__(data, labels)

    
    def pad_seq(self, seq,max_len):
        pass

    def apply_embeddings(self, embeddings_path,vocab):
        pass
    def build_vocab(self, data):
       pass
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

