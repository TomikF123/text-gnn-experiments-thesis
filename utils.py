
from collections import OrderedDict
import datetime
from os.path import dirname, abspath, join, expanduser, isfile, exists
from os import environ, makedirs
import pytz
import re
from socket import gethostname
from sklearn.model_selection import train_test_split
import torch


def get_root_path():
    return dirname(abspath(__file__))


def get_data_path():
    return join(get_root_path(), 'data')


def get_configs_path():
    return join(get_root_path(), 'runConfigs')


def get_models_path():
    return join(get_root_path(), 'models')

def get_saved_path():
    return join(get_root_path(), 'saved') # cached torch.datasets, splitted according to a config

def get_tensors_tvt_split(tensors: dict,tvt_split: list,seed: int = 42)->dict[str, tuple]:
    #os.makedirs(save_path, exist_ok=True)
    #from sklearn.model_selection import train_test_split
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

def load_glove_embeddings(glove_path: str, vocab: dict, embedding_dim: int) -> torch.Tensor:
    embedding_matrix = torch.randn(len(vocab), embedding_dim)  # Random init for all
    embedding_matrix[vocab["<PAD>"]] = torch.zeros(embedding_dim)  # PAD = 0

    found = 0
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            if word in vocab:
                vec = torch.tensor([float(x) for x in parts[1:]], dtype=torch.float)
                embedding_matrix[vocab[word]] = vec
                found += 1

    print(f"Found {found} GloVe vectors for {len(vocab)} vocab words.")
    return embedding_matrix