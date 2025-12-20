from collections import OrderedDict
import datetime
from os.path import dirname, abspath, join, expanduser, isfile, exists
from os import environ, makedirs
import pytz
import re
from socket import gethostname
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import zipfile
from contextlib import contextmanager
import importlib
import inspect
import pandas as pd
from torch.utils.data import random_split
# src/textgnn/paths.py
from pathlib import Path


def get_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # Fallback: repo root is two levels up from this file (…/src/textgnn/ -> repo/)
    return here.parents[2]


def get_data_path() -> Path:
    return get_project_root() / "data"


def get_configs_path() -> Path:
    return get_project_root() / "runConfigs"


def get_saved_path() -> Path:
    return get_project_root() / "saved"


def get_tensors_tvt_split(
    tensors: dict, tvt_split: list, seed: int = 42
) -> dict[str, tuple]:
    X = tensors["X"]
    y = tensors["y"]
    train_ratio = tvt_split[0]
    val_ratio = tvt_split[1]
    test_ratio = tvt_split[2]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(1 - train_ratio), random_state=seed, stratify=y
    )
    if val_ratio > 0:
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=(test_ratio / (test_ratio + val_ratio)),
            random_state=seed,
            stratify=y_temp,
        )
    else:
        X_val, y_val = None, None
        X_test, y_test = X_temp, y_temp
    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }





def load_glove_embeddings(
    vocab: dict,
    embedding_dim: int,
    tokens_trained_on: str = 6,
    glove_path: str = None,
    return_missing: bool = False,
    seed: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, list[str]]:
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    if generator is None:
        embedding_matrix = torch.randn(len(vocab), embedding_dim)
    else:
        embedding_matrix = torch.randn(
            len(vocab), embedding_dim, generator=generator
        )
    embedding_matrix[vocab["<PAD>"]] = torch.zeros(embedding_dim)
    found = 0
    found_words = set()
    glove_path = join(get_data_path(), "glove", f"glove.{tokens_trained_on}B.zip")
    inside_zip_name = f"glove.{tokens_trained_on}B.{embedding_dim}d.txt"
    if glove_path.endswith(".zip"):
        assert inside_zip_name, "You must specify the .txt filename inside the zip."
        with zipfile.ZipFile(glove_path) as zf:
            with zf.open(inside_zip_name) as f:
                for line in f:
                    line = line.decode("utf-8")
                    parts = line.strip().split()
                    word = parts[0]
                    if word in vocab:
                        vec = torch.tensor(
                            [float(x) for x in parts[1:]], dtype=torch.float
                        )
                        embedding_matrix[vocab[word]] = vec
                        found += 1
                        found_words.add(word)

    missing_words = sorted(set(vocab.keys()) - found_words)
    print(f"Found {found} GloVe vectors for {len(vocab)} vocab words.")
    print(f"Missing {len(missing_words)} words (e.g., {missing_words[:50]}...)")
    if return_missing:
        return embedding_matrix, missing_words
    else:
        return embedding_matrix


def set_global_seed(seed: int, include_cuda: bool = True):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if include_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"✅ Global seed set to {seed}")


# def get_current_seed():
#     return torch.initial_seed() if torch.cuda.is_available() else torch.seed()
@contextmanager
def with_local_seed(seed: int):
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )

    torch.manual_seed(seed)
    if cuda_rng_state:
        torch.cuda.manual_seed_all(seed)

    try:
        yield
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state:
            torch.cuda.set_rng_state_all(cuda_rng_state)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_function_from_path(path: str):
    module_name, fn_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, fn_name)


def flatten_and_filter(d):
    flat = {}
    for key, value in d.items():
        if isinstance(value, dict):
            flat.update(flatten_and_filter(value))  # Recurse into nested dicts
        else:
            flat[key] = value
    return flat


def filter_kwargs_for_class(cls, config_dict) -> dict:
    signature = inspect.signature(cls.__init__)
    valid_keys = set(signature.parameters.keys()) - {"self"}

    flattened = flatten_and_filter(config_dict)
    return {k: v for k, v in flattened.items() if k in valid_keys}


def create_file_name(**kwargs):  # unused
    flat = flatten_and_filter(kwargs)
    print("Flattened kwargs:", flat)


def slugify(text):
    import re

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def create_act(act, num_parameters=None):
    if act == "relu":
        return nn.ReLU()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    elif act == "softmax":
        return nn.Softmax(dim=1)
    elif act == "softplus":
        return nn.Softplus()
    else:
        raise ValueError(f"Unsupported activation function: {act}")


if __name__ == "__main__":
    pass
