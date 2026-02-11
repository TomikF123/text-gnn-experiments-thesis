import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from textgnn.dataset import TextDataset
from textgnn.config_class import DatasetConfig
from textgnn.utils import (
    get_active_encoding,
    get_data_path,
    get_saved_path,
    get_tensors_tvt_split,
    load_glove_embeddings,
)
import pickle
from .create_basic_dataset import create_basic_dataset


def lstm_collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    inputs, labels = zip(*batch)

    # Determine if inputs are 1D (index) or 2D (glove vectors)
    if inputs[0].dim() == 1:
        # Pad 1D sequences (index encoding)
        padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    elif inputs[0].dim() == 2:
        # Pad 2D sequences (glove: [seq_len, embedding_dim])
        max_len = max(seq.shape[0] for seq in inputs)
        embedding_dim = inputs[0].shape[1]
        padded_inputs = torch.zeros(len(inputs), max_len, embedding_dim)
        for i, seq in enumerate(inputs):
            padded_inputs[i, : seq.shape[0], :] = seq
    else:
        raise ValueError("Unsupported tensor shape")

    labels = torch.stack(labels)
    return padded_inputs, labels


def create_lstm_artifacts(
    dataset_config: DatasetConfig,
    dataset_save_path: str,
    full_path: str,
    missing_parent: bool = False,
) -> None:
    """
    Save LSTM/RNN specific dataset files in the base dataset.

    Args:
        dataset_config: Pydantic DatasetConfig model
        dataset_save_path: Path to base dataset
        full_path: Path for LSTM-specific artifacts
        missing_parent: Whether to create base dataset first
    """
    if missing_parent:
        create_basic_dataset(
            dataset_config=dataset_config, dataset_save_path=dataset_save_path
        )

    vocab = pickle.load(open(os.path.join(dataset_save_path, "vocab.pkl"), "rb"))
    encoding = get_active_encoding(dataset_config)
    # Create GloVe embedding matrix if tokens_trained_on is specified
    # Model will load these into nn.Embedding weights (if None, uses random init)
    if hasattr(encoding, 'tokens_trained_on') and encoding.tokens_trained_on is not None:
        embedding_dim = encoding.embedding_dim
        tokens_trained_on = encoding.tokens_trained_on if encoding.tokens_trained_on is not None else 6
        embedding_matrix = load_glove_embeddings(
            vocab, embedding_dim, tokens_trained_on=tokens_trained_on
        )
        os.makedirs(full_path, exist_ok=True)
        torch.save(embedding_matrix, os.path.join(full_path, "embedding_matrix.pt"))


from textgnn.utils import slugify


def get_lstm_dataset_object(
    dataset_save_path: str,
    full_path: str,
    dataset_config: DatasetConfig,
    split: str,
    model_type: str,
) -> TextDataset:
    """
    Get LSTM dataset object for a specific split.

    Args:
        dataset_save_path: Path to base dataset
        full_path: Path to LSTM-specific artifacts
        dataset_config: Pydantic DatasetConfig model
        split: train/val/test
        model_type: Model type

    Returns:
        LSTMDataset instance
    """
    csv_path = os.path.join(dataset_save_path, f"{split}.csv")
    vocab_path = os.path.join(dataset_save_path, "vocab.pkl")
    embedding_matrix_path = os.path.join(full_path, "embedding_matrix.pt")

    encoding = get_active_encoding(dataset_config)
    return LSTMDataset(
        # encode_token_type=encoding.encode_token_type if encoding.encode_token_type is not None else "index",
        embedding_matrix_path=embedding_matrix_path,
        csv_path=csv_path,
        vocab_path=vocab_path,
        max_len=dataset_config.max_len if dataset_config.max_len is not None else None,
    )


def create_lstm_filename(dataset_config: DatasetConfig, model_type: str) -> str:
    """
    Create filename for LSTM artifacts.

    Args:
        dataset_config: Pydantic DatasetConfig model
        model_type: Model type

    Returns:
        Filename string
    """
    encoding = get_active_encoding(dataset_config)
    name = dataset_config.name
    tokens_trained_on = encoding.tokens_trained_on
    embed_dim = encoding.embedding_dim if hasattr(encoding, 'embedding_dim') else None
    # encode_token_type = (
    #     encoding.encode_token_type + str(tokens_trained_on) + "B"
    #     if tokens_trained_on is not None
    #     else ""
    # )
    # encode_token_type += f"_{embed_dim}d" if embed_dim else ""

    parts = [
        f"{name}_seed_{dataset_config.random_seed}",
        # f"text_encoded_{encode_token_type}",
    ]
    return slugify("_".join(parts))


class LSTMDataset(TextDataset):
    def __init__(
        self,
        encode_token_type: str = "index",
        embedding_matrix_path: str = None,
        csv_path: str = None,
        vocab_path: str = None,
        max_len: int = None,
    ):

        df = pd.read_csv(csv_path)
        vocab = pickle.load(open(vocab_path, "rb")) if vocab_path else None
        super().__init__(df=df, vocab=vocab, encode_token_type=encode_token_type)
        self.embedding_matrix = (
            torch.load(embedding_matrix_path) if embedding_matrix_path else None
        )
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.max_len = max_len
        self.collate_fn = lstm_collate_fn
        assert self.vocab is not None, "Vocabulary must be provided."

    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:
        if self.max_len is not None:
            tokens = tokens[: self.max_len]
        # Always return integer indices (fast GPU embedding in model)
        return torch.tensor(
            [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens],
            dtype=torch.long,
        )

    def __getitem__(self, idx):
        tokens = self.texts[idx].split()
        label = int(self.labels[idx])
        encoded = self.encode_tokens(tokens)
        return encoded, torch.tensor(label)
