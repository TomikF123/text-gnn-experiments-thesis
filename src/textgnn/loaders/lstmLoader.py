import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from textgnn.dataset import TextDataset
from textgnn.prepData import clean_data
from textgnn.utils import (
    get_data_path,
    get_saved_path,
    get_tensors_tvt_split,
    load_glove_embeddings,
)
import pickle
from .utils import create_dir_name_based_on_dataset_config
from .create_basic_dataset import create_basic_dataset

# def encode_lstm_dataset(df, encode_token_type, vocab):
#     X = encode_tokens(encode_token_type, df=df["text"], vocab=vocab)
#     y = encode_labels(df["label"])
#     return X, y


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
    dataset_config: dict,
    dataset_save_path: str,
    full_path: str,
    missing_parrent: bool = False,
):
    """
    save some lstm/rnn specific dataset files in the base dataset
    """
    if missing_parrent:
        create_basic_dataset(
            dataset_config=dataset_config, dataset_save_path=dataset_save_path
        )

    vocab = pickle.load(open(os.path.join(dataset_save_path, "vocab.pkl"), "rb"))
    if dataset_config["encoding"]["encode_token_type"] == "glove":
        # glove_path = dataset_config["encoding"]["glove_path"]
        embedding_dim = dataset_config["encoding"]["embedding_dim"]
        embedding_matrix = load_glove_embeddings(
            vocab, embedding_dim, tokens_trained_on=6
        )  # TODO: tokens_trained_on value is hardcoded, include somehow in config
        # Save embedding matrix
        # create the sub dir at path = full_path
        os.makedirs(full_path, exist_ok=True)
        torch.save(embedding_matrix, os.path.join(full_path, "embedding_matrix.pt"))


from textgnn.utils import slugify


def get_lstm_dataset_object(
    dataset_save_path: str,
    full_path: str,
    dataset_config: dict,
    split: str,
    model_type: str,
) -> TextDataset:
    csv_path = os.path.join(dataset_save_path, f"{split}.csv")
    vocab_path = os.path.join(dataset_save_path, "vocab.pkl")
    embedding_matrix_path = os.path.join(full_path, "embedding_matrix.pt")

    return LSTMDataset(
        encode_token_type=dataset_config["encoding"].get("encode_token_type", "index"),
        embedding_matrix_path=embedding_matrix_path,
        csv_path=csv_path,
        vocab_path=vocab_path,
        max_len=dataset_config.get("max_len", None),
    )


def create_lstm_filename(dataset_config: dict, model_type: str) -> str:
    dataset_config["encoding"] = dataset_config["rnn_encoding"]  # TODO this is bad
    name = dataset_config["name"]
    tokens_trained_on = dataset_config["encoding"].get("tokens_trained_on", None)
    embed_dim = dataset_config["encoding"].get("embedding_dim", None)
    encode_token_type = (
        dataset_config["encoding"]["encode_token_type"] + str(tokens_trained_on) + "B"
        if str(tokens_trained_on)
        else ""
    )
    encode_token_type += f"_{embed_dim}d" if embed_dim else ""

    parts = [
        f"{name}_seed_{dataset_config['random_seed']}",
        f"text_encoded_{encode_token_type}",
    ]
    # PROPERTIES
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
        # self.encode_token_type = encode_token_type
        super().__init__(df=df, vocab=vocab, encode_token_type=encode_token_type)
        self.embedding_matrix = (
            torch.load(embedding_matrix_path) if embedding_matrix_path else None
        )
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.max_len = max_len
        self.collate_fn = lstm_collate_fn
        assert self.vocab is not None, "Vocabulary must be provided."

        # self.max_len = max_len if max_len is not None else max(len(text.split()) for text in self.texts)

        # self.min_len = min(len(text.split()) for text in self.texts)

    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:  # TODO: add lru chache
        if self.max_len is not None:
            tokens = tokens[: self.max_len]
        if self.encode_token_type == "index":
            return torch.tensor(
                [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens],
                dtype=torch.long,
            )
        elif self.encode_token_type == "glove":
            assert (
                self.embedding_matrix is not None
            ), "Embedding matrix must be provided for GloVe encoding."
            vectors = []
            for token in tokens:
                idx = self.vocab.get(token, self.vocab["<UNK>"])
                vectors.append(self.embedding_matrix[idx])
            return (
                torch.stack(vectors).float()
                if vectors
                else torch.zeros((1, self.embedding_matrix.shape[1]))
            )
        elif self.encode_token_type == "tf-idf":
            raise NotImplementedError("TF-IDF encoding is not implemented yet.")
        elif self.encode_token_type == "word2vec":
            raise NotImplementedError("Word2Vec encoding is not implemented yet.")
        elif self.encode_token_type == "weighted_bow":
            raise NotImplementedError("Weighted BoW encoding is not implemented yet.")
        else:
            raise ValueError(f"Unknown encoding type: {self.encode_token_type}")

    # def __len__(self):
    #     return len(self.df)

    def __getitem__(self, idx):
        # row = self.df.iloc[idx] # iloc is o(n) supposedly, will use list instead.
        tokens = self.texts[idx].split()
        label = int(self.labels[idx])
        encoded = self.encode_tokens(tokens)
        return encoded, torch.tensor(label)
