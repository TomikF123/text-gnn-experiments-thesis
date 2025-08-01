import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset import TextDataset
from prepData import clean_data
from utils import (
    get_data_path,
    get_saved_path,
    get_tensors_tvt_split,
    load_glove_embeddings,
)
import pickle

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


def create_lstm_dataset(dataset_config: dict, save_fn: str):
    name = dataset_config["name"]
    preprocess_config = dataset_config["preprocess"]
    vocab_size = dataset_config.get("vocab_size", None)
    df = pd.read_csv(os.path.join(get_data_path(), f"{name}.csv"))
    df, vocab = clean_data(
        df,
        remove_stop_words=preprocess_config["remove_stopwords"],
        remove_rare_words=preprocess_config["remove_rare_words"],
        vocab_size=vocab_size,
    )
    # X_tensor,y_tensors = encode_lstm_dataset(df, encode_token_type=dataset_config["encoding"]["encode_token_type"],vocab=vocab)
    split_dict = get_tensors_tvt_split(
        tensors={"X": df["text"], "y": df["label"]},
        tvt_split=dataset_config["tvt_split"],
        seed=dataset_config["random_seed"],
    )
    save_dir = os.path.join(get_saved_path(), save_fn)
    os.makedirs(save_dir, exist_ok=True)
    for split, (text, labels) in split_dict.items():
        if text is None or labels is None:
            continue  # Skip val split if not used
        df_split = pd.DataFrame(
            data={
                "text": text.reset_index(drop=True).apply(lambda x: " ".join(x)),
                "label": labels.reset_index(drop=True),
            }
        )
        df_split.to_csv(os.path.join(save_dir, f"{split}.csv"), index=False)
    import pickle

    with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    if dataset_config["encoding"]["encode_token_type"] == "glove":
        # glove_path = dataset_config["encoding"]["glove_path"]
        embedding_dim = dataset_config["encoding"]["embedding_dim"]
        embedding_matrix = load_glove_embeddings(
            vocab, embedding_dim, tokens_trained_on=6
        )  # TODO: tokens_trained_on value is hardcoded, include somehow in config
        # Save embedding matrix
        torch.save(embedding_matrix, os.path.join(save_dir, "embedding_matrix.pt"))


def create_lstm_filename(dataset_config: dict) -> str:
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]
    vocab_size = dataset_config.get("vocab_size", None)
    tokens_trained_on = dataset_config["encoding"].get("tokens_trained_on", None)
    embed_dim = dataset_config["encoding"].get("embedding_dim", None)
    encode_token_type = (
        dataset_config["encoding"]["encode_token_type"] + str(tokens_trained_on) + "B"
        if str(tokens_trained_on)
        else ""
    )
    encode_token_type += f"_{embed_dim}d" if embed_dim else ""
    # PROPERTIES
    return f"{name}_train_{train_ratio}_val_{val_ratio}_test_{test_ratio}_seed_{dataset_config['random_seed']}_stop_words_{remove_stopwords}_rare_words_{remove_rare_words}_vocab_size_{vocab_size}_text_encoded_{encode_token_type}"


class LSTMDataset(TextDataset):
    def __init__(
        self,
        encode_token_type: str = "index",
        embedding_matrix_path: str = None,
        csv_path: str = None,
        vocab_path: str = None,
        max_len: int = None,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.vocab = pickle.load(open(vocab_path, "rb")) if vocab_path else None
        self.encode_token_type = encode_token_type
        self.embedding_matrix = (
            torch.load(embedding_matrix_path) if embedding_matrix_path else None
        )
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()
        self.max_len = max_len
        assert self.vocab is not None, "Vocabulary must be provided."

        # self.max_len = max_len if max_len is not None else max(len(text.split()) for text in self.texts)

        # self.min_len = min(len(text.split()) for text in self.texts)

    def encode_tokens(self, tokens: list[str]) -> torch.Tensor:
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
