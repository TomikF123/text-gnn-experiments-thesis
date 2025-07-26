import os
import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from dataset import TextDataset
from prepData import clean_data, encode_tokens, encode_labels
from utils import get_data_path, get_saved_path,get_tensors_tvt_split,load_glove_embeddings


def encode_lstm_dataset(df, encode_token_type, vocab):
    X = encode_tokens(encode_token_type, df=df["text"], vocab=vocab)
    y = encode_labels(df["label"])
    return X, y

def create_lstm_dataset(dataset_config: dict, save_fn: str):
    name = dataset_config["name"]
    preprocess_config = dataset_config["preprocess"]
    vocab_size = dataset_config.get("vocab_size", None)
    df= pd.read_csv(os.path.join(get_data_path(), f"{name}.csv"))
    df,vocab = clean_data(df, remove_stop_words=preprocess_config["remove_stopwords"], remove_rare_words=preprocess_config["remove_rare_words"],vocab_size=vocab_size) 
    #X_tensor,y_tensors = encode_lstm_dataset(df, encode_token_type=dataset_config["encoding"]["encode_token_type"],vocab=vocab)
    split_dict = get_tensors_tvt_split(tensors={"X": df["text"], "y": df["label"]},tvt_split= dataset_config["tvt_split"],seed=dataset_config["random_seed"])
    save_dir = os.path.join(get_saved_path(), save_fn)
    os.makedirs(save_dir, exist_ok=True)
    for split, (text, labels) in split_dict.items():
        if text is None or labels is None:
            continue  # Skip val split if not used
        df_split = pd.DataFrame(data={"text": text.reset_index(drop=True).apply(lambda x: " ".join(x)),"label": labels.reset_index(drop=True)})
        df_split.to_csv(os.path.join(save_dir, f"{split}.csv"), index=False)
    import pickle
    with open(os.path.join(save_dir, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)
    if dataset_config["encoding"]["encode_token_type"] == "glove":
        glove_path = dataset_config["encoding"]["glove_path"]
        embedding_dim = dataset_config["encoding"]["embedding_dim"]
        embedding_matrix = load_glove_embeddings(glove_path, vocab, embedding_dim)
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
    
    return f"{name}_train_{train_ratio}_val_{val_ratio}_test_{test_ratio}_seed_{dataset_config['random_seed']}_stop_words_{remove_stopwords}_rare_words_{remove_rare_words}_vocab_size_{vocab_size}"

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, vocab: dict, encode_token_type="index", max_len=None):
        self.df = pd.read_csv(csv_path)
        self.vocab = vocab
        self.encode_token_type = encode_token_type
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"].split()  # assumes clean_doc() has already been used
        label = int(row["label"])

        token_ids = [self.vocab.get(w, self.vocab["<UNK>"]) for w in text]

        if self.max_len:
            token_ids = token_ids[:self.max_len]

        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)

