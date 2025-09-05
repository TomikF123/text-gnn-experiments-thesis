import os
import pandas as pd
import torch
from textgnn.utils import (
    get_data_path,
    get_saved_path,
    get_tensors_tvt_split,
    load_glove_embeddings,
)
from textgnn.dataset import TextDataset
from textgnn.prepData import clean_data


def create_basic_dataset(dataset_config: dict, dataset_save_path: str):
    """
    Creates the base of the dataset needed. That is a preprocessed CSV files and vocab.
    """

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
    save_dir = dataset_save_path
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
    # if dataset_config["encoding"]["encode_token_type"] == "glove":
    #     # glove_path = dataset_config["encoding"]["glove_path"]
    #     embedding_dim = dataset_config["encoding"]["embedding_dim"]
    #     embedding_matrix = load_glove_embeddings(
    #         vocab, embedding_dim, tokens_trained_on=6
    #     )  # TODO: tokens_trained_on value is hardcoded, include somehow in config
    #     # Save embedding matrix
    #     torch.save(embedding_matrix, os.path.join(save_dir, "embedding_matrix.pt"))
