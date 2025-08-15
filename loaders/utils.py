def make_dataset_key(dataset_config: dict) -> str:
    pass  # Placeholder for future implementation


def slugify(text):
    import re

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-")


def create_dir_name_based_on_dataset_config(dataset_config: dict) -> str:
    name = dataset_config["name"]
    train_ratio = int(dataset_config["tvt_split"][0] * 100)
    val_ratio = int(dataset_config["tvt_split"][1] * 100)
    test_ratio = 100 - train_ratio - val_ratio
    preprocess_config = dataset_config["preprocess"]
    remove_stopwords = preprocess_config["remove_stopwords"]
    remove_rare_words = preprocess_config["remove_rare_words"]
    vocab_size = dataset_config.get("vocab_size", None)
    # tokens_trained_on = dataset_config["encoding"].get("tokens_trained_on", None)
    # embed_dim = dataset_config["encoding"].get("embedding_dim", None)

    # encode_token_type = (
    #     dataset_config["encoding"]["encode_token_type"] + str(tokens_trained_on) + "B"
    #     if str(tokens_trained_on)
    #     else ""
    # )
    # encode_token_type += f"_{embed_dim}d" if embed_dim else ""
    parts = [
        name,
        f"train_{train_ratio}",
        f"val_{val_ratio}",
        f"test_{test_ratio}",
        f"stop_words_remove_{remove_stopwords}",
        f"rare_words_remove_{remove_rare_words}",
        f"vocab_size_{vocab_size}",
    ]
    dir_name = "_".join(parts)
    return slugify(dir_name)


if __name__ == "__main__":
    # Example usage
    dataset_config = {
        "name": "20ng",
        "tvt_split": [0.8, 0, 0.1],
        "random_seed": 42,
        "vocab_size": None,
        "preprocess": {"remove_stopwords": False, "remove_rare_words": 5},
        "encoding": {
            "encode_token_type": "glove",
            "embedding_dim": 50,
            "tokens_trained_on": 6,
        },
    }
    folder_name = create_folder_based_on_dataset_config(dataset_config)
    print("Folder name based on dataset config:", folder_name)
