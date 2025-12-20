import os
import pandas as pd
from textgnn.utils import get_data_path, get_saved_path, split_dataframe_tvt
from textgnn.prep_data import clean_text_pipeline, build_vocabulary, apply_vocabulary
from textgnn.logger import setup_logger

logger = setup_logger(__name__)


def create_basic_dataset(dataset_config: dict, dataset_save_path: str) -> None:
    """
    Creates the base preprocessed dataset with proper train/val/test splitting.

    IMPORTANT: Vocabulary is built ONLY from training data to prevent data leakage.

    Args:
        dataset_config: Dataset configuration dictionary
        dataset_save_path: Path to save preprocessed dataset artifacts
    """
    name = dataset_config["name"]
    preprocess_config = dataset_config["preprocess"]
    vocab_size = dataset_config.get("vocab_size", None)

    # Step 1: Load raw data
    logger.info(f"Loading dataset: {name}")
    df = pd.read_csv(os.path.join(get_data_path(), f"{name}.csv"))

    # Drop rows with missing labels
    df = df.dropna(subset=["label"])
    logger.info(f"Dataset size after dropping NAs: {len(df)}")

    # Step 2: SPLIT FIRST (before any vocabulary-dependent preprocessing)
    logger.info(f"Splitting dataset with ratios: {dataset_config['tvt_split']}")
    split_dfs = split_dataframe_tvt(
        df=df,
        tvt_split=dataset_config["tvt_split"],
        seed=dataset_config["random_seed"],
        label_col="label"
    )

    # Step 3: Clean text for ALL splits (tokenization, lowercasing, etc.)
    logger.info("Cleaning text for all splits...")
    for split_name in split_dfs.keys():
        split_dfs[split_name]["text"] = clean_text_pipeline(
            split_dfs[split_name]["text"]
        )

    # Step 4: Build vocabulary ONLY from training data
    logger.info("Building vocabulary from TRAINING data only...")
    vocab = build_vocabulary(
        text_series=split_dfs["train"]["text"],
        remove_stop_words=preprocess_config["remove_stopwords"],
        remove_rare_words=preprocess_config["remove_rare_words"],
        vocab_size=vocab_size
    )

    # Step 5: Apply vocabulary filtering to ALL splits
    logger.info("Applying vocabulary to all splits...")
    for split_name in split_dfs.keys():
        split_dfs[split_name]["text"] = apply_vocabulary(
            text_series=split_dfs[split_name]["text"],
            vocab=vocab,
            remove_stop_words=preprocess_config["remove_stopwords"]
        )

    # Step 6: Remove empty documents (can occur after vocab filtering)
    for split_name in list(split_dfs.keys()):
        original_len = len(split_dfs[split_name])
        split_dfs[split_name] = split_dfs[split_name][
            split_dfs[split_name]["text"].apply(lambda x: len(x) > 0)
        ].copy()
        new_len = len(split_dfs[split_name])
        if new_len < original_len:
            logger.info(f"Removed {original_len - new_len} empty documents from {split_name} split")

    # Step 7: Save splits to CSV
    os.makedirs(dataset_save_path, exist_ok=True)
    for split_name, split_df in split_dfs.items():
        if split_df is not None and len(split_df) > 0:
            # Convert token lists back to space-separated strings
            split_df_to_save = split_df.copy()
            split_df_to_save["text"] = split_df_to_save["text"].apply(lambda x: " ".join(x))

            csv_path = os.path.join(dataset_save_path, f"{split_name}.csv")
            split_df_to_save.to_csv(csv_path, index=False)
            logger.info(f"Saved {split_name} split: {len(split_df_to_save)} samples to {csv_path}")

    # Step 8: Save vocabulary
    import pickle
    vocab_path = os.path.join(dataset_save_path, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    logger.info(f"Saved vocabulary ({len(vocab)} tokens) to {vocab_path}")
