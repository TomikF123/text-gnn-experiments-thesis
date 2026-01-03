import os
import pandas as pd
import torch
from textgnn.dataset import TextDataset
from textgnn.config_class import DatasetConfig
from .create_basic_dataset import create_basic_dataset

def create_fasttext_artifacts(
    dataset_save_path: str,
    full_path: str,
    dataset_config: DatasetConfig,
    missing_parent: bool
):
    """
    Create FastText-specific artifacts.

    FastText doesn't need pretrained embeddings like LSTM,
    so this function is minimal. It relies on the basic dataset
    artifacts (CSVs) created by create_basic_dataset.

    Args:
        dataset_save_path: Path to save directory
        full_path: Full path to dataset file
        dataset_config: Pydantic DatasetConfig
        missing_parent: Whether parent directory is missing
    """
    # FastText builds its own n-gram vocabulary dynamically during training,
    # so it doesn't need pretrained embeddings. However, we still need
    # to create the full_path directory to signal that artifacts exist.

    if missing_parent:
        create_basic_dataset(
            dataset_config=dataset_config, dataset_save_path=dataset_save_path
        )

    # Create the full_path directory to signal artifacts exist
    # This prevents infinite recursion in load_data()
    os.makedirs(full_path, exist_ok=True)

    # Create a marker file to document what this directory is for
    marker_path = os.path.join(full_path, ".fasttext_ready")
    with open(marker_path, "w") as f:
        f.write("FastText artifacts ready. N-grams built dynamically during training.\n")


def get_fasttext_dataset_object(
    dataset_save_path: str,
    full_path: str,
    dataset_config: DatasetConfig,
    split: str,
    model_type: str
) -> "FastTextDataset":
    """
    Create FastTextDataset instance for a specific split.

    Args:
        dataset_save_path: Path to base dataset directory (contains CSVs)
        full_path: Path to FastText-specific artifacts directory (not used)
        dataset_config: Pydantic DatasetConfig
        split: Which split to load ("train", "val", or "test")
        model_type: Model type string

    Returns:
        FastTextDataset instance
    """
    # Build path to the split CSV file
    csv_path = os.path.join(dataset_save_path, f"{split}.csv")

    return FastTextDataset(
        full_path=csv_path,
        max_len=dataset_config.max_len if dataset_config.max_len is not None else None
    )


def create_fasttext_filename(dataset_config: DatasetConfig, model_type: str) -> str:
    """
    Create filename for FastText dataset.

    Since FastText uses the same preprocessing as other models,
    we use a simple naming convention.

    Args:
        dataset_config: Pydantic DatasetConfig
        model_type: Model type string

    Returns:
        Filename string
    """
    name = dataset_config.name
    return f"{name}_fasttext.pt"


class FastTextDataset(TextDataset):
    """
    Dataset for FastText model.

    Unlike LSTM which encodes text to indices, FastText needs raw text strings
    because it builds character n-gram vocabulary dynamically.

    The dataset loads preprocessed text from CSVs (already tokenized and cleaned)
    and returns it as space-separated strings.
    """

    def __init__(self, full_path: str, max_len: int = None):
        """
        Initialize FastTextDataset.

        Args:
            full_path: Path to CSV file with preprocessed text
            max_len: Maximum sequence length (optional truncation)
        """
        super().__init__()

        # Load data from CSV
        df = pd.read_csv(full_path)

        # Parse tokenized text from string representation
        # Format in CSV: "['word1', 'word2', ...]"
        self.texts = df["text"].apply(eval).tolist()  # Convert string to list
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[str, int]:
        """
        Get a single item from dataset.

        Args:
            idx: Index

        Returns:
            Tuple of (text_string, label)
                - text_string: Space-separated tokenized text
                - label: Integer class label
        """
        tokens = self.texts[idx]

        # Truncate if max_len specified
        if self.max_len is not None:
            tokens = tokens[:self.max_len]

        # Convert list of tokens to space-separated string
        text_string = " ".join(tokens)

        label = self.labels[idx].item()

        return text_string, label

    @staticmethod
    def collate_fn(batch: list[tuple[str, int]]) -> tuple[list[str], torch.Tensor]:
        """
        Collate function for FastText batches.

        Unlike LSTM which needs padding, FastText just needs
        a list of text strings and tensor of labels.

        Args:
            batch: List of (text_string, label) tuples

        Returns:
            Tuple of (texts, labels)
                - texts: List of text strings
                - labels: [batch_size] tensor of labels
        """
        texts = [item[0] for item in batch]
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

        return texts, labels
