"""
TextING (Inductive Text GNN) Data Loader

Builds per-document graphs with word nodes connected by sliding windows.
Each document becomes its own graph, enabling inductive learning.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from textgnn.config_class import DatasetConfig
from textgnn.utils import slugify, get_data_path
from textgnn.loaders.create_basic_dataset import create_basic_dataset
import scipy.sparse as sp


def create_texting_filename(dataset_config: DatasetConfig, model_type: str) -> str:
    """Generate filename/directory name for TextING artifacts."""
    name = "texting"
    encoding = dataset_config.gnn_encoding

    window_size = encoding.window_size if encoding else 3
    embedding_dim = encoding.embedding_dim if encoding else 300

    parts = [f"{name}_window-{window_size}_dim-{embedding_dim}"]

    return slugify("_".join(parts))


def build_document_graph(words, word_embeddings, window_size=3, weighted=False):
    """
    Build graph for a single document using sliding window.

    Args:
        words: List of tokens in document
        word_embeddings: Dict mapping words to embedding vectors
        window_size: Sliding window size for co-occurrence
        weighted: Whether to use weighted edges (co-occurrence counts)

    Returns:
        adj: scipy sparse adjacency matrix [num_words, num_words]
        features: numpy array of word embeddings [num_words, embedding_dim]
    """
    num_words = len(words)

    if num_words == 0:
        # Empty document - return minimal graph
        return sp.csr_matrix((1, 1)), np.zeros((1, 300))

    # Get embeddings
    embedding_dim = 300
    features = np.zeros((num_words, embedding_dim))
    for i, word in enumerate(words):
        if word in word_embeddings:
            features[i] = word_embeddings[word]
        # else: keep as zeros

    # Build adjacency with sliding window
    edges = []
    edge_weights = {}

    if num_words <= window_size:
        # Document shorter than window - fully connected
        for i in range(num_words):
            for j in range(i + 1, num_words):
                edges.append((i, j))
                edges.append((j, i))
                if weighted:
                    edge_weights[(i, j)] = edge_weights.get((i, j), 0) + 1
                    edge_weights[(j, i)] = edge_weights.get((j, i), 0) + 1
    else:
        # Sliding window
        for start in range(num_words - window_size + 1):
            # Connect all pairs within window
            for i in range(start, start + window_size):
                for j in range(i + 1, start + window_size):
                    edges.append((i, j))
                    edges.append((j, i))
                    if weighted:
                        edge_weights[(i, j)] = edge_weights.get((i, j), 0) + 1
                        edge_weights[(j, i)] = edge_weights.get((j, i), 0) + 1

    # Create sparse adjacency matrix
    if len(edges) > 0:
        rows, cols = zip(*edges)
        if weighted:
            data = [edge_weights.get((r, c), 1.0) for r, c in edges]
        else:
            data = np.ones(len(edges))

        adj = sp.csr_matrix((data, (rows, cols)), shape=(num_words, num_words))
    else:
        # No edges - isolated nodes
        adj = sp.csr_matrix((num_words, num_words))

    return adj, features


def load_glove_embeddings(glove_path, embedding_dim=300):
    """Load GloVe embeddings from file."""
    word_embeddings = {}

    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                vector = np.array(values[1:], dtype='float32')
                if len(vector) == embedding_dim:
                    word_embeddings[word] = vector
            except:
                continue  # Skip malformed lines

    return word_embeddings


def create_texting_artifacts(
    dataset_config: DatasetConfig,
    dataset_save_path: str,
    full_path: str,
    missing_parent: bool = False
) -> None:
    """
    Create (minimal) TextING artifacts.
    TextING builds graphs on-the-fly, so no heavy artifacts needed.

    Args:
        dataset_config: Dataset configuration
        dataset_save_path: Path to base preprocessed dataset (CSVs + vocab)
        full_path: Path where TextING config will be saved
        missing_parent: If True, create basic dataset first
    """
    os.makedirs(full_path, exist_ok=True)

    # Create basic dataset if needed (CSVs + vocab)
    if missing_parent:
        print("Creating basic dataset (CSVs and vocab)...")
        create_basic_dataset(
            dataset_config=dataset_config,
            dataset_save_path=dataset_save_path
        )

    print(f"Configuring TextING at {full_path}...")

    # Get config parameters
    window_size = dataset_config.gnn_encoding.window_size if dataset_config.gnn_encoding else 3
    embedding_dim = dataset_config.gnn_encoding.embedding_dim if dataset_config.gnn_encoding else 300

    # Just save config for reference (no graph pre-computation!)
    config_data = {
        'window_size': window_size,
        'embedding_dim': embedding_dim,
    }
    config_path = os.path.join(full_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"TextING configured with window_size={window_size}, embedding_dim={embedding_dim}")
    print("Note: Graphs will be built on-the-fly during training (like LSTM encodes text)")
    print(f"Config saved to: {config_path}")


def get_texting_dataset_object(
    dataset_save_path: str,
    full_path: str,
    dataset_config: DatasetConfig,
    split: str,
    model_type: str
) -> "TextINGDataset":
    """
    Create TextING dataset with on-the-fly graph construction.

    Args:
        dataset_save_path: Path to base dataset (contains CSVs)
        full_path: Path to TextING artifacts (contains config)
        dataset_config: Dataset configuration
        split: "train", "val", or "test"
        model_type: Model type (texting)

    Returns:
        TextINGDataset instance
    """
    csv_path = os.path.join(dataset_save_path, f"{split}.csv")

    # Get config from dataset_config
    encoding = dataset_config.gnn_encoding
    window_size = encoding.window_size if encoding else 3
    embedding_dim = encoding.embedding_dim if encoding else 300

    return TextINGDataset(
        csv_path=csv_path,
        artifact_path=full_path,
        split=split,
        window_size=window_size,
        embedding_dim=embedding_dim
    )


class TextINGDataset(Dataset):
    """
    Dataset for TextING - each document is a separate graph.
    Uses inductive learning (documents have independent graphs).

    Builds graphs on-the-fly during __getitem__() (like LSTM encodes text).
    """

    def __init__(self, csv_path: str, artifact_path: str, split: str,
                 window_size: int = 3, embedding_dim: int = 300):
        """
        Initialize TextING dataset with on-the-fly graph construction.

        Args:
            csv_path: Path to CSV file with tokenized text
            artifact_path: Path to artifact directory (for config)
            split: "train", "val", or "test"
            window_size: Sliding window size for co-occurrence
            embedding_dim: GloVe embedding dimension
        """
        # Load CSV (lightweight, like LSTM)
        self.df = pd.read_csv(csv_path)
        self.texts = self.df["text"].tolist()
        self.labels = self.df["label"].tolist()

        # Load GloVe embeddings ONCE (shared across all documents)
        print(f"Loading GloVe {embedding_dim}d embeddings...")
        glove_dir = get_data_path() / "glove"
        glove_file = glove_dir / f"glove.6B.{embedding_dim}d.txt"

        if not glove_file.exists():
            raise FileNotFoundError(
                f"GloVe embeddings not found at {glove_file}. "
                f"Run: python -m textgnn.download_data"
            )

        self.word_embeddings = load_glove_embeddings(glove_file, embedding_dim)
        print(f"  Loaded embeddings for {len(self.word_embeddings):,} words")

        # Config
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.split = split
        self.collate_fn = texting_collate_fn

        # Get label info
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}
        self.num_classes = len(unique_labels)

        print(f"Loaded TextING dataset for {split} split:")
        print(f"  - Documents: {len(self.texts)}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Mode: On-the-fly graph construction (like LSTM)")

    def __len__(self):
        """Return number of documents."""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Build graph ON-THE-FLY for this document (like LSTM encodes on-the-fly).

        Returns dict with:
            adj: adjacency matrix (dense)
            features: node features [num_nodes, embedding_dim]
            label: one-hot label
        """
        # Get text tokens
        text = self.texts[idx]
        if isinstance(text, str):
            # Parse list from string representation
            words = eval(text) if text.startswith('[') else text.split()
        else:
            words = []

        # BUILD GRAPH NOW (not pre-computed!)
        adj, features = build_document_graph(
            words,
            self.word_embeddings,
            self.window_size,
            weighted=False
        )

        # Get label (one-hot encode)
        label_idx = self.label_to_idx[self.labels[idx]]
        label_one_hot = np.zeros(self.num_classes)
        label_one_hot[label_idx] = 1

        # Convert sparse adjacency to dense
        if sp.issparse(adj):
            adj = adj.toarray()

        return {
            'adj': adj.astype(np.float32),
            'features': features.astype(np.float32),
            'label': label_one_hot.astype(np.float32)
        }


def texting_collate_fn(batch):
    """
    Collate function for TextING - pads graphs to max size in batch.

    This is memory-efficient as it only pads to the max in THIS batch,
    not globally across all documents.

    Uses sparse adjacency matrices to reduce memory usage (sliding window
    graphs are very sparse, typically <5% density).
    """
    # Find max nodes in this batch
    max_nodes = max([item['adj'].shape[0] for item in batch])
    batch_size = len(batch)
    embedding_dim = batch[0]['features'].shape[1]
    num_classes = batch[0]['label'].shape[0]

    # Pre-allocate tensors
    adj_list = []  # Store as list of sparse tensors instead of dense batch
    features_batch = torch.zeros((batch_size, max_nodes, embedding_dim), dtype=torch.float32)
    mask_batch = torch.zeros((batch_size, max_nodes, 1), dtype=torch.float32)
    labels_batch = torch.zeros((batch_size, num_classes), dtype=torch.float32)

    # Fill with data (and padding)
    for i, item in enumerate(batch):
        num_nodes = item['adj'].shape[0]

        # Convert adjacency to sparse tensor with padding
        adj_dense = item['adj']
        adj_padded = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        adj_padded[:num_nodes, :num_nodes] = adj_dense

        # Convert to sparse COO format
        adj_sparse = torch.from_numpy(adj_padded).to_sparse_coo()
        adj_list.append(adj_sparse)

        # Copy features
        features_batch[i, :num_nodes, :] = torch.from_numpy(item['features'])

        # Create mask (1 for real nodes, 0 for padding)
        mask_batch[i, :num_nodes, :] = 1.0

        # Copy label
        labels_batch[i] = torch.from_numpy(item['label'])

    return {
        'adj': adj_list,  # List of sparse tensors
        'features': features_batch,
        'mask': mask_batch,
        'labels': labels_batch
    }
