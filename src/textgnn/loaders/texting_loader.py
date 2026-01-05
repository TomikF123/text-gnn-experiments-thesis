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


def build_adjacency_from_word_count(num_words, window_size=3):
    """
    Build adjacency matrix from word count only (no embedding lookup).

    This is much faster than build_document_graph() as it doesn't need
    to look up GloVe embeddings.

    Args:
        num_words: Number of words in document
        window_size: Sliding window size for co-occurrence

    Returns:
        adj: scipy sparse adjacency matrix [num_words, num_words]
    """
    if num_words == 0:
        return sp.csr_matrix((1, 1))

    # Build adjacency with sliding window
    edges = []

    if num_words <= window_size:
        # Document shorter than window - fully connected
        for i in range(num_words):
            for j in range(i + 1, num_words):
                edges.append((i, j))
                edges.append((j, i))
    else:
        # Sliding window
        for start in range(num_words - window_size + 1):
            # Connect all pairs within window
            for i in range(start, start + window_size):
                for j in range(i + 1, start + window_size):
                    edges.append((i, j))
                    edges.append((j, i))

    # Create sparse adjacency matrix
    if len(edges) > 0:
        rows, cols = zip(*edges)
        data = np.ones(len(edges))
        adj = sp.csr_matrix((data, (rows, cols)), shape=(num_words, num_words))
    else:
        # No edges - isolated nodes
        adj = sp.csr_matrix((num_words, num_words))

    return adj


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
    Create TextING artifacts with pre-computed embeddings.

    Pre-computes GloVe embeddings for all documents (eliminating CPU bottleneck),
    but builds adjacency matrices on-the-fly during training.

    Args:
        dataset_config: Dataset configuration
        dataset_save_path: Path to base preprocessed dataset (CSVs + vocab)
        full_path: Path where TextING artifacts will be saved
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

    print(f"Creating TextING artifacts at {full_path}...")

    # Get config parameters
    window_size = dataset_config.gnn_encoding.window_size if dataset_config.gnn_encoding else 3
    embedding_dim = dataset_config.gnn_encoding.embedding_dim if dataset_config.gnn_encoding else 300

    # Load GloVe embeddings once
    print(f"Loading GloVe {embedding_dim}d embeddings...")
    glove_dir = get_data_path() / "glove"
    glove_file = glove_dir / f"glove.6B.{embedding_dim}d.txt"

    if not glove_file.exists():
        raise FileNotFoundError(
            f"GloVe embeddings not found at {glove_file}. "
            f"Run: python -m textgnn.download_data"
        )

    word_embeddings = load_glove_embeddings(glove_file, embedding_dim)
    print(f"  Loaded {len(word_embeddings):,} word embeddings")

    # Pre-compute embeddings for each split
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(dataset_save_path, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {split} (CSV not found)")
            continue

        print(f"  Pre-computing embeddings for {split}...")
        df = pd.read_csv(csv_path)

        # Create split directory
        split_dir = os.path.join(full_path, split)
        os.makedirs(split_dir, exist_ok=True)

        # Pre-compute embeddings for each document
        for idx, row in df.iterrows():
            text = row['text']
            if isinstance(text, str):
                words = eval(text) if text.startswith('[') else text.split()
            else:
                words = []

            # Convert words to embeddings
            embeddings = np.zeros((len(words), embedding_dim), dtype=np.float32)
            for i, word in enumerate(words):
                if word in word_embeddings:
                    embeddings[i] = word_embeddings[word]

            # Save embeddings
            embed_path = os.path.join(split_dir, f"{idx}.npy")
            np.save(embed_path, embeddings)

        print(f"    Saved {len(df)} document embeddings")

    # Save config
    config_data = {
        'window_size': window_size,
        'embedding_dim': embedding_dim,
    }
    config_path = os.path.join(full_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"TextING artifacts created successfully!")
    print(f"  - Pre-computed embeddings (fast loading, no GloVe lookup at runtime)")
    print(f"  - Adjacency built on-the-fly (minimal overhead)")
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

    Loads pre-computed embeddings (fast, no GloVe lookup),
    builds adjacency on-the-fly (lightweight, just indices).
    """

    def __init__(self, csv_path: str, artifact_path: str, split: str,
                 window_size: int = 3, embedding_dim: int = 300):
        """
        Initialize TextING dataset with pre-computed embeddings.

        Args:
            csv_path: Path to CSV file with labels
            artifact_path: Path to artifact directory with pre-computed embeddings
            split: "train", "val", or "test"
            window_size: Sliding window size for co-occurrence
            embedding_dim: GloVe embedding dimension
        """
        # Load CSV for labels
        self.df = pd.read_csv(csv_path)
        self.labels = self.df["label"].tolist()

        # Path to pre-computed embeddings
        self.embeddings_dir = os.path.join(artifact_path, split)
        if not os.path.exists(self.embeddings_dir):
            raise FileNotFoundError(
                f"Pre-computed embeddings not found at {self.embeddings_dir}. "
                "Run load_data() to create artifacts first."
            )

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
        print(f"  - Documents: {len(self.labels)}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Mode: Pre-computed embeddings + on-the-fly adjacency")

    def __len__(self):
        """Return number of documents."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Load pre-computed embeddings and build adjacency on-the-fly.

        Returns dict with:
            adj: adjacency matrix (dense)
            features: node features [num_nodes, embedding_dim]
            label: one-hot label
        """
        # Load pre-computed embeddings (FAST - just numpy load)
        embed_path = os.path.join(self.embeddings_dir, f"{idx}.npy")
        features = np.load(embed_path)  # [num_words, embedding_dim]

        # Build adjacency from word count (FAST - no GloVe lookup!)
        num_words = len(features)
        adj = build_adjacency_from_word_count(num_words, self.window_size)

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
