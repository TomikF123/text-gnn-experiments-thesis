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
    max_len = dataset_config.max_len if dataset_config.max_len else "none"

    parts = [f"{name}_window-{window_size}_dim-{embedding_dim}_maxlen-{max_len}"]

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
    Create TextING artifacts with vocabulary-based approach (like LSTM).

    Stores word IDs (integers) instead of embeddings to avoid duplication.
    Creates embedding matrix once (like LSTM) for GPU to use.

    Args:
        dataset_config: Dataset configuration
        dataset_save_path: Path to base preprocessed dataset (CSVs + vocab)
        full_path: Path where TextING artifacts will be saved
        missing_parent: If True, create basic dataset first
    """
    os.makedirs(full_path, exist_ok=True)

    # Get config parameters
    window_size = dataset_config.gnn_encoding.window_size if dataset_config.gnn_encoding else 3
    embedding_dim = dataset_config.gnn_encoding.embedding_dim if dataset_config.gnn_encoding else 300
    max_len = dataset_config.max_len if dataset_config.max_len else None

    # CACHING: Check if artifacts already exist
    vocab_file = os.path.join(full_path, "vocab.pkl")
    cache_valid = os.path.exists(vocab_file)
    for split in ['train', 'val', 'test']:
        split_file = os.path.join(full_path, f"{split}_data.pkl")
        if not os.path.exists(split_file):
            cache_valid = False
            break

    if cache_valid:
        print(f"TextING artifacts already exist at {full_path} (using cache)")
        return

    # Create basic dataset if needed (CSVs + vocab)
    if missing_parent:
        print("Creating basic dataset (CSVs and vocab)...")
        create_basic_dataset(
            dataset_config=dataset_config,
            dataset_save_path=dataset_save_path
        )

    print(f"Creating TextING artifacts at {full_path}...")

    # Build vocabulary from all splits (like LSTM does)
    print("Building vocabulary from all splits...")
    vocab = {'<PAD>': 0, '<UNK>': 1}  # Special tokens
    word_to_idx = vocab.copy()

    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(dataset_save_path, f"{split}.csv")
        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path)
        for text in df['text']:
            if isinstance(text, str):
                words = eval(text) if text.startswith('[') else text.split()
                for word in words:
                    if word not in word_to_idx:
                        word_to_idx[word] = len(word_to_idx)

    print(f"  Vocabulary size: {len(word_to_idx):,} words")

    # Load GloVe and create embedding matrix (like LSTM does!)
    print(f"Loading GloVe {embedding_dim}d embeddings...")
    glove_dir = get_data_path() / "glove"
    glove_file = glove_dir / f"glove.6B.{embedding_dim}d.txt"

    if not glove_file.exists():
        raise FileNotFoundError(
            f"GloVe embeddings not found at {glove_file}. "
            f"Run: python -m textgnn.download_data"
        )

    word_embeddings = load_glove_embeddings(glove_file, embedding_dim)

    # Create embedding matrix (vocab_size × embedding_dim)
    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim), dtype=np.float32)
    found = 0
    for word, idx in word_to_idx.items():
        if word in word_embeddings:
            embedding_matrix[idx] = word_embeddings[word]
            found += 1

    print(f"  Found GloVe vectors for {found:,}/{len(word_to_idx):,} words ({100*found/len(word_to_idx):.1f}%)")

    # Process each split: Save word IDs + adjacencies
    for split in ['train', 'val', 'test']:
        csv_path = os.path.join(dataset_save_path, f"{split}.csv")
        if not os.path.exists(csv_path):
            print(f"  Skipping {split} (CSV not found)")
            continue

        print(f"  Processing {split}...")
        df = pd.read_csv(csv_path)

        word_ids_list = []
        adjacencies_list = []
        labels_list = []

        # Process each document
        for idx, row in df.iterrows():
            text = row['text']
            label = row['label']

            if isinstance(text, str):
                words = eval(text) if text.startswith('[') else text.split()
            else:
                words = []

            # Cap document length (max_len = max_nodes for TextING)
            if max_len is not None and len(words) > max_len:
                words = words[:max_len]

            # Convert words to IDs (INTEGERS - much smaller!)
            word_ids = [word_to_idx.get(word, 1) for word in words]  # 1 = <UNK>

            # Build adjacency (sparse scipy matrix)
            adj_scipy = build_adjacency_from_word_count(len(words), window_size)

            # CACHE: Convert to PyTorch sparse COO once (memory-efficient for large docs!)
            # This moves expensive conversion from training time to artifact creation time
            adj_coo = adj_scipy.tocoo()
            # Convert to numpy first, then to tensor (much faster!)
            indices = np.vstack([adj_coo.row, adj_coo.col])
            indices = torch.from_numpy(indices).long()
            values = torch.from_numpy(adj_coo.data).float()
            shape = torch.Size(adj_coo.shape)
            adj_torch_sparse = torch.sparse_coo_tensor(indices, values, shape).coalesce()

            word_ids_list.append(word_ids)
            adjacencies_list.append(adj_torch_sparse)  # Store PyTorch sparse (cached!)
            labels_list.append(label)

        # Save consolidated data (MUCH smaller now!)
        split_data = {
            'word_ids': word_ids_list,  # List of integers (tiny!)
            'adjacencies': adjacencies_list,  # List of PyTorch sparse tensors (cached!)
            'labels': labels_list,
            'num_classes': len(set(labels_list))
        }

        split_file = os.path.join(full_path, f"{split}_data.pkl")
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f)

        # Calculate memory usage
        ids_size = sum(len(ids) * 4 for ids in word_ids_list) / (1024**2)  # 4 bytes per int
        print(f"    Saved {len(df)} documents to {split}_data.pkl (~{ids_size:.1f} MB word IDs)")

    # Save vocabulary and embedding matrix
    vocab_data = {
        'word_to_idx': word_to_idx,
        'embedding_matrix': embedding_matrix,
        'vocab_size': len(word_to_idx),
        'embedding_dim': embedding_dim
    }
    with open(vocab_file, 'wb') as f:
        pickle.dump(vocab_data, f)

    emb_matrix_size = embedding_matrix.nbytes / (1024**2)
    print(f"  Saved vocabulary and embedding matrix ({emb_matrix_size:.1f} MB)")

    # Save config
    config_data = {
        'window_size': window_size,
        'embedding_dim': embedding_dim,
        'vocab_size': len(word_to_idx)
    }
    config_path = os.path.join(full_path, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"TextING artifacts created successfully!")
    print(f"  - Vocabulary approach (like LSTM): word IDs instead of embeddings")
    print(f"  - Embedding matrix: {emb_matrix_size:.1f} MB (loaded once on GPU)")
    print(f"  - 500× smaller than storing duplicate embeddings!")
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

    Loads word IDs (like LSTM) - embedding lookup happens on GPU!
    """

    def __init__(self, csv_path: str, artifact_path: str, split: str,
                 window_size: int = 3, embedding_dim: int = 300):
        """
        Initialize TextING dataset with vocabulary-based approach (like LSTM).

        Args:
            csv_path: Path to CSV file (not used, for API compatibility)
            artifact_path: Path to artifact directory
            split: "train", "val", or "test"
            window_size: Sliding window size
            embedding_dim: GloVe embedding dimension
        """
        # Load split data (word IDs + adjacencies)
        split_file = os.path.join(artifact_path, f"{split}_data.pkl")
        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"Artifacts not found at {split_file}. "
                "Run load_data() to create artifacts first."
            )

        print(f"Loading TextING dataset for {split}...")
        with open(split_file, 'rb') as f:
            data = pickle.load(f)

        # Store in memory (TINY - just integers!)
        self.word_ids = data['word_ids']  # List of lists of integers
        self.adjacencies = data['adjacencies']  # List of sparse matrices (tiny!)
        self.labels = data['labels']
        self.num_classes = data['num_classes']

        # Config
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.split = split
        self.collate_fn = texting_collate_fn

        # Get label mapping
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        print(f"  - Documents: {len(self.word_ids)}")
        print(f"  - Classes: {self.num_classes}")
        print(f"  - Mode: Word IDs (like LSTM) - embedding on GPU!")

    def __len__(self):
        """Return number of documents."""
        return len(self.word_ids)

    def __getitem__(self, idx):
        """
        Get word IDs and adjacency from memory (INSTANT!).

        Returns dict with:
            adj: PyTorch sparse COO tensor (PRE-CACHED!)
            word_ids: word indices [num_words] - GPU will embed these!
            label: one-hot label
        """
        # Get from memory (INSTANT - pre-cached sparse tensors!)
        word_ids = self.word_ids[idx]  # List of integers
        adj = self.adjacencies[idx]  # PyTorch sparse COO tensor (PRE-CACHED!)

        # Get label (one-hot encode)
        label_idx = self.label_to_idx[self.labels[idx]]
        label_one_hot = np.zeros(self.num_classes)
        label_one_hot[label_idx] = 1

        # Return sparse tensor directly (no conversion needed!)
        return {
            'adj': adj,  # PyTorch sparse COO tensor (cached!)
            'word_ids': np.array(word_ids, dtype=np.int64),  # Integers!
            'label': label_one_hot.astype(np.float32)
        }


def texting_collate_fn(batch):
    """
    Collate function for TextING with word IDs (like LSTM).

    NO PADDING for adjacency - just pass list of sparse tensors!
    Model handles variable-sized graphs natively.

    SUPER FAST: No padding, no conversion, just collect into lists!
    """
    # Find max nodes for word_ids/mask padding only
    max_nodes = max([item['adj'].shape[0] for item in batch])
    batch_size = len(batch)
    num_classes = batch[0]['label'].shape[0]

    # Adjacency: Just collect sparse tensors in a list (NO PADDING!)
    adj_list = [item['adj'] for item in batch]  # List of sparse tensors

    # Allocate tensors for word_ids, mask, labels (still need padding here)
    word_ids_batch = torch.zeros((batch_size, max_nodes), dtype=torch.long)
    mask_batch = torch.zeros((batch_size, max_nodes, 1), dtype=torch.float32)
    labels_batch = torch.zeros((batch_size, num_classes), dtype=torch.float32)

    # Fill word_ids, mask, labels
    for i, item in enumerate(batch):
        num_nodes = item['adj'].shape[0]

        # Copy word IDs (pad with 0 = <PAD>)
        word_ids_batch[i, :num_nodes] = torch.from_numpy(item['word_ids'])

        # Create mask (1 for real nodes, 0 for padding)
        mask_batch[i, :num_nodes, :] = 1.0

        # Copy label
        labels_batch[i] = torch.from_numpy(item['label'])

    return {
        'adj': adj_list,  # List of sparse tensors (NO PADDING!)
        'word_ids': word_ids_batch,  # [batch_size, max_nodes] integers for embedding!
        'mask': mask_batch,
        'labels': labels_batch
    }
