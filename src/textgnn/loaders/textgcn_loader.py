import torch
import torch.nn as nn
import pickle
import os
import pandas as pd
from pathlib import Path
from torch_geometric.data import Data as PyGData
from textgnn.dataset import TextDataset
from textgnn.utils import slugify, get_active_encoding
from textgnn.config_class import DatasetConfig
from textgnn.loaders.build_graph import build_text_graph_from_csv, load_dataset_csvs
from textgnn.loaders.create_basic_dataset import create_basic_dataset


def create_textgcn_filename(dataset_config: DatasetConfig, model_type: str) -> str:
    """Generate filename/directory name for TextGCN artifacts based on config."""
    name = "text_gcn"
    encoding = dataset_config.gnn_encoding

    x_type = encoding.x_type if encoding else "identity"
    window_size = encoding.window_size if encoding else 20

    parts = [f"{name}_x-{x_type}_window-{window_size}"]

    # Add embedding dimension if using external embeddings
    if encoding and encoding.embedding_dim:
        parts.append(f"dim-{encoding.embedding_dim}")

    return slugify("_".join(parts))


def create_textgcn_artifacts(
    dataset_config: DatasetConfig,
    dataset_save_path: str,
    full_path: str,
    missing_parent: bool = False
) -> None:
    """
    Create TextGCN graph artifacts from preprocessed CSVs.

    Args:
        dataset_config: Dataset configuration
        dataset_save_path: Path to base preprocessed dataset (contains CSVs and vocab)
        full_path: Path where TextGCN-specific artifacts will be saved
        missing_parent: If True, create basic dataset first
    """
    os.makedirs(full_path, exist_ok=True)

    # If parent dataset doesn't exist, create it first
    if missing_parent:
        print("Creating basic dataset (CSVs and vocab)...")
        if missing_parent:
            create_basic_dataset(
            dataset_config=dataset_config, dataset_save_path=dataset_save_path
        )

    print(f"Creating TextGCN artifacts at {full_path}...")

    # Load all CSVs to build complete graph
    dataset_path = Path(dataset_save_path)
    train_df = pd.read_csv(dataset_path / "train.csv")
    val_df = pd.read_csv(dataset_path / "val.csv")
    test_df = pd.read_csv(dataset_path / "test.csv")

    # Combine all documents for graph construction
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Build graph from all documents
    window_size = dataset_config.gnn_encoding.window_size if dataset_config.gnn_encoding else 20

    # Save combined CSV temporarily for graph building
    temp_csv_path = dataset_path / "ALL.csv"
    all_df.to_csv(temp_csv_path, index=False)

    try:
        # Build graph using existing function
        graph_data = build_text_graph_from_csv(
            dataset_path=str(dataset_path),
            text_col="text",
            label_col="label",
            split=None,  # Use all documents
            window_size=window_size
        )
    finally:
        # Clean up temporary file
        if temp_csv_path.exists():
            temp_csv_path.unlink()

    # Extract graph components
    adj = graph_data["adj"]  # scipy.sparse.csr_matrix
    labels = graph_data["labels"]  # list of labels
    vocab = graph_data["vocab"]  # list of words
    word_id_map = graph_data["word_id_map"]  # dict: word -> id
    docs = graph_data["docs"]  # list of document texts

    num_docs = len(docs)
    num_words = len(vocab)
    num_nodes = num_docs + num_words  # Document nodes + word nodes

    print(f"Graph statistics:")
    print(f"  - Documents: {num_docs}")
    print(f"  - Words: {num_words}")
    print(f"  - Total nodes: {num_nodes}")
    print(f"  - Edges: {adj.nnz}")

    # Convert scipy sparse matrix to PyTorch Geometric format
    adj_coo = adj.tocoo()
    edge_index = torch.tensor([adj_coo.row, adj_coo.col], dtype=torch.long)
    edge_attr = torch.tensor(adj_coo.data, dtype=torch.float)

    # Create labels tensor (only for document nodes, -1 for word nodes)
    # Map string labels to integers
    unique_labels = sorted(set(labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    y = torch.full((num_nodes,), -1, dtype=torch.long)
    for i, label in enumerate(labels):
        y[i] = label_to_idx[label]

    # Create masks for document vs word nodes
    doc_mask = torch.zeros(num_nodes, dtype=torch.bool)
    doc_mask[:num_docs] = True

    word_mask = torch.zeros(num_nodes, dtype=torch.bool)
    word_mask[num_docs:] = True

    # Create split masks based on original CSVs
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # Track cumulative index as we process each split
    current_idx = 0
    for df, mask in [(train_df, train_mask), (val_df, val_mask), (test_df, test_mask)]:
        num_docs_in_split = len(df)
        mask[current_idx:current_idx + num_docs_in_split] = True
        current_idx += num_docs_in_split

    # Save all artifacts
    print(f"Saving artifacts to {full_path}...")
    torch.save(edge_index, os.path.join(full_path, "ALL_edge_index.pt"))
    torch.save(edge_attr, os.path.join(full_path, "ALL_edge_attr.pt"))
    torch.save(y, os.path.join(full_path, "ALL_y.pt"))
    torch.save(doc_mask, os.path.join(full_path, "ALL_doc_mask.pt"))
    torch.save(word_mask, os.path.join(full_path, "ALL_word_mask.pt"))
    torch.save(train_mask, os.path.join(full_path, "train_mask.pt"))
    torch.save(val_mask, os.path.join(full_path, "val_mask.pt"))
    torch.save(test_mask, os.path.join(full_path, "test_mask.pt"))

    # Save metadata
    meta = {
        "vocab": vocab,
        "word_id_map": word_id_map,
        "num_classes": len(unique_labels),
        "num_nodes": num_nodes,
        "num_docs": num_docs,
        "num_words": num_words,
        "label_to_idx": label_to_idx,
        "idx_to_label": {idx: label for label, idx in label_to_idx.items()},
    }
    with open(os.path.join(full_path, "ALL_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("TextGCN artifacts created successfully!")


def get_textgcn_dataset_object(
    dataset_save_path: str,
    full_path: str,
    dataset_config: DatasetConfig,
    split: str,
    model_type: str
) -> "TextGCNDataset":
    """
    Load and return TextGCN dataset object for a specific split.

    Args:
        dataset_save_path: Path to base dataset
        full_path: Path to TextGCN artifacts
        dataset_config: Dataset configuration
        split: "train", "val", or "test"
        model_type: Model type (should be "text_gcn")

    Returns:
        TextGCNDataset instance
    """
    x_type = dataset_config.gnn_encoding.x_type if dataset_config.gnn_encoding else "identity"

    return TextGCNDataset(
        artifact_path=full_path,
        split=split,
        x_type=x_type
    )


class TextGCNDataset():
    """
    Dataset for TextGCN using transductive learning.

    All splits (train/val/test) share the same graph structure,
    but have different masks indicating which nodes belong to each split.
    """

    def __init__(self, artifact_path: str, split: str, x_type: str = "identity"):
        """
        Initialize TextGCN dataset.

        Args:
            artifact_path: Path to saved graph artifacts
            split: "train", "val", or "test"
            x_type: Node feature type ("identity", "glove", or "bert")
        """
        # Don't call super().__init__() - TextGCN doesn't use text sequences

        # Load graph artifacts (same for all splits)
        self.edge_index = torch.load(os.path.join(artifact_path, "ALL_edge_index.pt"))
        self.edge_attr = torch.load(os.path.join(artifact_path, "ALL_edge_attr.pt"))
        self.y = torch.load(os.path.join(artifact_path, "ALL_y.pt"))
        self.doc_mask = torch.load(os.path.join(artifact_path, "ALL_doc_mask.pt"))
        self.word_mask = torch.load(os.path.join(artifact_path, "ALL_word_mask.pt"))

        # Load split-specific mask
        self.split_mask = torch.load(os.path.join(artifact_path, f"{split}_mask.pt"))

        # Load metadata
        with open(os.path.join(artifact_path, "ALL_meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        self.vocab = meta["vocab"]
        self.num_classes = meta["num_classes"]
        self.num_nodes = meta["num_nodes"]
        self.num_docs = meta["num_docs"]
        self.num_words = meta["num_words"]
        self.label_to_idx = meta["label_to_idx"]
        self.idx_to_label = meta["idx_to_label"]
        self.x_type = x_type
        self.split = split

        # Set collate function
        self.collate_fn = textgcn_collate_fn

        print(f"Loaded TextGCN dataset for {split} split:")
        print(f"  - Total nodes: {self.num_nodes} (docs: {self.num_docs}, words: {self.num_words})")
        print(f"  - Nodes in {split} split: {self.split_mask.sum().item()}")
        print(f"  - Edges: {self.edge_index.shape[1]}")
        print(f"  - Classes: {self.num_classes}")

    def __len__(self):
        """Return 1 since we use transductive learning (single graph)."""
        return 1

    def __getitem__(self, idx):
        """
        Return PyTorch Geometric Data object containing the full graph.

        For transductive learning, the entire graph is used for all splits,
        with different masks indicating which nodes to use for training/validation/testing.
        """
        # Initialize node features based on x_type
        if self.x_type == "identity":
            # Identity matrix created on-the-fly in model (too large to store)
            x = None
        else:
            # Future: load pre-computed GloVe/BERT features
            # For now, use None and handle in model
            x = None

        # Create PyTorch Geometric Data object
        data = PyGData(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=self.y,  # Labels for all nodes (-1 for word nodes)
        )

        # Add custom masks as attributes
        data.doc_mask = self.doc_mask  # Boolean mask for document nodes
        data.word_mask = self.word_mask  # Boolean mask for word nodes
        data.split_mask = self.split_mask  # Boolean mask for this split

        return data


def textgcn_collate_fn(batch):
    """
    Collate function for TextGCN.

    Since we use transductive learning with a single graph,
    just return the single Data object.
    """
    # Batch contains only one element (the full graph)
    return batch[0]
