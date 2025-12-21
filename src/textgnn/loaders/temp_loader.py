# datasets/text_graph_csv.py
from __future__ import annotations
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix

from textgnn.loaders.build_graph import build_text_graph_from_csv  # your builder
from textgnn.loaders.create_basic_dataset import create_basic_dataset
from textgnn.config_class import DatasetConfig

# loaders/gnn_dataset.py
# from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

from textgnn.loaders.build_graph import build_text_graph_from_csv
from textgnn.loaders.build_graph import build_text_graph_from_csv
import pickle
import os

PRESETS = {"text_gcn": {}, "gat": {}}


def _load_artifacts(dataset_save_path: str, split: str, window_size: int = 20):
    art = build_text_graph_from_csv(
        dataset_path=dataset_save_path,
        text_col="text",
        label_col="label",
        split=split,
        window_size=window_size,
    )
    adj: sp.csr_matrix = art["adj"]
    labels_list = art["labels"]
    vocab = art["vocab"]
    word_id_map = art["word_id_map"]
    num_docs = len(labels_list)
    num_words = len(vocab)
    return adj, labels_list, vocab, word_id_map, num_docs, num_words


def _build_edges(adj: sp.csr_matrix):
    # Returns CPU tensors; caller decides device
    edge_index, edge_attr = from_scipy_sparse_matrix(adj)
    return edge_index, edge_attr


def _build_node_features(x_type: str, N: int):
    # AFTER (Option A):
    if x_type == "identity":
        x = None  # <- no identity matrix saved/used
    else:
        # keep whatever other feature options you add later (TF-IDF/embeddings)
        x = None


def _encode_labels(labels_list, num_docs: int, N: int):
    # Returns y (size N), and class mapping if needed
    if isinstance(labels_list[0], str):
        classes = sorted(set(labels_list))
        cls2id = {c: i for i, c in enumerate(classes)}
        y_docs = torch.tensor([cls2id[c] for c in labels_list], dtype=torch.long)
    else:
        cls2id = None
        y_docs = torch.tensor(labels_list, dtype=torch.long)

    y = torch.full((N,), -1, dtype=torch.long)
    y[:num_docs] = y_docs
    return y, cls2id


def _build_masks(num_docs: int, N: int):
    doc_mask = torch.zeros(N, dtype=torch.bool)
    doc_mask[:num_docs] = True
    word_mask = ~doc_mask
    return doc_mask, word_mask


def _assemble_data(x, edge_index, edge_attr, y, doc_mask, word_mask):
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.doc_mask = doc_mask
    data.word_mask = word_mask
    return data


def create_gnn_artifacts(
    dataset_save_path: str,
    dataset_config: DatasetConfig,
    full_path: str,
    missing_parent: bool = False,
):
    """
    Create GNN-specific artifacts.

    Args:
        dataset_save_path: Path to base dataset
        dataset_config: Pydantic DatasetConfig model
        full_path: Path for GNN-specific artifacts
        missing_parent: Whether to create base dataset first
    """
    if missing_parent:
        create_basic_dataset(
            dataset_config=dataset_config, dataset_save_path=dataset_save_path
        )

    inductive = False

    split = None if not inductive else 0

    if inductive:
        pass
    else:
        gnn_encoding = dataset_config.gnn_encoding
        adj, labels_list, vocab, word_id_map, num_docs, num_words = _load_artifacts(
            dataset_save_path=dataset_save_path,
            split=split,
            window_size=gnn_encoding.window_size if gnn_encoding.window_size is not None else 20,
        )
        N = num_docs + num_words

        # 2) Build CPU tensors
        edge_index, edge_attr = _build_edges(adj)
        x_type = gnn_encoding.x_type if gnn_encoding.x_type is not None else "identity"
        x = _build_node_features(N=N, x_type=x_type)
        y, cls2id = _encode_labels(labels_list, num_docs, N)
        doc_mask, word_mask = _build_masks(num_docs, N)

        meta = {
            "labels_list": labels_list,
            "class_to_id": cls2id,  # helpful later for decoding predictions
            "vocab": vocab,
            "word_id_map": word_id_map,
            "num_docs": num_docs,
            "num_words": num_words,
        }

    os.makedirs(full_path, exist_ok=True)

    prefix = "ALL" if split is None else split.upper()

    torch.save(edge_index, os.path.join(full_path, f"{prefix}_edge_index.pt"))
    torch.save(edge_attr, os.path.join(full_path, f"{prefix}_edge_attr.pt"))
    # torch.save(x, os.path.join(full_path, f"{prefix}_x.pt")) #TODO ???
    torch.save(y, os.path.join(full_path, f"{prefix}_y.pt"))
    torch.save(doc_mask, os.path.join(full_path, f"{prefix}_doc_mask.pt"))
    torch.save(word_mask, os.path.join(full_path, f"{prefix}_word_mask.pt"))

    with open(os.path.join(full_path, f"{prefix}_meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print(f"Saved {prefix} artifacts to {full_path}")


def load_gnn_artifacts(full_path: str, split: str = None):
    prefix = "ALL" if split is None else split.upper()

    edge_index = torch.load(f"{full_path}/{prefix}_edge_index.pt", map_location="cpu")
    edge_attr = torch.load(f"{full_path}/{prefix}_edge_attr.pt", map_location="cpu")
    #    x = torch.load(f"{full_path}/{prefix}_x.pt", map_location="cpu") # TODO identity matrix is useless to be saved
    x = None  # <- no identity matrix saved/used
    y = torch.load(f"{full_path}/{prefix}_y.pt", map_location="cpu")
    doc_mask = torch.load(f"{full_path}/{prefix}_doc_mask.pt", map_location="cpu")
    word_mask = torch.load(f"{full_path}/{prefix}_word_mask.pt", map_location="cpu")

    with open(f"{full_path}/{prefix}_meta.pkl", "rb") as f:
        meta = pickle.load(f)

    # Assemble PyG Data object
    data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.doc_mask = doc_mask
    data.word_mask = word_mask
    # prevent dtype mismatch
    if getattr(data, "edge_attr", None) is not None:
        data.edge_attr = data.edge_attr.to(torch.float32)

    data.y = data.y.to(torch.long)
    data.doc_mask = data.doc_mask.to(torch.bool)
    data.word_mask = data.word_mask.to(torch.bool)

    return data, meta


def get_gnn_dataset_object(
    model_type: str,
    dataset_save_path: str,
    full_path: str,
    dataset_config: DatasetConfig,
    split: str,
):
    """
    Get GNN dataset object.

    Args:
        model_type: Model type
        dataset_save_path: Path to base dataset
        full_path: Path to GNN artifacts
        dataset_config: Pydantic DatasetConfig model
        split: train/val/test

    Returns:
        GNN data object
    """
    split = None
    preset_name = dataset_config.preset if hasattr(dataset_config, 'preset') else None
    if preset_name is not None:
        return None
    inductive = False

    data, meta = load_gnn_artifacts(full_path=full_path, split=split)
    return data


def create_gnn_filename(model_type: str, dataset_config: DatasetConfig) -> str:
    """
    Create filename for GNN artifacts.

    Args:
        model_type: Model type
        dataset_config: Pydantic DatasetConfig model

    Returns:
        Filename string
    """
    gnn_encoding = dataset_config.gnn_encoding
    x_type = gnn_encoding.x_type if gnn_encoding.x_type is not None else "missing"
    window_size = gnn_encoding.window_size if gnn_encoding.window_size is not None else "missing"
    model_type = model_type
    parts = [x_type, "test", model_type, window_size]
    return "_".join(str(p) for p in parts)


if __name__ == "__main__":
    # Example usage
    dataset_config = {
        "name": "mr",
        "preprocess": {
            "remove_stopwords": True,
            "remove_rare_words": 0,
        },
        "tvt_split": [0.8, 0, 0.2],
        "random_seed": 42,
        "encoding": {
            "x_type": "identity",
            "window_size": 20,
        },
    }
    dataset_save_path = "./data/example_dataset"
    create_text_gnn_dataset(
        dataset_config, dataset_save_path, full_path="./data/example_dataset.csv"
    )
