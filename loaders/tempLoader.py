# datasets/text_graph_csv.py
from __future__ import annotations
from pathlib import Path
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_scipy_sparse_matrix

from loaders.build_graph import build_text_graph_from_csv  # your builder
from loaders.create_basic_dataset import create_basic_dataset

# loaders/gnn_dataset.py
# from __future__ import annotations
from pathlib import Path
from typing import Optional, Literal, Dict, Any, Tuple

import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from torch_geometric.utils import from_scipy_sparse_matrix
import scipy.sparse as sp

from loaders.build_graph import build_text_graph_from_csv
from loaders.build_graph import build_text_graph_from_csv
import pickle
import os


class GraphTextDataset:
    """
    Thin wrapper around a PyG Data graph built from CSVs (TextGCN/GAT style).
    - data: PyG Data with (x, edge_index, edge_attr, y, doc_mask, word_mask)
    - meta: labels_list, vocab, word_id_map, counts
    """

    def __init__(
        self,
        dataset_path: str | Path,
        split: Optional[str],  # None for merged (transductive), or "train"/"val"/"test"
        window_size: int = 20,
        x_type: Optional[Literal["identity"]] = "identity",
        device: Optional[str | torch.device] = None,
        cache_dir: Optional[
            str | Path
        ] = None,  # e.g. saved/<dataset_key>/models/<model_key>/
        cache_tag: Optional[str] = None,  # e.g. text_gcn_seed42_win20
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.window_size = window_size
        self.x_type = x_type
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.cache_tag = cache_tag

        self.data: Optional[PyGData] = None
        self.meta: Dict[str, Any] = {}

        if self.cache_dir and self.cache_tag and self._try_load_cache():
            return

        self._build()
        if self.cache_dir and self.cache_tag:
            self._save_cache()

    # -------- public API --------
    def get_data(self) -> PyGData:
        assert self.data is not None
        return self.data

    def get_meta(self) -> Dict[str, Any]:
        return self.meta

    # -------- build --------
    def _build(self) -> None:
        art = build_text_graph_from_csv(
            dataset_path=str(self.dataset_path),
            text_col="text",
            label_col="label",
            split=self.split,  # None (merged) or "train"/"val"/"test"
            window_size=self.window_size,
        )

        adj: sp.csr_matrix = art["adj"]
        labels_list = art["labels"]
        vocab = art["vocab"]
        word_id_map = art["word_id_map"]

        num_docs = len(labels_list)
        num_words = len(vocab)
        N = num_docs + num_words

        # CSR -> edge_index/edge_attr
        edge_index, edge_attr = from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)

        # x features
        if self.x_type == "identity":
            idx = torch.arange(N, device=self.device, dtype=torch.long)
            x = torch.sparse_coo_tensor(
                indices=torch.stack([idx, idx], 0),
                values=torch.ones(N, device=self.device),
                size=(N, N),
                device=self.device,
            ).coalesce()
        else:
            x = None  # extend later with TF-IDF/embeddings

        # y (labels only for doc nodes)
        if isinstance(labels_list[0], str):
            classes = sorted(set(labels_list))
            map_ = {c: i for i, c in enumerate(classes)}
            y_docs = torch.tensor(
                [map_[c] for c in labels_list], dtype=torch.long, device=self.device
            )
        else:
            y_docs = torch.tensor(labels_list, dtype=torch.long, device=self.device)

        y = torch.full((N,), -1, dtype=torch.long, device=self.device)
        y[:num_docs] = y_docs

        # masks
        doc_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        doc_mask[:num_docs] = True
        word_mask = ~doc_mask

        data = PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.doc_mask = doc_mask
        data.word_mask = word_mask

        self.data = data
        self.meta = {
            "labels_list": labels_list,
            "vocab": vocab,
            "word_id_map": word_id_map,
            "num_docs": num_docs,
            "num_words": num_words,
        }

    # -------- caching --------
    def _cache_paths(self) -> Tuple[Path, Path]:
        split_tag = self.split if self.split is not None else "all"
        base = self.cache_dir / f"{self.cache_tag}__{split_tag}"
        return base.with_suffix(".pt"), base.with_suffix(".meta.pt")

    def _try_load_cache(self) -> bool:
        data_pt, meta_pt = self._cache_paths()
        if data_pt.exists() and meta_pt.exists():
            self.data = torch.load(data_pt, map_location=self.device)
            self.meta = dict(torch.load(meta_pt, map_location="cpu"))
            return True
        return False

    def _save_cache(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        data_pt, meta_pt = self._cache_paths()
        torch.save(self.data, data_pt)
        torch.save(self.meta, meta_pt)


# ========== FACTORIES (mirror your LSTM pattern) ==========


def create_text_gcn_dataset(
    dataset_config: dict,
    model_config: dict,
    dataset_path: str | Path,
    cache_dir: Optional[str | Path] = None,
) -> GraphTextDataset:
    """
    Transductive: one merged graph (split=None), training uses masks externally.
    """
    window_size = model_config.get("graph", {}).get("window_size", 20)
    x_type = model_config.get("graph", {}).get("x_type", "identity")
    cache_tag = model_config.get("cache_tag", f"text_gcn_win{window_size}")
    device = model_config.get("device", "cpu")

    ds = GraphTextDataset(
        dataset_path=dataset_path,
        split=None,  # merged graph
        window_size=window_size,
        x_type=x_type,
        device=device,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )
    return ds


def create_gnn_artifacts(
    dataset_save_path: str,
    dataset_config: dict,
    full_path: str,
    missing_parrent: bool = False,
):
    if missing_parrent:
        create_basic_dataset(
            dataset_config=dataset_config, dataset_save_path=dataset_save_path
        )
    return
    # TODO: ?? move artifacts creation from dataset class to here.
    artifacts = build_text_graph_from_csv(dataset_path=dataset_save_path)
    os.makedirs(full_path, exist_ok=True)
    for key, value in artifacts.items():
        filename = f"{key}.pkl"
        with open(os.path.join(full_path, filename), "wb") as f:
            pickle.dump(value, f)


def create_gat_inductive_dataset(
    dataset_config: dict,
    model_config: dict,
    dataset_path: str | Path,
    split: Literal["train", "val", "test"],
    cache_dir: Optional[str | Path] = None,
) -> GraphTextDataset:
    """
    Inductive: build a graph per split. IMPORTANT:
      Ensure your build_text_graph_from_csv uses TRAIN-ONLY stats (PMI/IDF)
      for val/test to avoid leakage (adjust builder accordingly).
    """
    window_size = model_config.get("graph", {}).get("window_size", 20)
    x_type = model_config.get("graph", {}).get("x_type", "identity")
    cache_tag = model_config.get("cache_tag", f"text_gat_win{window_size}")
    device = model_config.get("device", "cpu")

    ds = GraphTextDataset(
        dataset_path=dataset_path,
        split=split,  # per-split graph
        window_size=window_size,
        x_type=x_type,
        device=device,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )
    return ds


def get_gnn_dataset_object(
    model_type: str,
    dataset_save_path: str,
    full_path: str,
    dataset_config: dict,
    split: str,
) -> GraphTextDataset:
    if model_type == "text_gcn":
        pass
    elif model_type == "gat":
        pass
    else:
        raise ValueError(f"Unsupported GNN model type: {model_type}")
    return GraphTextDataset(
        split=None,
        dataset_path=dataset_save_path,
        window_size=dataset_config["gnn_encoding"].get("window_size", 20),
        x_type=dataset_config["gnn_encoding"].get("x_type", "identity"),
        device=dataset_config.get("device", "cpu"),
        cache_dir=full_path,
        cache_tag=create_gnn_filename(model_type, dataset_config),
    )
    return "idk what to return here, need to implement the dataset object"


def create_gnn_filename(model_type: str, dataset_config: dict) -> str:
    print(dataset_config.keys())
    print(dataset_config["gnn_encoding"])
    print(dataset_config["preprocess"].keys())
    x_type = dataset_config["gnn_encoding"].get("x_type", "missing")
    window_size = dataset_config["gnn_encoding"].get("window_size", "missing")
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
