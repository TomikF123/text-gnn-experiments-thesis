import pandas as pd
import scipy.sparse as sp
from collections import defaultdict
from math import log
from tqdm import tqdm
from pathlib import Path


def load_dataset_csvs(
    dataset_path: str | Path, split: str | None = None
) -> pd.DataFrame:
    """
    If split is None: load and merge all CSVs found in `dataset_path`.
    If split is one of {"train","val","test"} (or any filename stem): load `<split>.csv`.
    """
    dataset_path = Path(dataset_path)

    if split is None:
        # Find all CSVs in the directory (non-recursive). Adjust to **/*.csv if you want recursive.
        csv_files = sorted(p for p in dataset_path.glob("*.csv") if p.is_file())
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in: {dataset_path}")

        # Read and concatenate
        frames = []
        for p in csv_files:
            df = pd.read_csv(p)
            frames.append(df)
        merged = pd.concat(frames, ignore_index=True)
        return merged

    else:
        # Load a specific split like train/val/test
        csv_file = dataset_path / f"{split}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV not found: {csv_file}")
        return pd.read_csv(csv_file)


# ----------------------------
# Public entry point
# ----------------------------
def build_text_graph_from_csv(
    dataset_path: str,
    text_col: str = "text",
    label_col: str = "label",
    split: str | None = None,
    window_size: int = 20,
) -> dict:
    """
    Build a TextGCN-style graph from CSV files.

    Constructs a heterogeneous graph with document and word nodes.
    Uses PMI (Pointwise Mutual Information) for word-word edges and
    TF-IDF for document-word edges.

    Args:
        dataset_path: Path to directory containing CSV files
        text_col: Name of text column in CSV (default: "text")
        label_col: Name of label column in CSV (default: "label")
        split: Load specific split CSV or None to merge all CSVs
        window_size: Sliding window size for word co-occurrence (default: 20)

    Returns:
        Dictionary with:
          - adj: scipy.sparse.csr_matrix adjacency matrix (symmetrized)
          - labels: list of labels (length = num_docs)
          - vocab: list of unique tokens
          - word_id_map: dict mapping words to IDs
          - docs: list of document texts
    """
    df = load_dataset_csvs(dataset_path=dataset_path, split=split)
    # Basic checks
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not in CSV.")
    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not in CSV.")

    # Prepare documents (simple whitespace tokenization; plug in your cleaner if needed)
    df_text = df[text_col].astype(str).tolist()
    doc_list = [" ".join(d.split()) for d in df_text]  # normalize spaces

    labels_list = df[label_col].tolist()

    # Build vocab and frequencies
    word_freq = _get_vocab(doc_list)
    vocab = list(word_freq.keys())
    word_id_map = {w: i for i, w in enumerate(vocab)}

    # For TF-IDF in doc-word edges
    words_in_docs, word_doc_freq = _build_word_doc_edges(doc_list)

    # Build adjacency (PMI for word-word; TF-IDF for doc-word)
    adj = _build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size)

    return {
        "adj": adj,  # csr_matrix(compressed sparse row), shape (num_docs+|V|, num_docs+|V|)
        "labels": labels_list,  # list, len = num_docs
        "vocab": vocab,  # list[str]
        "word_id_map": word_id_map,  # dict[str->int]
        "docs": doc_list,  # list[str]
    }


# ----------------------------
# Helpers (mostly your originals, tidied)
# ----------------------------
def _build_edges(doc_list, word_id_map, vocab, word_doc_freq, window_size=20):
    # Build sliding windows across all documents
    windows = []
    for doc_words in doc_list:
        words = doc_words.split()
        L = len(words)
        if L <= window_size:
            windows.append(words)
        else:
            for i in range(L - window_size + 1):
                windows.append(words[i : i + window_size])

    # Frequency of words appearing in windows
    word_window_freq = defaultdict(int)
    for window in windows:
        appeared = set()
        for w in window:
            if w not in appeared:
                word_window_freq[w] += 1
                appeared.add(w)

    # Word-pair co-occurrence counts (within windows)
    word_pair_count = defaultdict(int)
    for window in tqdm(windows, desc="Counting word pairs"):
        for i in range(1, len(window)):
            wi = window[i]
            wi_id = word_id_map.get(wi)
            if wi_id is None:
                continue
            for j in range(i):
                wj = window[j]
                wj_id = word_id_map.get(wj)
                if wj_id is None or wi_id == wj_id:
                    continue
                word_pair_count[(wi_id, wj_id)] += 1
                word_pair_count[(wj_id, wi_id)] += 1

    row, col, weight = [], [], []

    num_docs = len(doc_list)
    num_windows = len(windows)

    # PMI weights for word-word edges
    for (i, j), count in tqdm(word_pair_count.items(), desc="Computing PMI"):
        freq_i = word_window_freq[vocab[i]]
        freq_j = word_window_freq[vocab[j]]
        p_ij = count / num_windows
        p_i = freq_i / num_windows
        p_j = freq_j / num_windows
        # Avoid log of zero; skip non-positive PMI
        denom = p_i * p_j if p_i > 0 and p_j > 0 else 0.0
        if denom <= 0.0:
            continue
        pmi = log(p_ij / denom)
        if pmi <= 0:
            continue
        # Offset by num_docs so word nodes start after doc nodes
        row.append(num_docs + i)
        col.append(num_docs + j)
        weight.append(pmi)

    # Doc-word TF-IDF edges
    doc_word_freq = defaultdict(int)
    for di, doc_words in enumerate(doc_list):
        for w in doc_words.split():
            wid = word_id_map[w]
            doc_word_freq[(di, wid)] += 1

    for di, doc_words in enumerate(doc_list):
        seen = set()
        for w in doc_words.split():
            if w in seen:
                continue
            wid = word_id_map[w]
            tf = doc_word_freq[(di, wid)]
            idf = log(num_docs / max(1, word_doc_freq[vocab[wid]]))
            row.append(di)
            col.append(num_docs + wid)
            weight.append(tf * idf)
            seen.add(w)

    n_nodes = num_docs + len(vocab)
    adj_mat = sp.csr_matrix((weight, (row, col)), shape=(n_nodes, n_nodes))
    # Symmetrize: adj = A + A^T - min(A, A^T)
    adj = (
        adj_mat
        + adj_mat.T.multiply(adj_mat.T > adj_mat)
        - adj_mat.multiply(adj_mat.T > adj_mat)
    )
    return adj.tocsr()


def _get_vocab(text_list):
    freq = defaultdict(int)
    for doc in text_list:
        for w in doc.split():
            freq[w] += 1
    return freq


def _build_word_doc_edges(doc_list):
    """Calculate word-document frequencies for TF-IDF computation."""
    words_in_docs = defaultdict(set)
    for di, doc in enumerate(doc_list):
        for w in doc.split():
            words_in_docs[w].add(di)
    word_doc_freq = {w: len(dset) for w, dset in words_in_docs.items()}
    return words_in_docs, word_doc_freq
