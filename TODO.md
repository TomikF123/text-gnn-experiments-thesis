Here’s a compact **TODO.md** draft you can drop straight in:

---

# TODO – Model & Codebase Refactor

## 1. Registry & Imports

* [ ] Decide between **lazy import** (string paths + `get_function_from_path`) or **code-level registry** (real imports in `registry.py`).
* [ ] If moving to code-level registry:

  * [ ] Create `registry.py` at repo root with separate dicts: `MODEL_CREATORS`, `DATASET_CREATORS`, `TRAINING_LOOPS` (optional), etc.
  * [ ] Remove `get_function_from_path` and string path registries once migrated.

## 2. Model Architecture Unification

* [ ] Create a **generic `TextGraphClassifier`**:

  * Accepts a list of conv layers (`[("gcn", 128), ("gat", 64)]`, etc.).
  * Uses PyG layers (`GCNConv`, `GATConv`, `SAGEConv`, …) from a registry.
  * Adds classifier head (Linear or optional MLP).
  * Keeps **loss in training loop**, not in `forward`.
* [ ] Implement config-driven creation (conv type, dims, dropout, BN, activation).
* [ ] Remove dataset-specific coupling from GNN `forward()` (use masks in loop).

## 3. MLP Utility Class

* [ ] Create a small `MLP` helper:

  * Variable hidden layers & dims.
  * Consistent init, activation, dropout, BN.
  * Use in classifier heads instead of hand-rolling `nn.Sequential`.

## 4. Dataset Handling

* [ ] Add `GraphTextDataset` for PyG `Data` objects.
* [ ] Support both:

  * **Transductive** (TextGCN) → one big graph, masks for train/val/test.
  * **Inductive** (SAGE/GAT) → neighbor sampling / per-batch graphs.
* [ ] Refactor `build_graph.py`:

  * Remove file/path heuristics (`*_sentences_clean.txt`).
  * Use your `prepData.clean_data` tokens directly.
  * Return `Data` object instead of custom `TextDataset`.

## 5. Weight Initialization

* [ ] Add `reset_parameters()` method to each model:

  * Default: Xavier uniform (gain from activation).
  * Optional config override (`init.scheme`, `init.act`, `init.embedding_policy`).
* [ ] Call `reset_parameters()` in `__init__`.
* [ ] Keep config simple — no per-layer micro-control unless needed.

## 6. Training Loop Consistency

* [ ] Pick **one** approach:

  * Generic loop with `training_step()` hooks in models, OR
  * Model-specific loop from registry (but not both).
* [ ] Ensure `collate_fn` comes from dataset object; remove unused `COLLATE_FN_CREATORS`.

## 7. Memory Optimization (Thesis Focus)

* [ ] Investigate PyG **NeighborLoader**, GraphSAINT, or Cluster-GCN for large graphs.
* [ ] Implement mixed precision (AMP) for GNN training.
* [ ] Explore activation checkpointing for deep GNNs.
* [ ] Reduce feature sizes where possible; consider shared embeddings.

---

Do you want me to also make you a **visual architecture map** of how the new registry + dataset + model creation flow will connect? That way, when you refactor, you can follow the arrows.
