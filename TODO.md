Here’s a compact **TODO.md** draft you can drop straight in:

---

## 0. - Provide user guide to choosing the right torch and troch_geometric - given their gpu/gpus and their compute capability (CC)
https://docs.nvidia.com/cuda/archive/12.1.0/cuda-toolkit-release-notes/index.html?utm_source=chatgpt.com
https://developer.nvidia.com/cuda-legacy-gpus?utm_source=chatgpt.com
# TODO – Model & Codebase Refactor

## 1. Registry & Imports???
 - idea: create a registry file and a config.py file. This config.py file reads the registry file and creates and the import dicts (ARTIFACT_CREATORS, DATASETS..) automaticaly, other modules will import these objects instead of storing them. The registry file could be a yaml file with list of implemented models (just names i guess).


## 2. further addons to the hyperparameter space?
  - include optimizers, loss function in run config - model.common_params
  - include regularization techniques as well - 

## 4. Dataset Handling - graph datasets

* [ ] Add `GraphTextDataset` for PyG `Data` objects.
* [ ] Support both:

  * **Transductive** (TextGCN) → one big graph, masks for train/val/test.
  * **Inductive** (SAGE/GAT) → neighbor sampling / per-batch graphs.

## 5. Weight Initialization = overkill?

* [ ] Add `reset_parameters()` method to each model:

  * Default: Xavier uniform (gain from activation).
  * Optional config override (`init.scheme`, `init.act`, `init.embedding_policy`).
* [ ] Call `reset_parameters()` in `__init__`.
* [ ] Keep config simple — no per-layer micro-control unless needed.

## 6. non specifc stuf
   -  Implement basic train function in textgnn.train.py
   - implement the text_gcn.model correctly


---