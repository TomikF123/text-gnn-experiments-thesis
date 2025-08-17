
# text-gnn-experiments-thesis

A text classification framework supporting multiple deep learning architectures including LSTM, FastText, and Graph Neural Networks. This repository contains the practical implementation for text classification experiments, developed as part of a bachelor thesis. **Note: The project is still under development, and some components are incomplete.**

---

## Features

* **Multiple Model Architectures**: LSTM, FastText, Text-GCN, and Text-GNN support.
* **Preprocessing Pipeline**: Configurable text preprocessing with stopword removal and vocabulary filtering.
* **Embedding Support**: Integration with GloVe embeddings and custom token encoding.
* **Experiment Management**: MLflow integration for experiment tracking and model versioning.
* **Configuration-Driven**: JSON-based configuration system for easy experiment setup.
* **Modular Design**: Clear separation between data loading, model creation, and training.

---

## Project Structure (Partial Overview)

```
experiments/
├── data/                     # Dataset files and GloVe embeddings
├── models/                   # Model implementations
│   ├── lstm/                 # LSTM model and training
│   ├── fastText/             # FastText implementation (work in progress)
│   ├── text_gcn/             # Text-GCN graph neural network (work in progress)
│   ├── text_gnn.py           # Text-GNN implementation (work in progress)
│   └── mlp.py                # MLP utility class used in other models
├── loaders/                  # Model-specific preprocessing, encoding, and artifacts
├── runConfigs/               # JSON configuration files
├── saved/                    # Cached datasets and graph/model artifacts
├── mlruns/                   # MLflow experiment tracking
├── modelFactory.py           # Model creation and selection logic
├── loadData.py               # Dataset loading and mapping
├── prepData.py               # Preprocessing scripts
├── main.py                   # Main training script
├── train.py                  # Training loop implementation
├── eval.py                   # Model evaluation
└── utils.py                  # Utility functions
```

> **Note:** This is a partial overview; the project structure may include additional files and directories.

---

## Installation

1. **Clone the repository**:

```bash
git clone https://github.com/TomikF123/text-gnn-experiments-thesis
cd experiments
```

2. **Create a virtual environment**:

```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Download datasets, embeddings, and stopwords**:

```bash
python downloadData.py
```

---

## Supported Datasets

* **20 Newsgroups (20ng)**: Multi-class news classification.
* **Movie Review (mr)**: Binary sentiment analysis.
* **Custom datasets**: Add your own CSV files to the `data/` directory.

---

## Model Architectures

### LSTM

* Bidirectional LSTM with optional attention.
* Configurable hidden dimensions and layers.
* Dropout and batch normalization support.

### FastText - TODO

* Subword-aware text classification.
* Efficient training and inference.
* Implementation is still a work in progress.

### Text-GCN  - TODO

* Graph Convolutional Networks for text classification.
* Transductive learning on document-word graphs.
* Built on PyTorch Geometric.
* Implementation is in progress.

---

## Configuration

Experiments are configured using JSON files in the `runConfigs/` directory. Example:

```json
{
  "experiment_name": "test",
  "run_name": "test_run",
  "dataset": {
    "name": "mr",
    "tvt_split": [0.9, 0, 0.1],
    "shuffle": true,
    "random_seed": 123,
    "preprocess": {
      "remove_stopwords": false,
      "remove_rare_words": 0
    },
    "rnn_encoding": {
      "encode_token_type": "glove",
      "tokens_trained_on": 6,
      "embedding_dim": 300
    }
  },
  "model_config": {
    "model_type": "lstm",
    "common_params": {
      "epochs": 15,
      "batch_size": 32,
      "seed": 123
    },
    "model_specific_params": {
      "hidden_dim": 128,
      "dropout": 0.5,
      "num_layers": 2,
      "bidirectional": true,
      "embedding_dim": 300,
      "lr": 0.001
    }
  }
}
```

---

## Usage

### Training

```bash
# Train LSTM on MR dataset
python main.py --config test.json

# Train Text-GCN on 20NG dataset
python main.py --config testGCN.json
```

## TODO

* Complete FastText implementation.
* Complete Text-GCN and Text-GNN implementations.
* Improve modularity for additional GNN architectures (GAT, GraphSAGE, etc.).
* Add full documentation and code comments.
* Implement neighbor sampling and lazy loading for large inductive graphs.

---
