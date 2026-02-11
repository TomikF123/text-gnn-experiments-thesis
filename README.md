# text-gnn-experiments-thesis

Text classification with LSTM, TextGCN, and TextING models.

## Installation

```bash
git clone https://github.com/TomikF123/text-gnn-experiments-thesis
cd text-gnn-experiments-thesis
uv sync
source .venv/bin/activate
```

Or with pip:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Note:** This project uses PyTorch 2.3.0 with CUDA 12.1 (compute capability 5.0-9.0). If you have a different GPU/CUDA version, install the appropriate PyTorch version from https://pytorch.org/get-started/locally/

## Download Data

```bash
python src/textgnn/download_data.py
```

Downloads:
- Movie Review (MR) dataset
- 20 Newsgroups dataset
- NLTK stopwords
- GloVe embeddings (6B)

## Run Experiments

Single experiment:
```bash
python main.py --config runConfigs/experiments/baseline/lstm_20ng_baseline.json
```

All experiments:
```bash
python run_experiments.py
```

Options:
- `--timeout 1800` - timeout per experiment (default 30min)
- `--no-resume` - ignore completed runs
- `--filter baseline` - only run matching configs

Results tracked in `mlruns/` (MLflow).

## View Results

```bash
mlflow ui
```

Opens the MLflow dashboard at http://localhost:5000 to view experiment metrics and compare runs.

## Project Structure

```
main.py               # Single experiment entry point
run_experiments.py    # Batch experiment runner
src/textgnn/          # Main source code
  models/             # Model implementations (LSTM, TextGCN, TextING)
  loaders/            # Data loaders for each model
  pipelines/          # Training pipelines
runConfigs/           # Experiment configuration files
data/                 # Downloaded datasets
mlruns/               # MLflow experiment tracking
saved/                # Cached preprocessing results
```
