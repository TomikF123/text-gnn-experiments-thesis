#!/usr/bin/env python
"""Debug model initialization to find memory explosion.

NOTE: btop shows ~15GB increase vs psutil's 6.4GB due to:
- Memory fragmentation
- OS-level caching
- Copy-on-write behavior
Real culprit is likely the same - need to find what allocates 6-15GB!
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import psutil
import torch
import json

def get_memory_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_mem(label):
    mem = get_memory_mb()
    print(f"[{mem:>8.1f} MB] {label}")
    return mem

print("=" * 80)
print("DEBUGGING MODEL INITIALIZATION")
print("=" * 80)
print("NOTE: btop may show higher (15GB) due to OS-level memory management")
print("=" * 80)

# Load config
with open("runConfigs/testGCN.json") as f:
    config_dict = json.load(f)

from textgnn.config_class import ModelConfig, DatasetConfig

dataset_config = DatasetConfig(**config_dict["dataset"])
model_config = ModelConfig(**config_dict["model_config"])

# Get metadata
import pickle
dataset_dir = "saved/20ng-train-70-val-20-test-10-stop-words-remove-true-rare-words-remove-50-vocab-size-none-v2/text-gcn-x-identity-window-20"
with open(f"{dataset_dir}/ALL_meta.pkl", "rb") as f:
    meta = pickle.load(f)

num_nodes = meta["num_nodes"]
num_classes = meta["num_classes"]

print(f"\nModel parameters:")
print(f"  num_nodes: {num_nodes}")
print(f"  num_classes: {num_classes}")
print(f"  hidden_dims: {model_config.model_specific_params['hidden_dims']}")

print("\n" + "=" * 80)
print("STEP-BY-STEP MODEL CREATION")
print("=" * 80)

mem_start = print_mem("\n0. Before any imports")

from textgnn.models.textgcn.model import TextGCNClassifier
from textgnn.models.base_text_classifier import GraphTextClassifier
mem_after_import = print_mem("1. After importing classes")

print("\n2. Checking what GraphTextClassifier base class does...")
print(f"   It will be initialized with:")
print(f"   → vocab_size={num_nodes}")
print(f"   → embedding_dim={num_nodes}")
print(f"   → output_size={num_classes}")

print(f"\n   CRITICAL: If base class creates nn.Embedding:")
print(f"   → torch.nn.Embedding({num_nodes}, {num_nodes})")
print(f"   → Size: {num_nodes} * {num_nodes} * 4 bytes = {num_nodes * num_nodes * 4 / 1024**3:.2f} GB!")
print(f"   → This would explain the 6-15GB memory jump!")

# Create model
print("\n3. Creating TextGCNClassifier (watch memory carefully)...")
mem_before_model = print_mem("   Before model creation")

model = TextGCNClassifier(
    num_nodes=num_nodes,
    num_classes=num_classes,
    hidden_dims=model_config.model_specific_params.get("hidden_dims", [200]),
    x_type="identity",
    pred_type=model_config.model_specific_params.get("pred_type", "softmax"),
    act=model_config.model_specific_params.get("act", "relu"),
    use_bn=model_config.model_specific_params.get("use_bn", True),
    dropout=model_config.model_specific_params.get("dropout", 0.5),
    use_edge_weights=model_config.model_specific_params.get("use_edge_weights", True),
)

mem_after_model = print_mem("   After model creation")
print(f"\n   → Memory increase (psutil): {mem_after_model - mem_before_model:.1f} MB")
print(f"   → Expected from btop: ~{(mem_after_model - mem_before_model) * 2.3:.1f} MB")

# Inspect model parameters
print("\n" + "=" * 80)
print("MODEL PARAMETERS (searching for culprit)")
print("=" * 80)

total_params = 0
largest_params = []

for name, param in model.named_parameters():
    param_size_mb = param.numel() * param.element_size() / 1024 / 1024
    total_params += param.numel()
    largest_params.append((param_size_mb, name, param.shape))

# Sort by size
largest_params.sort(reverse=True)

print("\nTop 20 largest parameters:")
for size_mb, name, shape in largest_params[:20]:
    if size_mb > 1:  # Only show params > 1 MB
        print(f"  {name:50s} {str(shape):30s} {size_mb:>10.2f} MB")

print(f"\nTotal parameters: {total_params:,}")
print(f"Expected memory: {total_params * 4 / 1024 / 1024:.2f} MB (float32)")

# Check for embedding layer (THE LIKELY CULPRIT)
print("\n" + "=" * 80)
print("CHECKING FOR EMBEDDING LAYER (LIKELY CULPRIT)")
print("=" * 80)

found_embedding = False
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Embedding):
        found_embedding = True
        size_gb = module.num_embeddings * module.embedding_dim * 4 / 1024**3
        print(f"\n🚨 FOUND EMBEDDING LAYER: {name}")
        print(f"   → num_embeddings: {module.num_embeddings:,}")
        print(f"   → embedding_dim: {module.embedding_dim:,}")
        print(f"   → Total elements: {module.num_embeddings * module.embedding_dim:,}")
        print(f"   → Memory: {size_gb:.2f} GB")
        print(f"\n   THIS IS THE PROBLEM! TextGCN doesn't need an embedding layer!")
        print(f"   It uses sparse identity matrix as features, not learned embeddings!")

if not found_embedding:
    print("No embedding layer found.")
    print("\nChecking all modules:")
    for name, module in model.named_modules():
        print(f"  {type(module).__name__:30s} {name}")

print("\n" + "=" * 80)
print(f"Final memory: {get_memory_mb():.1f} MB")
print("=" * 80)
