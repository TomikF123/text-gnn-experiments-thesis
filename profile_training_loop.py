#!/usr/bin/env python
"""
Profile memory during actual training to find periodic spikes.
This will show EXACTLY when memory jumps from 1GB → 11GB.
"""

import sys
import os
import gc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import json
from textgnn.config_class import Config
from textgnn.load_data import load_data
from textgnn.model_factory import create_model
from torch.utils.data import DataLoader

def get_memory_allocated_mb():
    """Get PyTorch CUDA memory allocated (if GPU) or estimate CPU memory."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    else:
        # For CPU, track tensor memory manually
        total = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj):
                    total += obj.element_size() * obj.nelement()
            except:
                pass
        return total / 1024 / 1024

def print_mem(label, step=None):
    """Print memory usage."""
    mem_mb = get_memory_allocated_mb()
    if step is not None:
        print(f"[Step {step:3d}] [{mem_mb:>8.1f} MB] {label}")
    else:
        print(f"[{mem_mb:>8.1f} MB] {label}")
    return mem_mb

print("=" * 80)
print("PROFILING TRAINING LOOP MEMORY")
print("=" * 80)
print("This will identify what causes 1GB → 11GB spikes during training")
print("=" * 80)

# Load config
with open("runConfigs/testGCN.json") as f:
    config_dict = json.load(f)

# Fix key name for pydantic (model_config → model_conf)
if "model_config" in config_dict:
    config_dict["model_conf"] = config_dict.pop("model_config")

config = Config(**config_dict)

print("\n--- Loading datasets ---")
train_dataset = load_data(
    dataset_config=config.dataset,
    model_type=config.model_conf.model_type,
    split="train"
)

train_loader = DataLoader(
    train_dataset,
    batch_size=1,
    collate_fn=train_dataset.collate_fn,
    shuffle=False
)

print("\n--- Creating model ---")
mem_before_model = print_mem("Before model creation")
model = create_model(
    model_config=config.model_conf,
    dataset_config=config.dataset
)
mem_after_model = print_mem("After model creation")
print(f"→ Model creation memory: {mem_after_model - mem_before_model:.1f} MB")

# Move to device
device = torch.device("cpu")  # Use CPU for consistent profiling
model = model.to(device)
print_mem(f"After moving to {device}")

# Create optimizer
lr = config.model_conf.model_specific_params.get("lr", 0.02)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
mem_after_optimizer = print_mem("After creating optimizer")

# Check optimizer state size
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel parameters: {total_params:,}")
print(f"Optimizer (Adam) overhead: ~2x parameters for momentum/variance")
print(f"Expected optimizer memory: ~{total_params * 2 * 4 / 1024 / 1024:.1f} MB")

criterion = torch.nn.CrossEntropyLoss()

print("\n" + "=" * 80)
print("TRAINING LOOP PROFILING")
print("=" * 80)

# Get single batch
data = next(iter(train_loader))
data = data.to(device)

print(f"\nData properties:")
print(f"  edge_index: {data.edge_index.shape}")
print(f"  edge_attr: {data.edge_attr.shape}")
print(f"  y: {data.y.shape}")
print(f"  doc_mask: {data.doc_mask.sum()} / {data.doc_mask.shape[0]} nodes")
print(f"  train split_mask: {data.split_mask.sum()} / {data.doc_mask.sum()} doc nodes")

train_nodes = data.doc_mask & data.split_mask
num_train = train_nodes.sum().item()
print(f"  Training on {num_train} nodes")

# Simulate multiple training steps
num_steps = 10
print(f"\n--- Running {num_steps} training steps (watch for spikes!) ---\n")

mem_history = []

for step in range(num_steps):
    print(f"\n{'='*60}")
    print(f"STEP {step + 1}")
    print(f"{'='*60}")

    mem_start_step = print_mem("Start of step", step + 1)

    # Forward pass
    print("  [1] Forward pass...")
    model.train()
    mem_before_forward = get_memory_allocated_mb()

    logits = model(data)

    mem_after_forward = print_mem("      After forward", step + 1)
    print(f"      → Forward pass memory: +{mem_after_forward - mem_before_forward:.1f} MB")

    # Loss computation
    print("  [2] Computing loss...")
    mem_before_loss = get_memory_allocated_mb()

    loss = criterion(logits[train_nodes], data.y[train_nodes])

    mem_after_loss = print_mem("      After loss", step + 1)
    print(f"      → Loss computation: +{mem_after_loss - mem_before_loss:.1f} MB")

    # Backward pass
    print("  [3] Backward pass...")
    mem_before_backward = get_memory_allocated_mb()

    optimizer.zero_grad()
    loss.backward()

    mem_after_backward = print_mem("      After backward", step + 1)
    print(f"      → Backward pass memory: +{mem_after_backward - mem_before_backward:.1f} MB")

    # Optimizer step
    print("  [4] Optimizer step...")
    mem_before_optim = get_memory_allocated_mb()

    optimizer.step()

    mem_after_optim = print_mem("      After optimizer", step + 1)
    print(f"      → Optimizer step: +{mem_after_optim - mem_before_optim:.1f} MB")

    mem_end_step = print_mem("End of step", step + 1)
    mem_history.append(mem_end_step)

    print(f"\n  Step {step + 1} total memory: {mem_end_step:.1f} MB (Δ = {mem_end_step - mem_start_step:+.1f} MB)")

    # Check for memory growth
    if step > 0:
        growth = mem_end_step - mem_history[step - 1]
        if abs(growth) > 100:
            print(f"  ⚠️  LARGE MEMORY CHANGE: {growth:+.1f} MB from previous step!")

print("\n" + "=" * 80)
print("MEMORY SUMMARY")
print("=" * 80)

print(f"\nMemory by step:")
for i, mem in enumerate(mem_history):
    delta = mem - mem_history[i-1] if i > 0 else 0
    print(f"  Step {i+1}: {mem:>8.1f} MB (Δ = {delta:+7.1f} MB)")

print(f"\nMemory growth over {num_steps} steps: {mem_history[-1] - mem_history[0]:.1f} MB")
print(f"Average per-step growth: {(mem_history[-1] - mem_history[0]) / num_steps:.1f} MB")

# Check for memory leak
if mem_history[-1] > mem_history[0] + 100:
    print(f"\n⚠️  MEMORY LEAK DETECTED: Growing {(mem_history[-1] - mem_history[0]) / num_steps:.1f} MB per step")
    print("Possible causes:")
    print("  - Gradients not being cleared")
    print("  - Tensors accumulating in lists/caches")
    print("  - PyTorch autograd graph not being released")
else:
    print("\n✓ No significant memory leak detected")

print("\n" + "=" * 80)
print("CLEANUP TEST")
print("=" * 80)

print("\nDeleting model, optimizer, data...")
del model, optimizer, data, logits, loss
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

final_mem = print_mem("After cleanup")
print(f"\nMemory released: {mem_history[-1] - final_mem:.1f} MB")
