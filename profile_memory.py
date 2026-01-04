#!/usr/bin/env python
"""
Memory profiler to identify exact location of memory explosion.
"""

import sys
import os
import gc
from pathlib import Path
import tracemalloc

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def get_memory_mb():
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def print_memory(label):
    """Print current memory usage with label."""
    mem_mb = get_memory_mb()
    print(f"[{mem_mb:>8.1f} MB] {label}")
    return mem_mb

def profile_textgcn_loading():
    """Profile memory usage during TextGCN dataset loading."""
    print("=" * 80)
    print("MEMORY PROFILING: TextGCN Dataset Loading")
    print("=" * 80)
    print()

    initial_mem = print_memory("Initial (before imports)")

    # Import modules
    print("\n--- Importing modules ---")
    from textgnn.loaders.textgcn_loader import TextGCNDataset
    import torch
    print_memory("After imports")

    # Find artifact path
    artifact_path = "saved/20ng-train-70-val-20-test-10-stop-words-remove-true-rare-words-remove-50-vocab-size-none-v2/text-gcn-x-identity-window-20"

    if not os.path.exists(artifact_path):
        print(f"\nERROR: Artifact path not found: {artifact_path}")
        print("Run: python main.py --config testGCN.json")
        return

    print(f"\n--- Loading dataset from: {artifact_path} ---")

    # Clear cache first
    TextGCNDataset.clear_cache()
    gc.collect()
    print_memory("After clearing cache")

    # Load train split
    print("\n1. Loading TRAIN dataset:")
    mem_before_train = print_memory("  Before load")
    train_ds = TextGCNDataset(artifact_path=artifact_path, split='train', x_type='identity')
    mem_after_train = print_memory("  After load")
    print(f"  → Memory increase: {mem_after_train - mem_before_train:.1f} MB")

    # Load val split (should reuse cache)
    print("\n2. Loading VAL dataset:")
    mem_before_val = print_memory("  Before load")
    val_ds = TextGCNDataset(artifact_path=artifact_path, split='val', x_type='identity')
    mem_after_val = print_memory("  After load")
    print(f"  → Memory increase: {mem_after_val - mem_before_val:.1f} MB")

    # Load test split (should reuse cache)
    print("\n3. Loading TEST dataset:")
    mem_before_test = print_memory("  Before load")
    test_ds = TextGCNDataset(artifact_path=artifact_path, split='test', x_type='identity')
    mem_after_test = print_memory("  After load")
    print(f"  → Memory increase: {mem_after_test - mem_before_test:.1f} MB")

    print("\n" + "=" * 80)
    print(f"Total memory increase for dataset loading: {mem_after_test - initial_mem:.1f} MB")
    print("=" * 80)

    return train_ds, val_ds, test_ds


def profile_model_creation():
    """Profile memory during model creation."""
    print("\n" + "=" * 80)
    print("MEMORY PROFILING: Model Creation")
    print("=" * 80)
    print()

    from textgnn.config_class import ModelConfig, DatasetConfig
    from textgnn.models.textgcn.model import create_textgcn_model
    import json

    # Load config
    with open("runConfigs/testGCN.json") as f:
        config_dict = json.load(f)

    dataset_config = DatasetConfig(**config_dict["dataset"])
    model_config = ModelConfig(**config_dict["model_config"])

    print("--- Creating model ---")
    mem_before_model = print_memory("Before model creation")

    model = create_textgcn_model(
        model_config=model_config,
        dataset_config=dataset_config
    )

    mem_after_model = print_memory("After model creation")
    print(f"→ Memory increase: {mem_after_model - mem_before_model:.1f} MB")

    # Print model info
    print(f"\nModel info:")
    print(f"  - num_nodes: {model.num_nodes}")
    print(f"  - num_classes: {model.num_classes}")
    print(f"  - x_type: {model.x_type}")
    print(f"  - Cached identity: {model._cached_identity is not None}")

    return model


def profile_forward_pass(model, dataset):
    """Profile memory during forward pass."""
    print("\n" + "=" * 80)
    print("MEMORY PROFILING: Forward Pass")
    print("=" * 80)
    print()

    import torch

    # Get data
    print("--- Getting data from dataset ---")
    mem_before_getitem = print_memory("Before dataset[0]")
    data = dataset[0]
    mem_after_getitem = print_memory("After dataset[0]")
    print(f"→ Memory increase: {mem_after_getitem - mem_before_getitem:.1f} MB")

    # Check data properties
    print(f"\nData properties:")
    print(f"  - x: {data.x}")
    print(f"  - edge_index shape: {data.edge_index.shape}")
    print(f"  - edge_attr shape: {data.edge_attr.shape}")
    print(f"  - y shape: {data.y.shape}")

    # Move to device
    device = torch.device("cpu")  # Use CPU for profiling
    print(f"\n--- Moving data to {device} ---")
    mem_before_device = print_memory("Before to(device)")
    data = data.to(device)
    model = model.to(device)
    mem_after_device = print_memory("After to(device)")
    print(f"→ Memory increase: {mem_after_device - mem_before_device:.1f} MB")

    # Forward pass
    print("\n--- Forward pass ---")
    model.eval()

    # Enable detailed tracking
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()

    mem_before_forward = print_memory("Before forward()")

    with torch.no_grad():
        print("\n  Calling model.forward(data)...")

        # Step through forward pass manually
        print("  [1] Extracting data attributes...")
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        print(f"      x is None: {x is None}")

        print(f"\n  [2] Checking if identity matrix needed...")
        print(f"      x_type: {model.x_type}")
        print(f"      Cached identity exists: {model._cached_identity is not None}")

        mem_before_identity = get_memory_mb()

        if x is None and model.x_type == "identity":
            print(f"\n  [3] Creating/retrieving identity matrix...")
            if model._cached_identity is None:
                print(f"      → Creating NEW sparse identity matrix")
                print(f"      → num_nodes: {model.num_nodes}")
                mem_before_create = get_memory_mb()

                # Manual creation to track each step
                indices = torch.arange(model.num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
                mem_after_indices = get_memory_mb()
                print(f"      → After creating indices: +{mem_after_indices - mem_before_create:.1f} MB")

                values = torch.ones(model.num_nodes, device=edge_index.device, dtype=torch.float32)
                mem_after_values = get_memory_mb()
                print(f"      → After creating values: +{mem_after_values - mem_after_indices:.1f} MB")

                model._cached_identity = torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=(model.num_nodes, model.num_nodes),
                    dtype=torch.float32,
                    device=edge_index.device
                )
                mem_after_sparse = get_memory_mb()
                print(f"      → After sparse_coo_tensor: +{mem_after_sparse - mem_after_values:.1f} MB")

                print(f"      → Sparse identity created: {model._cached_identity.is_sparse}")
                print(f"      → Shape: {model._cached_identity.shape}")
                print(f"      → nnz: {model._cached_identity._nnz()}")
            else:
                print(f"      → Using CACHED sparse identity")

            x = model._cached_identity

        mem_after_identity = get_memory_mb()
        print(f"\n      Identity matrix memory: +{mem_after_identity - mem_before_identity:.1f} MB")

        print(f"\n  [4] Passing through GCN layers...")
        for i, layer in enumerate(model.layers):
            mem_before_layer = get_memory_mb()
            print(f"\n      Layer {i}:")
            print(f"        Before: {mem_before_layer:.1f} MB")
            print(f"        x.is_sparse: {x.is_sparse if hasattr(x, 'is_sparse') else False}")
            print(f"        x.shape: {x.shape}")

            x = layer(x, edge_index, edge_attr)

            mem_after_layer = get_memory_mb()
            print(f"        After: {mem_after_layer:.1f} MB (+{mem_after_layer - mem_before_layer:.1f} MB)")
            print(f"        Output x.shape: {x.shape}")
            print(f"        Output x.is_sparse: {x.is_sparse if hasattr(x, 'is_sparse') else False}")

        logits = x

    mem_after_forward = print_memory("\nAfter forward()")
    print(f"→ Total forward pass memory increase: {mem_after_forward - mem_before_forward:.1f} MB")

    # Show top memory allocations
    snapshot2 = tracemalloc.take_snapshot()
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')

    print("\n--- Top 10 memory allocations during forward pass ---")
    for stat in top_stats[:10]:
        print(f"{stat}")

    tracemalloc.stop()

    return logits


def main():
    """Run complete memory profiling."""
    print("Starting memory profiling...")
    print("This will identify EXACTLY where memory explodes.\n")

    # Check if psutil is installed
    try:
        import psutil
    except ImportError:
        print("ERROR: psutil not installed")
        print("Install with: pip install psutil")
        return

    try:
        # Profile dataset loading
        train_ds, val_ds, test_ds = profile_textgcn_loading()

        # Profile model creation
        model = profile_model_creation()

        # Profile forward pass
        profile_forward_pass(model, train_ds)

        print("\n" + "=" * 80)
        print("PROFILING COMPLETE")
        print("=" * 80)
        final_mem = get_memory_mb()
        print(f"Final memory usage: {final_mem:.1f} MB")

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"ERROR OCCURRED:")
        print(f"{'='*80}")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print(f"\nMemory at error: {get_memory_mb():.1f} MB")


if __name__ == "__main__":
    main()
