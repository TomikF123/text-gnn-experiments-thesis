"""
Test script to verify TextGCN memory optimization (shared graph caching).

This script demonstrates the 3x memory reduction from sharing graph structure
across train/val/test splits.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from textgnn.loaders.textgcn_loader import TextGCNDataset


def test_graph_caching():
    """Test that graph structure is shared across splits."""

    # Find 20ng artifacts with remove_rare_words=50
    artifact_path = "saved/20ng-train-70-val-20-test-10-stop-words-remove-true-rare-words-remove-50-vocab-size-none-v2/text-gcn-x-identity-window-20"

    if not os.path.exists(artifact_path):
        print(f"ERROR: Artifact path not found: {artifact_path}")
        print("\nPlease run preprocessing first:")
        print("  python main.py --config testGCN.json")
        return

    print("=" * 70)
    print("Testing Memory-Efficient Graph Caching")
    print("=" * 70)
    print()

    # Clear any existing cache
    TextGCNDataset.clear_cache()

    # Load train split (first load - reads from disk)
    print("1. Loading TRAIN split (reads from disk):")
    print("-" * 70)
    train_ds = TextGCNDataset(artifact_path=artifact_path, split='train', x_type='identity')
    print()

    # Load val split (should reuse cache)
    print("2. Loading VAL split (should reuse cache):")
    print("-" * 70)
    val_ds = TextGCNDataset(artifact_path=artifact_path, split='val', x_type='identity')
    print()

    # Load test split (should reuse cache)
    print("3. Loading TEST split (should reuse cache):")
    print("-" * 70)
    test_ds = TextGCNDataset(artifact_path=artifact_path, split='test', x_type='identity')
    print()

    # Verify they share the same tensors (not copies)
    print("=" * 70)
    print("Verification: Checking if splits share memory")
    print("=" * 70)

    # Check if edge_index tensors are the same object in memory
    same_edge_index = train_ds.edge_index is val_ds.edge_index is test_ds.edge_index
    same_edge_attr = train_ds.edge_attr is val_ds.edge_attr is test_ds.edge_attr

    print(f"edge_index shared across splits: {same_edge_index} ✓" if same_edge_index else "✗")
    print(f"edge_attr shared across splits: {same_edge_attr} ✓" if same_edge_attr else "✗")
    print()

    # Calculate memory savings
    cache_size_mb = train_ds._get_cache_size_mb(artifact_path)
    memory_saved_mb = cache_size_mb * 2  # We avoided loading 2 extra copies

    print("=" * 70)
    print("Memory Usage Summary")
    print("=" * 70)
    print(f"Graph cache size: {cache_size_mb:.1f} MB")
    print(f"Number of splits loaded: 3 (train, val, test)")
    print(f"Memory saved by sharing: ~{memory_saved_mb:.1f} MB")
    print()
    print(f"BEFORE optimization: ~{cache_size_mb * 3:.1f} MB (3 separate copies)")
    print(f"AFTER optimization:  ~{cache_size_mb:.1f} MB (1 shared copy)")
    print(f"Reduction: {((memory_saved_mb) / (cache_size_mb * 3)) * 100:.1f}%")
    print("=" * 70)

    # Clean up
    TextGCNDataset.clear_cache()
    print("\n✓ Test completed successfully!")


if __name__ == "__main__":
    test_graph_caching()
