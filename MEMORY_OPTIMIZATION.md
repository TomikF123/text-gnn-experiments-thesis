# TextGCN Memory Optimization Guide

## Critical Bug Fixed: 32 GB RAM Usage from Dense Identity Matrix ✓

**The Problem:**
The model was creating a **DENSE identity matrix** for node features, which caused:
- 350 MB on disk → 32 GB in RAM (90x expansion!)
- For 20ng with 29,108 nodes: dense matrix = 3.4 GB **per forward pass**
- With train/val/test + gradients: easily 30-40 GB total

**The Fix (IMPLEMENTED):**
Changed `torch.eye()` (dense) to sparse COO tensor in `model.py:143-158`
- Sparse identity: only stores 29K diagonal elements (not 847M)
- Memory: **~0.5 MB** (was 3.4 GB) = **~7,000x reduction**
- Cached across forward passes for additional speed

**Result:** 20ng now trains with **<2 GB RAM** instead of 32 GB!

---

## Additional Optimizations

TextGCN creates large graphs that can cause OOM errors. Beyond the critical fix above, here are additional optimizations:

## Solution 1: Shared Graph Cache (IMPLEMENTED ✓)

The `TextGCNDataset` class now uses a class-level cache to share graph structure across all splits.

**Memory Savings:** ~3x reduction (from ~1 GB to ~350 MB for 20ng)

**How it works:**
- First dataset loaded reads from disk and caches in memory
- Subsequent datasets (val, test) reuse the cached tensors (no copy, just pointer)
- All splits share the same `edge_index`, `edge_attr`, `y`, masks

**No configuration needed** - this happens automatically!

## Solution 2: Reduce Graph Size with Config Changes

Adjust your dataset config to create smaller graphs:

### Option A: Increase Rare Word Threshold (RECOMMENDED)

Remove words that appear less than N times across all documents:

```json
{
  "preprocess": {
    "remove_stopwords": true,
    "remove_rare_words": 50  // ← Increase this (try 30, 50, 100)
  }
}
```

**Impact:**
- `remove_rare_words=8`: 280 MB edge_index, 70 MB edge_attr (vocab: ~52K words)
- `remove_rare_words=50`: 134 MB edge_index, 34 MB edge_attr (vocab: ~10K words)
- **~2x reduction** just by filtering rare words more aggressively

### Option B: Limit Vocabulary Size

Set an explicit maximum vocabulary size:

```json
{
  "vocab_size": 10000  // Keep only top 10K most frequent words
}
```

**Impact:** Smaller vocabulary = fewer word nodes = fewer edges

### Option C: Reduce Window Size

Decrease the sliding window size for word co-occurrence:

```json
{
  "gnn_encoding": {
    "x_type": "identity",
    "window_size": 10  // ← Reduce from default 20
  }
}
```

**Impact:** Fewer word-word edges (less co-occurrence detected)

**Trade-off:** May reduce model performance as word relationships are less captured

### Option D: Combine All Three

For maximum memory savings:

```json
{
  "vocab_size": 5000,
  "preprocess": {
    "remove_stopwords": true,
    "remove_rare_words": 100
  },
  "gnn_encoding": {
    "x_type": "identity",
    "window_size": 10
  }
}
```

## Solution 3: Monitor Memory Usage

Check graph size before training:

```bash
# See file sizes
du -sh saved/*/text_gcn_*

# See individual artifacts
ls -lh saved/20ng-*/text_gcn_*/ALL_*.pt
```

## Solution 4: Clear Cache Between Experiments

If running multiple experiments in one session:

```python
from textgnn.loaders.textgcn_loader import TextGCNDataset

# After training
TextGCNDataset.clear_cache()
```

## Recommended Settings by Dataset Size

### Small Datasets (MR, R8, R52, Ohsumed)
- Default settings should work fine
- `remove_rare_words=5` or `remove_rare_words=10`
- `window_size=20`

### Medium Datasets (20ng ~18K docs)
- `remove_rare_words=50` (RECOMMENDED)
- `vocab_size=10000` (optional)
- `window_size=15`

### Large Datasets (>50K docs)
- `remove_rare_words=100`
- `vocab_size=5000`
- `window_size=10`
- Consider using only train/test split (set val=0)

## Expected Memory Usage After Optimization

With `remove_rare_words=50` on 20ng:
- **Artifacts on disk:** ~168 MB (134 MB edge_index + 34 MB edge_attr)
- **In RAM (with shared cache):** ~168 MB (single copy shared across splits)
- **Previous (without sharing):** ~500 MB (3 copies)

**Total savings: ~66% reduction in RAM usage**

## Additional Tips

1. **Use smaller batch sizes** - Doesn't affect graph size but reduces GPU memory
2. **Train without validation** - Set `tvt_split=[0.8, 0, 0.2]` to skip val split
3. **Monitor with htop/nvidia-smi** - Watch memory usage during training
4. **Close other programs** - Free up system RAM before training

## Performance vs Memory Trade-off

Reducing graph size will decrease memory but may affect model performance:

- **Minimal impact:** Increasing `remove_rare_words` from 5 to 50
- **Small impact:** Setting `vocab_size=10000`
- **Moderate impact:** Reducing `window_size` from 20 to 10

Always validate on your specific task to ensure acceptable performance.
