"""Verify each config gets the artifacts it needs."""
import json, glob, os

saved_root = "saved"
issues = []
ok = 0

from textgnn.config_class import DatasetConfig
from textgnn.load_data import create_dir_name_based_on_dataset_config, create_file_name

for f in sorted(glob.glob("runConfigs/experiments/*/*.json")):
    c = json.load(open(f))
    name = c["run_name"]
    ds = c["dataset"]
    model_type = c["model_config"]["model_type"]

    ds_config = DatasetConfig(**ds)
    base_dir = os.path.join(saved_root, create_dir_name_based_on_dataset_config(ds_config))
    model_dir = os.path.join(base_dir, create_file_name(ds_config, model_type))

    if not os.path.exists(base_dir):
        issues.append(f"{name}: BASE MISSING {base_dir}")
        continue
    if not os.path.exists(model_dir):
        issues.append(f"{name}: MODEL DIR MISSING {model_dir}")
        continue

    if model_type == "lstm":
        enc = ds.get("rnn_encoding", {})
        needs_glove = enc.get("tokens_trained_on") is not None
        has_glove = os.path.exists(os.path.join(model_dir, "embedding_matrix.pt"))
        if needs_glove and not has_glove:
            issues.append(f"{name}: NEEDS GLOVE BUT MISSING {model_dir}")
        elif not needs_glove and has_glove:
            issues.append(f"{name}: RANDINIT BUT HAS GLOVE {model_dir}")
        else:
            ok += 1

    elif model_type == "text_gcn":
        missing = [n for n in ["ALL_edge_index.pt", "ALL_edge_attr.pt", "ALL_y.pt", "train_mask.pt"] if not os.path.exists(os.path.join(model_dir, n))]
        if missing:
            issues.append(f"{name}: MISSING {missing} in {model_dir}")
        else:
            ok += 1

    elif model_type == "texting":
        missing = [n for n in ["train_data.pkl", "val_data.pkl", "test_data.pkl"] if not os.path.exists(os.path.join(model_dir, n))]
        if missing:
            issues.append(f"{name}: MISSING {missing} in {model_dir}")
        else:
            ok += 1

print(f"OK: {ok}, Issues: {len(issues)}")
for i in issues:
    print(f"  {i}")
