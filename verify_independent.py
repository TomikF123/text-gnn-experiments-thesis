"""Independent verification: configs vs filesystem, no code functions used."""
import json, glob, os

# 1. From configs: what unique artifact combos SHOULD exist?
expected_base = set()  # (dataset, preprocessing) combos
expected_model = set()  # (dataset, preprocessing, model_type, model_settings) combos

for f in sorted(glob.glob("runConfigs/experiments/*/*.json")):
    c = json.load(open(f))
    ds = c["dataset"]
    pp = ds.get("preprocess", {})
    model_type = c["model_config"]["model_type"]

    base_key = (
        ds["name"],
        tuple(ds["tvt_split"]),
        pp.get("remove_stopwords"),
        pp.get("remove_rare_words"),
    )
    expected_base.add(base_key)

    if model_type == "lstm":
        enc = ds.get("rnn_encoding", {})
        model_key = base_key + ("lstm", enc.get("tokens_trained_on"))
    elif model_type == "text_gcn":
        enc = ds.get("gnn_encoding", {})
        model_key = base_key + ("text_gcn", enc.get("window_size"), enc.get("x_type", "identity"))
    elif model_type == "texting":
        enc = ds.get("gnn_encoding", {})
        model_key = base_key + ("texting", enc.get("window_size"), enc.get("embedding_dim"), ds.get("max_len"))

    expected_model.add(model_key)

# 2. From filesystem: what actually exists?
actual_base = set()
actual_model = set()

for d in os.listdir("saved"):
    base_path = os.path.join("saved", d)
    if not os.path.isdir(base_path):
        continue
    actual_base.add(d)
    for sub in os.listdir(base_path):
        sub_path = os.path.join(base_path, sub)
        if os.path.isdir(sub_path):
            actual_model.add((d, sub))

# 3. Report
print(f"Expected base dirs: {len(expected_base)}")
print(f"Actual base dirs:   {len(actual_base)}")
print(f"Expected model dirs: {len(expected_model)}")
print(f"Actual model dirs:   {len(actual_model)}")

print(f"\n=== Actual base dirs ===")
for d in sorted(actual_base):
    print(f"  {d}")

print(f"\n=== Actual model subdirs ===")
for base, sub in sorted(actual_model):
    print(f"  {base}/{sub}")

# 4. Check: multiple model types sharing a subdir?
from collections import Counter
subdir_names = Counter(sub for _, sub in actual_model)
shared = {name: count for name, count in subdir_names.items() if count > 1}
if shared:
    print(f"\n=== SHARED subdir names (same name in multiple base dirs) ===")
    for name, count in sorted(shared.items()):
        dirs = [base for base, sub in actual_model if sub == name]
        print(f"  {name} appears in {count} base dirs: {dirs}")
