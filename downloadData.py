from datasets import load_dataset
from datasets import concatenate_datasets
import os

os.makedirs("data", exist_ok=True)

def save_dataset_to_csv(dataset, name,suffix, directory="data"):

    if not f"{directory}/{name}_{suffix}.csv" in os.listdir(directory):
        os.makedirs(directory, exist_ok=True)
        dataset.to_csv(f"{directory}/{name}_{suffix}.csv", index=False)



# Download & save AG News
mr = load_dataset("rotten_tomatoes")
save_dataset_to_csv(dataset = mr["train"], name="mr", suffix="train", directory="data")
save_dataset_to_csv(dataset = mr["test"], name="mr", suffix="test", directory="data")
save_dataset_to_csv(dataset = mr["validation"], name="mr", suffix="val", directory="data")

mr_test_val = concatenate_datasets([mr["test"], mr["validation"]])
save_dataset_to_csv(dataset = mr_test_val, name="mr", suffix="test_val", directory="data")
mr_all = concatenate_datasets([mr["train"], mr_test_val])
save_dataset_to_csv(dataset = mr_all, name="mr", suffix="all", directory="data")



ng_20 = load_dataset("SetFit/20_newsgroups")
save_dataset_to_csv(dataset = ng_20["train"], name="ng_20", suffix="train", directory="data")
save_dataset_to_csv(dataset = ng_20["test"], name="ng_20", suffix="test", directory="data")
ng_20_all = concatenate_datasets([ng_20["train"], ng_20["test"]])
save_dataset_to_csv(dataset = ng_20_all, name="ng_20", suffix="all", directory="data")


