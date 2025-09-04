from datasets import load_dataset
from datasets import concatenate_datasets
import os


data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def save_dataset_to_csv(dataset, name,suffix, directory=data_dir):

    if not f"{directory}/{name}{suffix}.csv" in os.listdir(directory):
        os.makedirs(directory, exist_ok=True)
        dataset.to_csv(f"{directory}/{name}{suffix}.csv", index=False)



# Download & save AG News
mr = load_dataset("rotten_tomatoes")
#save_dataset_to_csv(dataset = mr["train"], name="mr", suffix="train", directory=data_dir)
#save_dataset_to_csv(dataset = mr["test"], name="mr", suffix="test", directory=data_dir)
#save_dataset_to_csv(dataset = mr["validation"], name="mr", suffix="val", directory=data_dir)

mr_test_val = concatenate_datasets([mr["test"], mr["validation"]])
#save_dataset_to_csv(dataset = mr_test_val, name="mr", suffix="test_val", directory=data_dir)
mr_all = concatenate_datasets([mr["train"], mr_test_val])
save_dataset_to_csv(dataset = mr_all, name="mr", suffix="", directory=data_dir)



ng_20 = load_dataset("SetFit/20_newsgroups")
#save_dataset_to_csv(dataset = ng_20["train"], name="ng_20", suffix="train", directory=data_dir)
#save_dataset_to_csv(dataset = ng_20["test"], name="ng_20", suffix="test", directory=data_dir)
ng_20_all = concatenate_datasets([ng_20["train"], ng_20["test"]])
save_dataset_to_csv(dataset = ng_20_all, name="20ng", suffix="", directory=data_dir)

#Donload nltk stopwords
import nltk
nltk.download("stopwords", download_dir=data_dir)

#Download Glove embeddings
glove_dir = os.path.join(data_dir, "glove")
os.makedirs(glove_dir, exist_ok=True)
import requests
glove_url = "https://nlp.stanford.edu/data/glove.6B.zip"
glove_zip_path = os.path.join(glove_dir, "glove.6B.zip")
if not os.path.exists(glove_zip_path):
    "downloading glove embeddings..."
    response = requests.get(glove_url)
    with open(glove_zip_path, "wb") as f:
        f.write(response.content)

print("Glove embedings are ready.")
