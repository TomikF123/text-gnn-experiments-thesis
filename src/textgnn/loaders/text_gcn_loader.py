import torch
import torch.nn as nn
from dataset import TextDataset
import pickle
from utils import slugify
from textgnn.utils import get_active_encoding

def create_text_gcn_filename(dataset_config: dict) -> str:
    name = "text_gcn"
    encoding = get_active_encoding(dataset_config)
    window_size = encoding.get("window_size", 1)


    # tokens_trained_on = dataset_config["encoding"].get("tokens_trained_on", None)
    # embed_dim = dataset_config["encoding"].get("embedding_dim", None)
    # encode_token_type = (
        # dataset_config["encoding"]["encode_token_type"] + str(tokens_trained_on) + "B"
        # if str(tokens_trained_on)
        # else ""
    # )
    # encode_token_type += f"_{embed_dim}d" if embed_dim else ""

    parts = [
        f"{name}_seed_{dataset_config['random_seed']}",
        f"text_encoded_{encode_token_type}",
    ]
    # PROPERTIES
    return slugify("_".join(parts))

def create_text_gcn_dataset(missing_parent:str, missing_identity_graph:str):
    if missing_parent:
        pass
    if missing_identity_graph:
        pass
    pass
class GraphTextDataset(TextDataset):
    def __init__(self,  *args,graph_path:str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = graph_path  # Graph structure if available
        #self.node_features = kwargs.get("node_features", None)  # Node features if available

    def get_graph(self):
        return self.graph

    # def get_node_features(self):
    #     return self.node_features

    def __repr__(self):
        base = super().__repr__()
        return base + f"\nGraph: {self.graph}, Node Features: {self.node_features}"