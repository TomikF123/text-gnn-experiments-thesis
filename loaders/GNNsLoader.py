import torch
import torch.nn as nn
from dataset import TextDataset
import pickle
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