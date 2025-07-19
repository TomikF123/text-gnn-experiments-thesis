from dataset import TextDataset
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class LSTMDataset(TextDataset):
    def __init__(self, data:np.array, labels:np.array,pad_token:int=0, max_len:int=None, embeddings_path:str = None):
        super().__init__(data, labels)

    
    def pad_seq(self, seq,max_len):
        pass

    def apply_embeddings(self, embeddings_path,vocab):
        pass
    def build_vocab(self, data):
       pass
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

