import torch
from torch import nn
from models.base_text_classifier import BaseTextClassifier


def create_lstm_model():
    pass


class fastTextClassifier(BaseTextClassifier):
    def __init__(self, vocab_size, embedding_dim, output_dim, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.freeze_embeddings = freeze_embeddings

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

    def forward(self, x):
        embedded = self.embedding(x)
        pooled = torch.mean(embedded, dim=1)
        return self.fc(pooled)

    def __repr(self):
        base = super().__repr__()
        return base
