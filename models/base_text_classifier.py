import torch
from torch import nn


class BaseTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size=None,
        embedding_dim=None,
        output_size=None,
        freeze_embeddings=False,
    ):
        super().__init__()
        self.embedding = (
            nn.Embedding(vocab_size, embedding_dim)
            if (vocab_size and embedding_dim)
            else None
        )
        if self.embedding and freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.output_size = output_size
        self.embedding_dim = embedding_dim

    def forward(self):
        raise NotImplementedError("This method should be overridden by subclasses")

    def __repr__(self):
        base = super().__repr__()
        return (
            base
            + f"\nOutput size: {self.output_size}, Embedding dim: {self.embedding_dim}"
        )


class GraphTextClassifier(BaseTextClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for graph-based classifiers can go here
