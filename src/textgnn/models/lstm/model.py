import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from textgnn.loaders.lstm_loader import LSTMDataset
from textgnn.models.base_text_classifier import BaseTextClassifier
from textgnn.models.lstm.train import train as train_lstm
from textgnn.models.mlp import MLP
from textgnn.config_class import ModelConfig, DatasetConfig


def create_lstm_model(
    model_config: ModelConfig, dataset_config: DatasetConfig, dataset: LSTMDataset = None
):
    """
    Create LSTM model from configuration.

    Args:
        model_config: Pydantic ModelConfig model
        dataset_config: Pydantic DatasetConfig model
        dataset: Optional LSTMDataset instance

    Returns:
        LSTMClassifier instance
    """
    common_params = model_config.common_params
    model_specific_params = model_config.model_specific_params

    # Get vocab_size from dataset's vocab (not config, which may be None)
    vocab_size = len(dataset.vocab) if dataset and dataset.vocab else None
    embedding_dim = model_specific_params.get("embedding_dim", 50)
    hidden_dim = model_specific_params.get("hidden_dim", 128)
    num_classes = len(dataset.df["label"].unique()) if dataset is not None else 2
    output_dim = model_specific_params.get("output_size", num_classes)
    num_layers = model_specific_params.get("num_layers", 2)
    bidirectional = model_specific_params.get("bidirectional", True)
    dropout = model_specific_params.get("dropout", 0.5)
    embedding_matrix = dataset.embedding_matrix if dataset else None
    freeze_embeddings = model_specific_params.get("freeze_embeddings", True)

    pooling = model_specific_params.get("pooling", "last_hidden")

    return LSTMClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        embedding_matrix=embedding_matrix,
        freeze_embeddings=freeze_embeddings,
        pooling=pooling,
    )


class LSTMClassifier(BaseTextClassifier):
    def __init__(
        self,
        vocab_size: int = None,
        embedding_dim: int = None,
        hidden_dim: int = None,
        output_dim: int = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.5,
        embedding_matrix: torch.Tensor = None,
        freeze_embeddings: bool = False,
        pooling: str = 'last_hidden',
        use_attention: bool = False,
        use_mlp_as_head: bool = False,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            output_size=output_dim,
            #  padding_idx=0,
            freeze_embeddings=freeze_embeddings,
        )
        self.freeze_embeddings = freeze_embeddings
        self.use_attention = use_attention
        self.pooling = pooling

        # Always use nn.Embedding (dataset returns indices, model embeds on GPU)
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if embedding_matrix is not None:
            self.embedding.weight = nn.Parameter(embedding_matrix)
            self.embedding.weight.requires_grad = not self.freeze_embeddings

        self.direction_factor = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )

        if self.use_attention:
            self.attention = nn.Linear(hidden_dim * self.direction_factor, 1)

        if use_mlp_as_head:
            self.head = MLP(
                in_dim=hidden_dim * self.direction_factor,
                hidden_dims=[64],  # list of hidden sizes, or empty list for linear
                out_dim=output_dim,
                act="relu",
                use_bn=True,
                dropout=0.5,
            )
        else:
            self.head = nn.Linear(hidden_dim * self.direction_factor, output_dim)

        self.dropout = nn.Dropout(dropout)
        self.train_func = train_lstm

    def forward(self, x, lengths=None):
        x = self.embedding(x)

        if lengths is not None:
            lengths = lengths.clamp(min=1).cpu()
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            lstm_out_packed, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, (h_n, c_n) = self.lstm(x)

        if self.pooling == 'max':
            # Mask padding positions to -inf before max pooling
            if lengths is not None:
                mask = torch.arange(lstm_out.size(1), device=lstm_out.device)[None, :] >= lengths[:, None].to(lstm_out.device)
                lstm_out = lstm_out.masked_fill(mask.unsqueeze(-1), float('-inf'))
            pooled = lstm_out.max(dim=1).values  # [B, hidden*2]
            out = self.dropout(pooled)
        elif self.use_attention:
            attn_scores = self.attention(lstm_out)  # [B, T, 1]
            attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]
            context = torch.sum(attn_weights * lstm_out, dim=1)  # [B, H]
            out = self.dropout(context)
        else:
            if self.lstm.bidirectional:
                last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
            else:
                last_hidden = h_n[-1]
            out = self.dropout(last_hidden)

        return self.head(out)

    def __repr__(self):
        base = super().__repr__()
        return f"{base}\nfreezing embeddings: {self.freeze_embeddings}\ntrainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
