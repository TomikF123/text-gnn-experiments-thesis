import torch
import torch.nn as nn
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

    vocab_size = dataset_config.vocab_size
    embedding_dim = model_specific_params.get("embedding_dim", 50)
    hidden_dim = model_specific_params.get("hidden_size", 128)
    output_dim = model_specific_params.get("output_size", 20)
    num_layers = model_specific_params.get("num_layers", 2)
    bidirectional = model_specific_params.get("bidirectional", True)
    dropout = model_specific_params.get("dropout", 0.5)
    embedding_matrix = dataset.embedding_matrix if dataset else None
    freeze_embeddings = model_specific_params.get("freeze_embeddings", True)
    encoding_type = dataset_config.rnn_encoding.encode_token_type if dataset_config.rnn_encoding.encode_token_type is not None else "glove"

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
        encoding_type=encoding_type,
    )


class LSTMClassifier(BaseTextClassifier):
    def __init__(
        self,
        vocab_size: int = None,
        embedding_dim: int = None,
        hidden_dim: int = None,
        output_dim: int = None,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.5,
        embedding_matrix: torch.Tensor = None,
        freeze_embeddings: bool = False,
        encoding_type: str = "index",
        use_attention: bool = True,
        use_mlp_as_head: bool = True,
    ):
        super().__init__(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            output_size=output_dim,
            #  padding_idx=0,
            freeze_embeddings=freeze_embeddings,
        )
        self.freeze_embeddings = freeze_embeddings
        self.encoding_type = encoding_type
        self.use_attention = use_attention

        if encoding_type == "index":
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            if embedding_matrix is not None:
                self.embedding.weight = nn.Parameter(embedding_matrix)
                self.embedding.weight.requires_grad = not self.freeze_embeddings
        elif encoding_type == "glove":
            self.embedding = None  # Already embedded in dataset
        else:
            raise ValueError(f"Unsupported encoding_type: {encoding_type}")

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

    def forward(self, x):
        if self.encoding_type == "index":
            x = self.embedding(x)

        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.use_attention:
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
