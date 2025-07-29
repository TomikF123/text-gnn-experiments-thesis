import torch
import torch.nn as nn
from loaders.lstmLoader import LSTMDataset


def create_lstm_model(
    model_config: dict, dataset_config: dict, dataset: LSTMDataset = None
):
    common_params = model_config.get("common_params", {})
    model_specific_params = model_config.get("model_specific_params", {})
    vocab_size = dataset_config.get("vocab_size", None)
    embedding_dim = model_config.get("embedding_dim", 50)
    hidden_dim = model_specific_params.get("hidden_size", 128)
    output_dim = model_specific_params.get("output_size", 20)
    num_layers = model_specific_params.get("num_layers", 2)
    bidirectional = model_specific_params.get("bidirectional", True)
    dropout = model_specific_params.get("dropout", 0.5)
    embedding_matrix = dataset.embedding_matrix if dataset else None
    freeze_embeddings = model_specific_params.get("freeze_embeddings", True)
    encoding_type = dataset_config["encoding"].get("encode_token_type", "index")
    print("freeze=" + f"{freeze_embeddings}")
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


class LSTMClassifier(nn.Module):
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
        use_attention: bool = True,  # ✅ NEW FLAG
    ):
        super().__init__()
        self.freeze_embeddings = freeze_embeddings
        self.encoding_type = encoding_type
        self.use_attention = use_attention

        if encoding_type == "index":
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
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

        self.fc = nn.Linear(hidden_dim * self.direction_factor, output_dim)
        self.dropout = nn.Dropout(dropout)

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

        return self.fc(out)

    def __repr__(self):
        base = super().__repr__()
        return f"{base}\nfreezing embeddings: {self.freeze_embeddings}\ntrainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}"
