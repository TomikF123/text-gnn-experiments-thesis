"""
TextING Model Implementation

Inductive Text GNN with GRU-based message passing and attention readout.
Each document has its own graph with word nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from textgnn.models.base_text_classifier import GraphTextClassifier
from textgnn.config_class import ModelConfig, DatasetConfig


class TextINGClassifier(GraphTextClassifier):
    """
    TextING: Inductive Text Classification via Graph Neural Networks.

    Uses GRU-based graph convolution and attention-based readout.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 96,
        gru_steps: int = 2,
        dropout: float = 0.5,
        **kwargs
    ):
        """
        Initialize TextING model.

        Args:
            input_dim: Input feature dimension (GloVe embedding dim, e.g., 300)
            output_dim: Number of output classes
            hidden_dim: Hidden dimension for GNN layers
            gru_steps: Number of GRU message passing steps
            dropout: Dropout rate
        """
        # Don't create embedding layer (TextING uses pre-trained GloVe)
        super().__init__(vocab_size=None, embedding_dim=None, output_size=output_dim)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gru_steps = gru_steps
        self.dropout_rate = dropout

        # Graph layer with GRU-based message passing
        self.graph_layer = GraphLayer(
            input_dim=input_dim,
            output_dim=hidden_dim,
            act=nn.Tanh(),
            dropout=dropout,
            gru_steps=gru_steps
        )

        # Readout layer with attention
        self.readout_layer = ReadoutLayer(
            input_dim=hidden_dim,
            output_dim=output_dim,
            act=nn.Tanh(),
            dropout=dropout
        )

        # Set training function
        from textgnn.models.texting.train import train_texting
        self.train_func = train_texting

    def forward(self, adj, features, mask):
        """
        Forward pass through TextING.

        Args:
            adj: List of sparse adjacency matrices (one per document in batch)
            features: Batch of node features [batch_size, max_nodes, input_dim]
            mask: Batch of node masks [batch_size, max_nodes, 1]

        Returns:
            logits: Output logits [batch_size, num_classes]
            embeddings: Node embeddings [batch_size, max_nodes, hidden_dim]
        """
        # Graph convolution with GRU
        embeddings = self.graph_layer(features, adj, mask)

        # Readout (aggregate to graph-level representation)
        # Note: readout doesn't use adj, so we pass None
        logits = self.readout_layer(embeddings, None, mask)

        return logits, embeddings


class GRUUnit(nn.Module):
    """
    GRU unit for graph node updates.

    Implements: h_t = GRU(A @ h_{t-1}, h_{t-1})
    """

    def __init__(self, output_dim, act, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.dropout = nn.Dropout(p=dropout)
        self.act = act

        # Update gate parameters
        self.z0_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.z1_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.z0_bias = nn.Parameter(torch.zeros(output_dim))
        self.z1_bias = nn.Parameter(torch.zeros(output_dim))

        # Reset gate parameters
        self.r0_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.r1_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.r0_bias = nn.Parameter(torch.zeros(output_dim))
        self.r1_bias = nn.Parameter(torch.zeros(output_dim))

        # Candidate parameters
        self.h0_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.h1_weight = nn.Parameter(torch.Tensor(output_dim, output_dim))
        self.h0_bias = nn.Parameter(torch.zeros(output_dim))
        self.h1_bias = nn.Parameter(torch.zeros(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters with Xavier/Glorot."""
        for weight in [self.z0_weight, self.z1_weight, self.r0_weight,
                      self.r1_weight, self.h0_weight, self.h1_weight]:
            nn.init.xavier_uniform_(weight)

    def forward(self, adj, x, mask):
        """
        GRU update step.

        Args:
            adj: List of sparse adjacency matrices (one per document in batch)
            x: Node features [batch_size, num_nodes, hidden_dim]
            mask: Node mask [batch_size, num_nodes, 1]

        Returns:
            Updated node features [batch_size, num_nodes, hidden_dim]
        """
        # Aggregate from neighbors: a = A @ x
        # Process each graph in the batch separately (adj is a list of sparse tensors)
        batch_size = x.size(0)
        a_list = []

        for i in range(batch_size):
            # Get sparse adjacency for this document
            adj_i = adj[i]  # Sparse tensor [num_nodes, num_nodes]
            x_i = x[i]      # Dense tensor [num_nodes, hidden_dim]

            # Sparse matrix multiplication: a_i = adj_i @ x_i
            a_i = torch.sparse.mm(adj_i, x_i)  # [num_nodes, hidden_dim]
            a_list.append(a_i)

        # Stack results back into batch
        a = torch.stack(a_list, dim=0)  # [batch_size, num_nodes, hidden_dim]

        # Apply dropout to aggregated features (not adjacency)
        if self.training:
            a = self.dropout(a)

        # Update gate: z = sigmoid(W_z @ a + U_z @ x)
        z0 = torch.matmul(a, self.z0_weight) + self.z0_bias
        z1 = torch.matmul(x, self.z1_weight) + self.z1_bias
        z = torch.sigmoid(z0 + z1)

        # Reset gate: r = sigmoid(W_r @ a + U_r @ x)
        r0 = torch.matmul(a, self.r0_weight) + self.r0_bias
        r1 = torch.matmul(x, self.r1_weight) + self.r1_bias
        r = torch.sigmoid(r0 + r1)

        # Candidate: h_tilde = tanh(W_h @ a + U_h @ (r * x))
        h0 = torch.matmul(a, self.h0_weight) + self.h0_bias
        h1 = torch.matmul(r * x, self.h1_weight) + self.h1_bias
        h_tilde = self.act(mask * (h0 + h1))

        # Final update: h = z * h_tilde + (1-z) * x
        return z * h_tilde + (1 - z) * x


class GraphLayer(nn.Module):
    """
    Graph layer with GRU-based message passing.
    """

    def __init__(self, input_dim, output_dim, act=nn.Tanh(), dropout=0.5, gru_steps=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout = nn.Dropout(p=dropout)
        self.gru_steps = gru_steps

        # Encoding layer: project input to hidden dimension
        self.encode_weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.encode_bias = nn.Parameter(torch.zeros(output_dim))

        # GRU unit for message passing
        self.gru_unit = GRUUnit(output_dim=output_dim, act=act, dropout=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.encode_weight)

    def forward(self, features, adj, mask):
        """
        Forward pass through graph layer.

        Args:
            features: Node features [batch_size, num_nodes, input_dim]
            adj: List of sparse adjacency matrices (one per document in batch)
            mask: Node mask [batch_size, num_nodes, 1]

        Returns:
            Node embeddings [batch_size, num_nodes, output_dim]
        """
        # Dropout on input features
        features = self.dropout(features)

        # Encode: project to hidden dimension
        output = torch.matmul(features, self.encode_weight) + self.encode_bias
        output = mask * self.act(output)

        # GRU message passing steps
        for _ in range(self.gru_steps):
            output = self.gru_unit(adj, output, mask)

        return output


class ReadoutLayer(nn.Module):
    """
    Graph readout layer with attention-based aggregation.

    Aggregates node embeddings to graph-level representation using:
    1. Soft attention over nodes
    2. Mean + max pooling
    3. MLP classifier
    """

    def __init__(self, input_dim, output_dim, act=nn.ReLU(), dropout=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act
        self.dropout = nn.Dropout(p=dropout)

        # Attention weights
        self.att_weight = nn.Parameter(torch.Tensor(input_dim, 1))
        self.att_bias = nn.Parameter(torch.zeros(1))

        # Embedding transformation
        self.emb_weight = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.emb_bias = nn.Parameter(torch.zeros(input_dim))

        # MLP classifier
        self.mlp_weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        self.mlp_bias = nn.Parameter(torch.zeros(output_dim))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.att_weight)
        nn.init.xavier_uniform_(self.emb_weight)
        nn.init.xavier_uniform_(self.mlp_weight)

    def forward(self, x, adj, mask):
        """
        Forward pass through readout layer.

        Args:
            x: Node embeddings [batch_size, num_nodes, input_dim]
            adj: Adjacency (not used here)
            mask: Node mask [batch_size, num_nodes, 1]

        Returns:
            Graph-level logits [batch_size, output_dim]
        """
        # Soft attention: att = sigmoid(x @ W_att + b_att)
        att = torch.sigmoid(torch.matmul(x, self.att_weight) + self.att_bias)

        # Transform embeddings: emb = act(x @ W_emb + b_emb)
        emb = self.act(torch.matmul(x, self.emb_weight) + self.emb_bias)

        # Compute number of real nodes per graph (for mean pooling)
        N = torch.sum(mask, dim=1)  # [batch_size, 1]

        # Create large negative mask for max pooling (to ignore padding)
        M = (mask - 1) * 1e9  # [batch_size, num_nodes, 1]

        # Graph-level aggregation: weighted sum + max pooling
        g = mask * att * emb  # Apply mask and attention
        g_mean = torch.sum(g, dim=1) / N  # Mean pooling [batch_size, input_dim]
        g_max = torch.max(g + M, dim=1)[0]  # Max pooling [batch_size, input_dim]
        g = g_mean + g_max  # Combine [batch_size, input_dim]

        # Dropout
        g = self.dropout(g)

        # Classify: output = g @ W_mlp + b_mlp
        output = torch.matmul(g, self.mlp_weight) + self.mlp_bias

        return output


def create_texting_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    dataset=None
):
    """
    Factory function to create TextING model.

    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        dataset: Dataset instance (optional, for getting num_classes)

    Returns:
        TextINGClassifier instance
    """
    import os
    import pickle
    from textgnn.load_data import create_dir_name_based_on_dataset_config, create_file_name
    from textgnn.utils import get_saved_path

    # Extract config parameters
    gnn_encoding = dataset_config.gnn_encoding
    model_params = model_config.model_specific_params

    # Get input/output dimensions
    input_dim = gnn_encoding.embedding_dim if gnn_encoding else 300

    if dataset is not None:
        num_classes = dataset.num_classes
    else:
        # Load metadata from saved artifacts
        dataset_dir_name = create_dir_name_based_on_dataset_config(dataset_config)
        dataset_save_path = os.path.join(get_saved_path(), dataset_dir_name)
        save_fn = create_file_name(dataset_config, model_config.model_type)
        full_path = os.path.join(dataset_save_path, save_fn)

        # Load any split to get num_classes
        # train_data_path = os.path.join(full_path, "train_data.pkl")
        # if not os.path.exists(train_data_path):
        #     raise FileNotFoundError(
        #         f"Artifacts not found at {train_data_path}. "
        #         "Ensure artifacts are created by calling load_data() first."
        #     )

        # with open(train_data_path, 'rb') as f:
        #     data = pickle.load(f)
        num_classes = 20#data['num_classes']

    return TextINGClassifier(
        input_dim=input_dim,
        output_dim=num_classes,
        hidden_dim=model_params.get("hidden_dim", 96),
        gru_steps=model_params.get("gru_steps", 2),
        dropout=model_params.get("dropout", 0.5),
    )
