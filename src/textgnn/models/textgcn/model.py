"""TextGCN Model Implementation

Adapted from standalone TextGCN implementation to fit the framework architecture.
Uses Graph Convolutional Networks (GCN) for document classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import GCNConv as PyG_GCNConv
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros

from textgnn.models.base_text_classifier import GraphTextClassifier
from textgnn.config_class import ModelConfig, DatasetConfig
from textgnn.loaders.textgcn_loader import TextGCNDataset
from textgnn.utils import create_act

class TextGCNClassifier(GraphTextClassifier):
    """
    Text Classification using Graph Convolutional Networks.

    Implements the TextGCN model which constructs a heterogeneous graph with
    document and word nodes, using PMI for word-word edges and TF-IDF for
    document-word edges.
    """

    def __init__(
        self,
        num_nodes: int,
        num_classes: int,
        hidden_dims: list,
        x_type: str = "identity",
        pred_type: str = "softmax",
        act: str = "relu",
        use_bn: bool = True,
        dropout: float = 0.5,
        use_edge_weights: bool = True,
        **kwargs
    ):
        """
        Initialize TextGCN model.

        Args:
            num_nodes: Total number of nodes (documents + words)
            num_classes: Number of output classes
            hidden_dims: List of hidden dimensions for GCN layers
            x_type: Node feature type ("identity" for one-hot)
            pred_type: Prediction head type ("softmax" or "mlp")
            act: Activation function ("relu", "prelu", "sigmoid", "tanh")
            use_bn: Whether to use batch normalization
            dropout: Dropout rate (0 to disable)
            use_edge_weights: Whether to use edge weights (PMI/TF-IDF)
        """
        super().__init__(vocab_size=None, embedding_dim=None, output_size=num_classes)

        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.x_type = x_type
        self.pred_type = pred_type
        self.use_edge_weights = use_edge_weights
        self.dropout_rate = dropout

        # Cache for sparse identity matrix (created once, reused across forward passes)
        self._cached_identity = None

        # Build layer dimensions: [input_dim, *hidden_dims, output_dim]
        if pred_type == "softmax":
            # Last layer outputs num_classes directly
            self.layer_dims = [num_nodes] + hidden_dims + [num_classes]
        elif pred_type == "mlp":
            # Last GCN layer outputs hidden_dim, then MLP head
            self.layer_dims = [num_nodes] + hidden_dims
            mlp_dims = self._calc_mlp_dims(hidden_dims[-1], num_classes)
            self.mlp = MLP(
                input_dim=hidden_dims[-1],
                output_dim=num_classes,
                activation_type=act,
                num_hidden_lyr=len(mlp_dims),
                hidden_channels=mlp_dims,
                bn=False,
            )
        else:
            raise ValueError(f"Unknown pred_type: {pred_type}")

        # Create GCN layers
        self.num_layers = len(self.layer_dims) - 1
        self.layers = self._create_gcn_layers(act, use_bn, dropout)

        # Set training function (required by framework)
        from textgnn.models.textgcn.train import train_textgcn
        self.train_func = train_textgcn

    def _create_gcn_layers(self, act, use_bn, dropout):
        """Create list of GCN layers with activation, batch norm, and dropout."""
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            # Use identity activation for last layer
            layer_act = act if i < self.num_layers - 1 else "identity"
            # No dropout on first layer
            layer_dropout = dropout if i != 0 else 0.0

            layers.append(
                NodeEmbedding(
                    in_dim=self.layer_dims[i],
                    out_dim=self.layer_dims[i + 1],
                    act=layer_act,
                    bn=use_bn,
                    dropout=layer_dropout,
                )
            )
        return layers

    def _calc_mlp_dims(self, mlp_input_dim, output_dim):
        """Calculate MLP hidden layer dimensions (halving each time)."""
        dim = mlp_input_dim
        dims = []
        while dim > output_dim:
            dim = dim // 2
            dims.append(dim)
        # Remove last dimension (would be smaller than output)
        return dims[:-1] if dims else []

    def forward(self, data):
        """
        Forward pass through TextGCN.

        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features (None for identity)
                - edge_index: Edge connectivity
                - edge_attr: Edge weights (PMI/TF-IDF)
                - doc_mask: Boolean mask for document nodes
                - word_mask: Boolean mask for word nodes

        Returns:
            logits: Output logits for all nodes [num_nodes, num_classes]
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr if self.use_edge_weights else None

        # Forward through GCN layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)

        # Apply prediction head
        if self.pred_type == "softmax":
            logits = x  # Already has correct output dimension
        elif self.pred_type == "mlp":
            logits = self.mlp(x)
        else:
            raise ValueError(f"Unknown pred_type: {self.pred_type}")

        return logits


class NodeEmbedding(nn.Module):
    """
    GCN layer wrapper with activation, batch normalization, and dropout.
    """

    def __init__(self, in_dim, out_dim, act, bn, dropout):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Simple weight matrix for sparse multiplication (TensorFlow-style)
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.bias = nn.Parameter(torch.Tensor(out_dim))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

        # Cache for normalized adjacency (computed once)
        self.cached_adj = None

        # Activation function
        self.act = create_act(act, out_dim)

        # Batch normalization (optional)
        self.bn = nn.BatchNorm1d(out_dim) if bn else None

        # Dropout (optional)
        self.dropout_rate = dropout

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass - mimics TensorFlow TextGCN exactly.

        For featureless (x is sparse identity): compute A @ W directly
        For regular features: compute A @ (X @ W)
        """
        num_nodes = edge_index.max().item() + 1

        # Build normalized adjacency matrix (cached)
        if self.cached_adj is None:
            # Convert edge_index to sparse adjacency
            indices = edge_index
            values = edge_weight if edge_weight is not None else torch.ones(edge_index.shape[1], device=edge_index.device)

            # Add self-loops
            loop_index = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
            loop_weight = torch.ones(num_nodes, device=edge_index.device)
            indices = torch.cat([indices, loop_index], dim=1)
            values = torch.cat([values, loop_weight])

            # Create sparse adjacency
            adj = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes))

            # Normalize: D^(-1/2) @ A @ D^(-1/2)
            rowsum = torch.sparse.sum(adj, dim=1).to_dense()
            d_inv_sqrt = torch.pow(rowsum, -0.5)
            d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.

            # Create D^(-1/2) as sparse diagonal
            d_indices = torch.arange(num_nodes, device=edge_index.device).unsqueeze(0).repeat(2, 1)
            d_mat_inv_sqrt = torch.sparse_coo_tensor(d_indices, d_inv_sqrt, (num_nodes, num_nodes))

            # Normalize: D^(-1/2) @ A @ D^(-1/2)
            adj = torch.sparse.mm(torch.sparse.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
            self.cached_adj = adj.coalesce()

        # Check if input is sparse identity (featureless mode)
        is_featureless = (x is not None and
                         hasattr(x, 'is_sparse') and
                         x.is_sparse and
                         x.shape[0] == x.shape[1])

        if is_featureless or x is None:
            # Featureless mode: A @ W (skip identity multiplication!)
            output = torch.sparse.mm(self.cached_adj, self.weight)
        else:
            # Regular mode: A @ (X @ W)
            if self.dropout_rate > 0 and self.training:
                x = F.dropout(x, p=self.dropout_rate, training=self.training)
            h = torch.matmul(x, self.weight)
            output = torch.sparse.mm(self.cached_adj, h)

        # Add bias
        output = output + self.bias

        # Activation
        output = self.act(output)

        # Batch normalization
        if self.bn is not None:
            output = self.bn(output)

        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with configurable hidden layers.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        activation_type="relu",
        num_hidden_lyr=2,
        hidden_channels=None,
        bn=False,
    ):
        super().__init__()
        self.out_dim = output_dim

        # Set hidden layer dimensions
        if not hidden_channels:
            hidden_channels = [input_dim for _ in range(num_hidden_lyr)]
        elif len(hidden_channels) != num_hidden_lyr:
            raise ValueError(
                "number of hidden layers should be the same as the length of hidden_channels"
            )

        # Build layer dimensions
        self.layer_channels = [input_dim] + hidden_channels + [output_dim]
        self.activation = create_act(activation_type)

        # Create linear layers with Xavier initialization
        self.layers = nn.ModuleList()
        for i in range(len(self.layer_channels) - 1):
            layer = nn.Linear(self.layer_channels[i], self.layer_channels[i + 1])
            nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain("relu"))
            self.layers.append(layer)

        # Batch normalization (optional)
        self.bn = nn.BatchNorm1d(output_dim) if bn else None

    def forward(self, x):
        """Forward pass through MLP."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation to all but last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)

        # Apply batch normalization if enabled
        if self.bn is not None:
            x = self.bn(x)

        return x


def create_textgcn_model(
    model_config: ModelConfig,
    dataset_config: DatasetConfig,
    dataset: TextGCNDataset = None
):
    """
    Factory function to create TextGCN model.

    Args:
        model_config: Model configuration
        dataset_config: Dataset configuration
        dataset: TextGCNDataset instance (optional, used to get num_nodes and num_classes)

    Returns:
        TextGCNClassifier instance
    """
    import os
    import pickle
    from textgnn.load_data import create_dir_name_based_on_dataset_config, create_file_name
    from textgnn.utils import get_saved_path

    # Extract config parameters
    gnn_encoding = dataset_config.gnn_encoding
    model_params = model_config.model_specific_params

    # Get model parameters from dataset if available
    if dataset is not None:
        num_nodes = dataset.num_nodes
        num_classes = dataset.num_classes
    else:
        # Load metadata from saved artifacts
        dataset_dir_name = create_dir_name_based_on_dataset_config(dataset_config)
        dataset_save_path = os.path.join(get_saved_path(), dataset_dir_name)
        save_fn = create_file_name(dataset_config, model_config.model_type)
        full_path = os.path.join(dataset_save_path, save_fn)

        meta_path = os.path.join(full_path, "ALL_meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"Metadata file not found at {meta_path}. "
                "Ensure artifacts are created by calling load_data() first."
            )

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        num_nodes = meta["num_nodes"]
        num_classes = meta["num_classes"]

    return TextGCNClassifier(
        num_nodes=num_nodes,
        num_classes=num_classes,
        hidden_dims=model_params.get("hidden_dims", [200]),
        x_type=gnn_encoding.x_type if gnn_encoding else "identity",
        pred_type=model_params.get("pred_type", "softmax"),
        act=model_params.get("act", "relu"),
        use_bn=model_params.get("use_bn", True),
        dropout=model_params.get("dropout", 0.5),
        use_edge_weights=model_params.get("use_edge_weights", True),
    )
