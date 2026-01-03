"""TextGCN Model Implementation

Adapted from standalone TextGCN implementation to fit the framework architecture.
Uses Graph Convolutional Networks (GCN) for document classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_geometric.nn.inits import glorot, zeros

from textgnn.models.base_text_classifier import GraphTextClassifier
from textgnn.config_class import ModelConfig, DatasetConfig
from textgnn.loaders.textgcn_loader import TextGCNDataset

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
        super().__init__(vocab_size=num_nodes, embedding_dim=num_nodes, output_size=num_classes)

        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.x_type = x_type
        self.pred_type = pred_type
        self.use_edge_weights = use_edge_weights
        self.dropout_rate = dropout

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

        # Initialize node features if using identity (one-hot)
        if x is None and self.x_type == "identity":
            # Create sparse identity matrix for efficiency
            x = torch.eye(self.num_nodes, device=edge_index.device, dtype=torch.float32)

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

        # GCN convolution layer
        self.conv = GCNConv(in_dim, out_dim)

        # Activation function
        self.act = create_act(act, out_dim)

        # Batch normalization (optional)
        self.bn = nn.BatchNorm1d(out_dim) if bn else None

        # Dropout (optional)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x, edge_index, edge_weight=None):
        """Forward pass through GCN layer."""
        # Apply dropout to input
        if self.dropout is not None:
            x = self.dropout(x)

        # GCN convolution
        x = self.conv(x, edge_index, edge_weight=edge_weight)

        # Activation
        x = self.act(x)

        # Batch normalization
        if self.bn is not None:
            x = self.bn(x)

        return x


class GCNConv(MessagePassing):
    """
    Graph Convolutional Network layer.

    Implements the GCN operator from Kipf & Welling (2017):
    X' = D^(-1/2) A D^(-1/2) X W + b

    where A is the adjacency matrix with self-loops added.
    """

    def __init__(self, in_channels, out_channels, improved=False, cached=False, bias=True):
        super().__init__(aggr="add")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        # Weight matrix
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        # Bias vector (optional)
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        glorot(self.weight)
        zeros(self.bias)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        """
        Compute normalized edge weights: D^(-1/2) A D^(-1/2).

        Args:
            edge_index: Edge connectivity [2, num_edges]
            num_nodes: Total number of nodes
            edge_weight: Edge weights [num_edges]
            improved: If True, use 2*I instead of I for self-loops
            dtype: Data type for edge weights

        Returns:
            edge_index: Edge connectivity with self-loops
            edge_weight: Normalized edge weights
        """
        if edge_weight is None:
            edge_weight = torch.ones(
                (edge_index.size(1),), dtype=dtype, device=edge_index.device
            )

        # Remove existing self-loops
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        # Add self-loops
        fill_value = 2.0 if improved else 1.0
        edge_index, edge_weight = add_self_loops(
            edge_index,
            edge_attr=edge_weight,
            fill_value=fill_value,
            num_nodes=num_nodes
        )

        # Compute degree
        row, col = edge_index
        deg = torch.bincount(row, weights=edge_weight, minlength=num_nodes).float()

        # Compute D^(-1/2)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

        # Normalize: D^(-1/2) * edge_weight * D^(-1/2)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through GCN layer.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge connectivity [2, num_edges]
            edge_weight: Edge weights [num_edges] (optional)

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        # Apply linear transformation
        if x.is_sparse:
            x = torch.sparse.mm(x, self.weight)
        else:
            x = torch.matmul(x, self.weight)

        # Compute normalized adjacency (cache if enabled)
        if not self.cached or self.cached_result is None:
            edge_index, norm = GCNConv.norm(
                edge_index, x.size(0), edge_weight, self.improved, x.dtype
            )
            if self.cached:
                self.cached_result = edge_index, norm
        else:
            edge_index, norm = self.cached_result

        # Message passing
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        """Construct messages: normalize neighbor features."""
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        """Update node embeddings: add bias."""
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_channels}, {self.out_channels})"


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


def create_act(act, num_parameters=None):
    """
    Create activation function module.

    Args:
        act: Activation type ("relu", "prelu", "sigmoid", "tanh", "identity")
        num_parameters: Number of parameters for PReLU

    Returns:
        nn.Module: Activation function
    """
    if act == "relu":
        return nn.ReLU()
    elif act == "prelu":
        return nn.PReLU(num_parameters)
    elif act == "sigmoid":
        return nn.Sigmoid()
    elif act == "tanh":
        return nn.Tanh()
    elif act == "identity":
        class Identity(nn.Module):
            def forward(self, x):
                return x
        return Identity()
    else:
        raise ValueError(f"Unknown activation function: {act}")


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
