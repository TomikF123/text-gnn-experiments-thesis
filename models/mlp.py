import torch
import torch.nn as nn
import utils
from utils import create_act


class MLP(nn.Module):
    def __init__(
        self, in_dim, hidden_dims, out_dim, act="tanh", use_bn=False, dropout=0.0
    ):
        super().__init__()
        Act = create_act(act)
        self.use_bn = use_bn
        self.act = Act()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        dims = [in_dim] + list(hidden_dims) + [out_dim]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        # BN modules only for hidden layers
        self.bns = nn.ModuleList(
            [nn.BatchNorm1d(d) if use_bn else nn.Identity() for d in hidden_dims]
        )

    def forward(self, x):
        # Hidden layers
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.bns[i](x)
            x = self.act(x)
            x = self.dropout(x)
        # Output layer (no BN/activation)
        x = self.layers[-1](x)
        return x
    
    #TODO
    def reset_parameters(self,act = "relu"):
        pass


if __name__ == "__main__":
    # Example usage
    model = MLP(
        in_dim=10, hidden_dims=[20, 30], out_dim=5, act="relu", use_bn=True, dropout=0.1
    )
    x = torch.randn(2, 10)  # Batch of 2 samples with 10 features each
    output = model(x)
    print(output.shape, output)  # Should print torch.Size([2, 5])
