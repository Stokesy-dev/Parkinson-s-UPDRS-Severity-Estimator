"""
model.py
--------
Feature-Attention MLP for UPDRS severity regression.

Novel contribution: a learnable softmax attention layer over input features
that explicitly weights which voice biomarkers drive the severity prediction.
These attention weights feed directly into SHAP explanations (Week 3).

Architecture:
    Input (n_features)
        → FeatureAttention (learnable per-feature softmax weights)
        → FC(256) + BN + ReLU + Dropout
        → FC(128) + BN + ReLU + Dropout
        → FC(64)  + ReLU
        → Output (1, scalar UPDRS score)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAttention(nn.Module):
    """
    Learnable attention weights over input features.

    For each input x of shape (batch, n_features):
    - Compute attention scores via a linear projection
    - Apply softmax to get normalized weights (sum to 1)
    - Return element-wise weighted input

    These weights are interpretable: high attention = feature
    is globally important for UPDRS prediction across the dataset.
    Per-sample importance is captured by SHAP (explain.py).
    """

    def __init__(self, n_features: int):
        super().__init__()
        self.attention = nn.Linear(n_features, n_features, bias=True)

    def forward(self, x: torch.Tensor):
        scores = self.attention(x)
        weights = torch.softmax(scores / 0.1, dim=-1)  # temperature=0.1 softens distribution
        attended = x * weights
        return attended, weights

class FeatureAttentionMLP(nn.Module):
    """
    Feature-Attention MLP for UPDRS regression.

    Args:
        n_features: number of input voice biomarker features
        hidden_dims: tuple of hidden layer sizes
        dropout: dropout probability applied after each hidden layer
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        self.n_features = n_features

        # Attention layer (the novel part)
        self.feature_attention = FeatureAttention(n_features)

        # Build FC layers
        layers = []
        in_dim = n_features
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            # No BN on last hidden layer before output
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.fc_layers = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        attended, attn_weights = self.feature_attention(x)
        hidden = self.fc_layers(attended)
        out = self.output(hidden).squeeze(-1)   # (batch,)

        if return_attention:
            return out, attn_weights
        return out

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return attention weights for a batch. Used for analysis."""
        with torch.no_grad():
            _, weights = self.feature_attention(x)
        return weights


class PlainMLP(nn.Module):
    """
    Baseline MLP without attention — used for ablation study.
    Identical architecture minus the FeatureAttention layer.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        in_dim = n_features
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(in_dim, h_dim))
            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        self.fc_layers = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor, **kwargs):
        hidden = self.fc_layers(x)
        return self.output(hidden).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick sanity check
    n_features = 19
    batch_size = 32

    model = FeatureAttentionMLP(n_features=n_features)
    x = torch.randn(batch_size, n_features)

    out, attn = model(x, return_attention=True)

    print(f"Model: FeatureAttentionMLP")
    print(f"  Input shape : {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Attn shape  : {attn.shape}")
    print(f"  Attn sum    : {attn[0].sum().item():.4f}  (should be ~1.0)")
    print(f"  Parameters  : {count_parameters(model):,}")

    baseline = PlainMLP(n_features=n_features)
    print(f"\nBaseline PlainMLP parameters: {count_parameters(baseline):,}")
