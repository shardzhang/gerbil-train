"""Reusable neural network layers for model construction."""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["ActivationUnit", "Dice", "get_activation", "FullyConnectedLayer"]


def get_activation(name: str, num_features: int | None = None, dice_dim: int | None = None) -> nn.Module:
    """Get the activation function by name.

    :param name: Activation name (case-insensitive).
        Supported: ``relu``, ``gelu``, ``silu`` / ``swish``,
        ``leaky_relu``, ``prelu``, ``tanh``, ``dice``.
    :return: An nn.Module representing the activation function
    """
    name = name.lower().replace("-", "_")
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if name == "prelu":
        return nn.PReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "dice":
        return Dice(num_features, dim=dice_dim)
    raise ValueError(f"Unsupported activation: {name}")


class Dice(nn.Module):
    def __init__(self, num_features, dim=2):
        self.num_features = num_features
        self.dim = dim
        super(Dice, self).__init__()

        assert dim == 2 or dim == 3
    
        self.bn = nn.BatchNorm1d(num_features, eps=1e-9)
        self.sigmoid = nn.Sigmoid()
        self.dim = dim
        
        if self.dim == 3:
            self.alpha = nn.Parameter(torch.rand((num_features, 1)))
        elif self.dim == 2:
            self.alpha = nn.Parameter(torch.rand((num_features,)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dim == 3:
            # Input shape: [batch_size, seq_len, hidden_size]
            x = torch.transpose(x, 1, 2)
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
            out = torch.transpose(out, 1, 2)
        elif self.dim == 2:
            x_p = self.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        return out


class FullyConnectedLayer(nn.Module):
    """Multi-layer perceptron with optional batch norm, dropout, and sigmoid output."""
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        bias: list[bool],
        batch_norm: bool = True,
        dropout: float = 0.1,
        activation: str = "relu",
        sigmoid: bool = False,
    ):
        super().__init__()
        assert len(hidden_dims) >= 1 and len(bias) >= 1, "hidden_dims and bias must be non-empty lists"
        assert len(bias) == len(hidden_dims), "bias must have the same length as hidden_dims"
        self.sigmoid = sigmoid

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=bias[i]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = hidden_dim

        self.fc: nn.Sequential = nn.Sequential(*layers)
        if self.sigmoid:
            self.output_layer = nn.Sigmoid()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sigmoid:
            return self.output_layer(self.fc(x))
        return self.fc(x)


class ActivationUnit(nn.Module):
    """Local Activation Unit (LAU) from the DIN paper.

    Computes attention scores between a target item and each item in a behavior sequence:

        score_i = MLP(concat(behavior_i, target, behavior_i * target))

    :param input_emb_dim: Embedding dimension of both behavior and target items
    :param hidden_dim: Hidden layer size of the attention MLP (default: 36)
    """

    def __init__(self, input_emb_dim: int, hidden_dim: int = 36) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_emb_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, behavior_embs: Tensor, target_emb: Tensor) -> Tensor:
        """Compute raw attention scores for each behavior item.

        :param behavior_embs: ``[total_items, emb_dim]``
        :param target_emb: ``[total_items, emb_dim]``
        :return: ``[total_items]``
        """
        attn_input = torch.cat([behavior_embs, target_emb, behavior_embs * target_emb], dim=-1)
        return self.mlp(attn_input).squeeze(-1)

    def pool(
        self,
        behavior_embs: Tensor,
        target_emb: Tensor,
        lengths: Tensor,
    ) -> Tensor:
        """Attention pooling on padded behavior sequences.

        Accepts already-padded sequences, making it agnostic to the upstream
        data format (TFRecord offsets, packed sequences, etc).

        :param behavior_embs: Padded behavior item embeddings ``[batch, seq_len, emb_dim]``
        :param target_emb: Target item embedding ``[batch, emb_dim]``
        :param lengths: Actual sequence length per sample ``[batch]``
        :return: Interest embedding ``[batch, emb_dim]``
        """
        batch, seq_len, emb_dim = behavior_embs.shape
        target_expanded = target_emb.unsqueeze(1).expand(-1, seq_len, -1)

        scores = self.forward(
            behavior_embs.reshape(-1, emb_dim),
            target_expanded.reshape(-1, emb_dim),
        ).reshape(batch, seq_len)

        mask = torch.arange(seq_len, device=behavior_embs.device).unsqueeze(0) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, -float("inf"))
        has_items = lengths > 0
        scores = scores.masked_fill(~has_items.unsqueeze(1), 0.0)
        weights = torch.softmax(scores, dim=-1)

        return (weights.unsqueeze(-1) * behavior_embs).sum(dim=1)
