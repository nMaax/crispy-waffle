import torch
import torch.nn as nn


class ConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_conditions: int,
        embed_dim: int = 16,
        hidden_dims: list[int] = [256, 256],
        bias: bool = True,
    ):
        super().__init__()

        self.cond_embedding = nn.Embedding(num_conditions, embed_dim)

        layers = []
        current_dim = input_dim + embed_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim, bias=bias))
            layers.append(nn.ReLU())
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, cond_idx: torch.Tensor) -> torch.Tensor:
        cond_emb = self.cond_embedding(cond_idx)

        if x.dim() > cond_emb.dim():
            cond_emb = cond_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        x_cond = torch.cat([x, cond_emb], dim=-1)
        return self.net(x_cond)
