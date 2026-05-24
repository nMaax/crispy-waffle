import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


class ResWrapper(torch.nn.Module):
    def __init__(self, module, dim_res=2):
        super().__init__()
        self.module = module
        self.dim_res = dim_res

    def forward(self, x):
        res = x[:, : self.dim_res]
        out = self.module(x)
        return out + res


class EGNN(MessagePassing):
    """Fixed EGNN layer from https://arxiv.org/pdf/2102.09844.pdf"""

    def __init__(
        self, channels_h, channels_m, channels_a, aggr="add", hidden_channels=128, **kwargs
    ):
        super().__init__(aggr=aggr, **kwargs)

        self.phi_e = nn.Sequential(
            nn.Linear(2 * channels_h + 1 + channels_a, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, channels_m),
            nn.LayerNorm(channels_m),
            nn.SiLU(),
        )
        self.phi_x = nn.Sequential(
            nn.Linear(channels_m, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )
        self.phi_h = ResWrapper(
            nn.Sequential(
                nn.Linear(channels_h + channels_m, hidden_channels),
                nn.LayerNorm(hidden_channels),
                nn.SiLU(),
                nn.Linear(hidden_channels, channels_h),
            ),
            dim_res=channels_h,
        )

    def forward(self, x, h, edge_attr, edge_index, c=None):
        if c is None:
            c = degree(edge_index[0], x.shape[0]).unsqueeze(-1)
        return self.propagate(edge_index=edge_index, x=x, h=h, edge_attr=edge_attr, c=c)

    def message(self, x_i, x_j, h_i, h_j, edge_attr):
        mh_ij = self.phi_e(
            torch.cat(
                [h_i, h_j, torch.norm(x_i - x_j, dim=-1, keepdim=True) ** 2, edge_attr], dim=-1
            )
        )
        mx_ij = (x_i - x_j) * self.phi_x(mh_ij)
        return torch.cat((mx_ij, mh_ij), dim=-1)

    def update(self, aggr_out, x, h, edge_attr, c):
        m_x, m_h = aggr_out[:, :3], aggr_out[:, 3:]
        h_l1 = self.phi_h(torch.cat([h, m_h], dim=-1))
        x_l1 = x + (m_x / c)
        return x_l1, h_l1


class SiameseEGNNPlanner(nn.Module):
    """Wraps the EGNN to handle dense PyTorch batches and outputs the Plan Embedding."""

    def __init__(self, num_nodes=3, channels_h=7, channels_m=128, out_dim=64):
        super().__init__()
        self.num_nodes = num_nodes
        self.channels_h = channels_h
        self.channels_m = channels_m

        self.egnn_layers = nn.ModuleList(
            [
                EGNN(channels_h=channels_h, channels_m=channels_m, channels_a=1),
                EGNN(channels_h=channels_h, channels_m=channels_m, channels_a=1),
                EGNN(channels_h=channels_h, channels_m=channels_m, channels_a=1),
            ]
        )

        self.pool = nn.Linear(num_nodes * channels_h, out_dim)

    def forward(self, coords: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """
        coords: [Batch, Nodes, 3]
        feats: [Batch, Nodes, Features]
        """
        B = coords.shape[0]
        device = coords.device

        # Flatten (B * Nodes, Features)
        x = coords.view(-1, 3)
        h = feats.view(-1, self.channels_h)

        # Build the edge_index for a fully connected graph of 3 nodes (no self-loops)
        base_edges = torch.tensor(
            [[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long, device=device
        )

        # Replicate the edges for every item in the batch
        edge_index = []
        for batch_idx in range(B):
            edge_index.append(base_edges + batch_idx * self.num_nodes)
        edge_index = torch.cat(edge_index, dim=1)

        # Create dummy edge attributes
        edge_attr = torch.zeros((edge_index.shape[1], 1), device=device)

        x_out, h_out = x, h
        for egnn_layer in self.egnn_layers:
            x_out, h_out = egnn_layer(x_out, h_out, edge_attr, edge_index)

        # Reshape back to a dense batch and pool
        h_out = h_out.view(B, self.num_nodes, self.channels_h)
        plan_embedding = self.pool(h_out.view(B, -1))

        return plan_embedding
