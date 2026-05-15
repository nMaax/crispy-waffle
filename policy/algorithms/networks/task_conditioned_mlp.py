import torch
import torch.nn as nn


class TaskConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,  # e.g., 39 (from CanonicalPnPAdapter: 18 + 7 + 7 + 7)
        output_dim: int,  # e.g., 48 (StackCube-v1 format)
        num_tasks: int,  # Number of unique envs/tasks
        embed_dim: int = 16,  # Size of the task embedding
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()

        # TODO: Maybe rename variables to be just "conditioning" instead of "task conditioning" since it can be used for other types of conditioning as well
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)

        layers = []
        current_dim = input_dim + embed_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.ReLU())
            current_dim = h_dim

        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, task_idx: torch.Tensor) -> torch.Tensor:
        task_emb = self.task_embedding(task_idx)

        if x.dim() > task_emb.dim():
            task_emb = task_emb.unsqueeze(1).expand(-1, x.shape[1], -1)

        x_cond = torch.cat([x, task_emb], dim=-1)
        return self.net(x_cond)
