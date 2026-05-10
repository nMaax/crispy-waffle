import torch

from policy.algorithms.mlp_adapter import MLPAdapter


class LearnedMLPAdapter:
    def __init__(
        self, checkpoint_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = MLPAdapter.load_from_checkpoint(checkpoint_path)

        self.model.to(device)
        self.model.eval()
        self.model.freeze()
        self.device = device

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        obs = obs.to(self.device)

        with torch.no_grad():
            return self.model(obs)
