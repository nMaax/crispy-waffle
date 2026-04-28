import torch


class TensorZNormalizer(torch.nn.Module):
    def __init__(self, mean, std, clip=5.0, eps=1e-6):
        super().__init__()

        mean = torch.as_tensor(mean, dtype=torch.float32)
        std = torch.as_tensor(std, dtype=torch.float32)
        inv_std = 1.0 / std.clamp(min=eps)

        self.register_buffer("mean", mean)
        self.register_buffer("inv_std", inv_std)

        self.clip = clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(dtype=torch.float32)
        y = (x - self.mean) * self.inv_std

        if self.clip is not None:
            y = torch.clamp(y, -self.clip, self.clip)

        return y

    def unnormalize(self, y: torch.Tensor) -> torch.Tensor:
        x = y / self.inv_std + self.mean

        return x
