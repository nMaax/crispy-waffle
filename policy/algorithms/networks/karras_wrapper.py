import torch
import torch.nn as nn


class KarrasDenoiserWrapper(nn.Module):
    """A Karras et al.

    preconditioner for denoising diffusion models. Adapts BESO's GCDenoiser to your (sample,
    timestep, obs) interface.
    """

    def __init__(self, inner_model: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        # Instantiate the inner DiffusionGPT using Hydra config
        self.inner_model = inner_model
        self.sigma_data = sigma_data

    def get_scalings(self, sigma: torch.Tensor):
        """Computes the scalings for the denoising process based on the current noise level."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(
        self, sample: torch.Tensor, timestep: torch.Tensor, obs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sample: [B, pred_horizon, act_dim] (Noisy actions)
            timestep: [B] (Continuous sigma noise levels)
            obs: [B, obs_horizon * obs_dim] (Flattened conditioning)
        """
        # Ensure sigma is reshaped to broadcast across the sequence and action dimensions: [B, 1, 1]
        sigma_bd = timestep.view(-1, 1, 1)

        c_skip, c_out, c_in = self.get_scalings(sigma_bd)

        # Precondition the noisy input
        scaled_sample = sample * c_in

        # Forward pass through the inner Transformer
        # Note: We pass the raw `timestep` (1D tensor) to the inner model for its own sigma embeddings
        model_output = self.inner_model(sample=scaled_sample, timestep=timestep, obs=obs)

        # Scale output and add the skip connection to stabilize variance
        denoised_pred = model_output * c_out + sample * c_skip

        return denoised_pred
