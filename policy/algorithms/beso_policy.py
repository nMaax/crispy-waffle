
import torch
import torch.nn.functional as F
from diffusers import EDMEulerScheduler

from policy.algorithms import DiffusionPolicy
from policy.utils import flatten_tensor_from_mapping, get_batch_size


class BesoPolicy(DiffusionPolicy):
    """Trains a Behavioral Cloning via Score-Based Diffusion (BESO) policy.

    Replaces custom k-diffusion samplers with Hugging Face's EDMEulerScheduler.
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # Ensure we are using Karras scheduler
        if not isinstance(self.noise_scheduler, EDMEulerScheduler):
            raise ValueError(
                f"BesoPolicy requires an EDMEulerScheduler (Karras). Found: {type(self.noise_scheduler)}"
            )

    def _sample_lognormal_sigmas(
        self, batch_size: int, mean: float = -1.2, std: float = 1.2
    ) -> torch.Tensor:
        """Draws training sigmas from a log-normal distribution as per Karras et al."""
        rnd_normal = torch.randn(batch_size, device=self.device)
        return (rnd_normal * std + mean).exp()

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        num_inference_steps: int = 20,
        clamp_range: tuple | None = None,
    ):
        """Runs the Karras EDM reverse diffusion process via Hugging Face."""

        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        if self.noise_scheduler is None:
            raise ValueError(
                "Noise Scheduler not initialized. Call configure_model() before getting action."
            )

        if self.ema is None:
            raise ValueError(
                "EMA Model not initialized. Call configure_model() before getting action."
            )

        B = get_batch_size(obs_seq)
        obs_seq = flatten_tensor_from_mapping(obs_seq, device=self.device)

        self.ema.store(self.network.parameters())
        self.ema.copy_to(self.network.parameters())

        # Set up the HF EDM Scheduler discrete timesteps (which map to sigmas)
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            # Initial noise scaled by the maximum noise level (sigma_max)
            noisy_act_seq = (
                torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
                * self.noise_scheduler.init_noise_sigma  # sigma_max
            )

            for t in self.noise_scheduler.timesteps:
                # In EDM, the timestep itself is the sigma value
                sigma = t.expand(B)

                # We do not need to call noise_scheduler.scale_model_input() here
                # because our KarrasDenoiserWrapper explicitly handles the c_in scaling internally!

                # Get model prediction (Our wrapper outputs the denoised x_0)
                model_pred = self.network(sample=noisy_act_seq, timestep=sigma, obs=obs_seq)

                # HF Scheduler step maps the predicted x_0 and current noise to the previous noise level
                output = self.noise_scheduler.step(
                    model_output=model_pred,
                    timestep=t,
                    sample=noisy_act_seq,
                    return_dict=False,
                )
                noisy_act_seq = output[0]

        self.ema.restore(self.network.parameters())

        # Extract the actions to execute
        start = self.obs_horizon - 1
        end = start + self.act_horizon
        denoised_act_seq = noisy_act_seq[:, start:end]

        if clamp_range is not None:
            low, high = clamp_range
            denoised_act_seq = torch.clamp(denoised_act_seq, low, high)

        return denoised_act_seq

    def _compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """BESO Loss formulation.

        The wrapper handles the preconditioner scaling.
        """
        if self.network is None or self.ema is None:
            raise ValueError("Initialize network and EMA via configure_model() first.")

        B = obs_seq.shape[0]

        # Sample (continuous) noise levels
        sigmas = self._sample_lognormal_sigmas(B)

        # Add noise
        noise = torch.randn_like(act_seq)
        sigma_bd = sigmas.view(B, 1, 1)
        noisy_act_seq = act_seq + noise * sigma_bd

        # Predict denoised sequence (Karras wrapper returns predicted x_0)
        denoised_pred = self.network(sample=noisy_act_seq, timestep=sigmas, obs=obs_seq)

        # K-diffusion training computes MSE directly against the clean target
        loss = F.mse_loss(denoised_pred, act_seq)
        return loss
