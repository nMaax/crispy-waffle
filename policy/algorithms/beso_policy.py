import hydra_zen
import torch
import torch.nn.functional as F
from diffusers import EDMEulerScheduler

from policy.algorithms import DiffusionPolicy
from policy.utils import flatten_tensor_from_mapping, get_batch_size

# TODO: review BESO code and compare


class BesoPolicy(DiffusionPolicy):
    """Trains a Behavioral Cloning via Score-Based Diffusion (BESO) policy.

    Replaces custom k-diffusion samplers with Hugging Face's EDMEulerScheduler.
    """

    def __init__(
        self,
        *args,
        sigma_data: float = 0.5,
        sigma_mean: float = -1.2,
        sigma_std: float = 1.2,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.sigma_data = sigma_data
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std

        # Ensure we are using Karras scheduler
        if not isinstance(self.noise_scheduler, EDMEulerScheduler):
            raise ValueError(
                f"BesoPolicy requires an EDMEulerScheduler (Karras). Found: {type(self.noise_scheduler)}"
            )

    def configure_model(self) -> None:
        """Overrides DiffusionPolicy's configure_model to remove hardcoded U-Net dimensions.

        The Karras Wrapper and DiffusionGPT handle their own dimensions via the Hydra config.
        """
        if self.network is not None:
            return

        self.network = hydra_zen.instantiate(self.network_config)

        if self.ema is not None:
            return

        self.ema = hydra_zen.instantiate(self.ema_config, parameters=self.network.parameters())

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
            noisy_act_seq = (
                torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)
                * self.noise_scheduler.init_noise_sigma
            )

            for t in self.noise_scheduler.timesteps:
                # EDM timestep is the sigma value
                sigma = t.expand(B)
                sigma_bd = sigma.view(-1, 1, 1)

                # Get Karras scalings for current noise level
                c_skip, c_out, c_in = self._get_karras_scalings(sigma_bd)

                # Precondition the input
                scaled_sample = noisy_act_seq * c_in

                # Call to silence Diffusers warning (it just returns the input in EDM)
                _ = self.noise_scheduler.scale_model_input(noisy_act_seq, t)

                # Get raw network prediction
                model_pred = self.network(sample=scaled_sample, timestep=sigma, obs=obs_seq)

                # Apply skip connection to get the predicted clean action (x_0)
                x_0_pred = model_pred * c_out + noisy_act_seq * c_skip

                # Step the scheduler using the predicted clean action
                output = self.noise_scheduler.step(
                    model_output=x_0_pred,
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
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        B = obs_seq.shape[0]

        # Sample continuous noise levels
        sigmas = self._sample_lognormal_sigmas(B)
        sigma_bd = sigmas.view(B, 1, 1)

        # Add noise
        noise = torch.randn_like(act_seq)
        noisy_act_seq = act_seq + noise * sigma_bd

        # Get Karras scalings
        c_skip, c_out, c_in = self._get_karras_scalings(sigma_bd)

        # Predict raw model output (Network is now just DiffusionGPT)
        scaled_noisy_act = noisy_act_seq * c_in
        model_output = self.network(sample=scaled_noisy_act, timestep=sigmas, obs=obs_seq)

        # Compute the Karras target (reversing the skip connection)
        target = (act_seq - c_skip * noisy_act_seq) / c_out

        # Compute properly weighted MSE loss
        loss = F.mse_loss(model_output, target)

        return loss

    def _sample_lognormal_sigmas(self, batch_size: int) -> torch.Tensor:
        """Draws training sigmas from a log-normal distribution as per Karras et al."""
        rnd_normal = torch.randn(batch_size, device=self.device)
        return (rnd_normal * self.sigma_std + self.sigma_mean).exp()

    def _get_karras_scalings(self, sigma: torch.Tensor):
        """Computes the Karras preconditioning factors."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in
