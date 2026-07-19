from typing import cast

import torch
import torch.nn.functional as F

from policy.algorithms.base_diffusion_agent import BaseDiffusionAgent
from policy.utils import get_batch_size


class DiffusionPolicy(BaseDiffusionAgent):
    """Trains a diffusion policy to predict action sequences from observation histories.

    Diffusion Policy as in Cheng et. al (IJRR)

    Reference:
        - Arxiv: https://arxiv.org/abs/2303.04137v4
        - Paper website: https://diffusion-policy.cs.columbia.edu/
        - Maniskill implementation: https://github.com/haosulab/ManiSkill/tree/main/examples/baselines/diffusion_policy
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.ema_config is None:
            raise ValueError(
                "DiffusionPolicy requires an EMA model. Pass a valid `ema` config."
            )

    def _compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """Samples noise, adds it to the target sequence, and computes the reconstruction loss.

        Shapes:
            obs_seq: [B, obs_horizon * obs_dim] (flattened condition sequence)
            act_seq: [B, pred_horizon, act_dim] (target action chunk)
            returns: scalar loss tensor []
        """
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before computing loss."
            )

        if self.noise_scheduler is None:
            raise ValueError(
                "Noise Scheduler not initialized. Call configure_model() before computing loss."
            )

        B = obs_seq.shape[0]

        noise = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config["num_train_timesteps"],
            (B,),
            device=self.device,
            dtype=torch.int32,
        )
        timesteps = cast(torch.IntTensor, timesteps)

        noisy_act_seq = self.noise_scheduler.add_noise(act_seq, noise, timesteps)
        prediction = self.network(noisy_act_seq, timesteps, obs=obs_seq)

        pred_type = self.noise_scheduler.config.get("prediction_type", "epsilon")

        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = act_seq
        elif pred_type == "v_prediction":
            target = self.noise_scheduler.get_velocity(act_seq, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type}")

        loss = F.mse_loss(prediction, target)
        return loss

    def _run_diffusion_loop(
        self,
        network_cond: torch.Tensor,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        """Generic helper containing the actual reverse diffusion process loop."""
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

        B = get_batch_size(network_cond)

        self.ema.store(self.network.parameters())
        self.ema.copy_to(self.network.parameters())

        if num_inference_steps is None:
            num_inference_steps = int(self.noise_scheduler.config["num_train_timesteps"])

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            noisy_act_seq = torch.randn((B, self.pred_horizon, self.act_dim), device=self.device)

            for t in self.noise_scheduler.timesteps:
                t = int(t.item())

                latent_model_input = self.noise_scheduler.scale_model_input(noisy_act_seq, t)

                model_pred = self.network(
                    sample=latent_model_input,
                    timestep=t,
                    obs=network_cond,
                )

                output = self.noise_scheduler.step(
                    model_output=model_pred,
                    timestep=t,
                    sample=noisy_act_seq,
                    return_dict=False,
                )

                noisy_act_seq = output[0]

        self.ema.restore(self.network.parameters())

        start = self.obs_horizon - 1
        end = start + self.act_horizon

        denoised_act_seq = noisy_act_seq[:, start:end]
        if self.act_normalizer is not None:
            denoised_act_seq = self.act_normalizer.unnormalize(denoised_act_seq)
        if output_clip_range is not None:
            low, high = output_clip_range
            denoised_act_seq = torch.clamp(denoised_act_seq, low, high)

        return denoised_act_seq
