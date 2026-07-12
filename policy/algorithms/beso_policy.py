from collections import deque

import hydra_zen
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, EDMEulerScheduler

from policy.algorithms import DiffusionPolicy
from policy.utils import concat_leaf_tensors, get_batch_size, get_total_dim


class BesoPolicy(DiffusionPolicy):
    """Trains a BESO diffusion model to predict action sequences from observation histories.

    BESO as in Goal-Conditioned Imitation Learning using Score-based Diffusion Policies, Reuss et al. (2023)

    Reference:
        - Arxiv: https://arxiv.org/abs/2304.02532
        - Paper website: https://intuitive-robots.github.io/beso-website/
    """

    def __init__(
        self,
        *args,
        sigma_data: float = 0.5,
        sigma_mean: float = -1.2,
        sigma_std: float = 1.2,
        sigma_churn: float = 0.0,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)

        self.sigma_data = sigma_data
        self.sigma_mean = sigma_mean
        self.sigma_std = sigma_std
        self.sigma_churn = sigma_churn

        self.action_history = deque(maxlen=self.obs_horizon - 1)

        if "DiffusionGPT" not in self.network_config.get("_target_", None):
            raise ValueError(
                f"BesoPolicy requires a DiffusionGPT network to run. Found: {type(self.network)}"
            )

        if not isinstance(self.noise_scheduler, EDMEulerScheduler | DDIMScheduler):
            raise ValueError(
                f"BesoPolicy requires an EDMEulerScheduler or DDIMScheduler for inference. Found: {type(self.noise_scheduler)}"
            )

        self.network_cond_dim = get_total_dim(self.obs_dim)

    def configure_optimizers(self):
        """BESO custom optimizer configuration with weight decay handling for DiffusionGPT."""
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before configure_optimizers."
            )

        # Separate parameters into decay/no_decay sets
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Iterate over the DiffusionGPT network
        for mn, m in self.network.named_modules():
            for pn, p in m.named_parameters():
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        # Positional embedding in DiffusionGPT
        if hasattr(self.network, "pos_emb"):
            no_decay.add("pos_emb")

        # Validate that we categorized every parameter
        param_dict = {pn: p for pn, p in self.network.named_parameters() if p.requires_grad}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {inter_params} made it into both sets!"
        assert len(param_dict.keys() - union_params) == 0, "Some parameters were not categorized!"

        # Extract the weight decay value from the Hydra config (.keywords dict)
        optimizer_partial = hydra_zen.instantiate(self.optimizer_config)
        global_weight_decay = optimizer_partial.keywords.get("weight_decay", 1e-4)

        # Create PyTorch optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": global_weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,  # Explicitly disable weight decay for these!
            },
        ]

        optimizer = optimizer_partial(optim_groups)

        if self.lr_scheduler_config is not None:
            lr_scheduler_partial = hydra_zen.instantiate(self.lr_scheduler_config)
            lr_scheduler = lr_scheduler_partial(optimizer)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            return optimizer

    def reset(self):
        """Clears the action history.

        Call this at the start of every new rollout episode.
        """
        self.action_history = deque(maxlen=self.obs_horizon - 1)

    def _compute_loss(self, obs_seq: torch.Tensor, act_seq: torch.Tensor) -> torch.Tensor:
        """BESO Loss formulation using Karras preconditioning."""

        # NOTE: No need to normalize obs_seq here, as it is already normalized in the Diffusion Policy _shared_step()

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
        noise[:, :-1, :] = 0.0
        noisy_act_seq = act_seq + noise * sigma_bd

        # Get Karras scalings
        c_skip, c_out, c_in = self._get_karras_scalings(sigma_bd)

        # Predict raw model output (Network is now just DiffusionGPT)
        scaled_noisy_act = noisy_act_seq * c_in
        model_output = self.network(sample=scaled_noisy_act, timestep=sigmas, obs=obs_seq)

        # Compute the Karras target (reversing the skip connection)
        target = (act_seq - c_skip * noisy_act_seq) / c_out

        # Compute MSE loss
        loss = F.mse_loss(model_output[:, -1:, :], target[:, -1:, :])

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

    def _run_beso_diffusion_loop(
        self,
        network_cond: torch.Tensor,
        num_inference_steps: int = 20,
        clamp_range: tuple | None = None,
    ):
        """Generic helper containing the actual Karras preconditioned diffusion process loop."""
        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        if self.noise_scheduler is None:
            raise ValueError(
                "Noise Scheduler not initialized. Call configure_model() before getting action."
            )

        if not hasattr(self.noise_scheduler, "init_noise_sigma"):
            raise ValueError(
                "Noise Scheduler does not have 'init_noise_sigma' attribute. Ensure you are using EDMEulerScheduler."
            )

        if not hasattr(self.noise_scheduler, "sigmas"):
            raise ValueError(
                "Noise Scheduler does not have 'sigmas' attribute. Ensure you are using EDMEulerScheduler."
            )

        if self.ema is None:
            raise ValueError(
                "EMA Model not initialized. Call configure_model() before getting action."
            )

        B = get_batch_size(network_cond)

        # If the episode just started, pad the history with zeros
        while len(self.action_history) < self.obs_horizon - 1:
            self.action_history.append(torch.zeros((B, 1, self.act_dim), device=self.device))

        # Combine the deque into a single [B, 7, act_dim] tensor
        clean_past_actions = torch.cat(list(self.action_history), dim=1)

        self.ema.store(self.network.parameters())
        self.ema.copy_to(self.network.parameters())

        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)

        with torch.no_grad():
            # Only the CURRENT action (1 token) gets initialized with pure noise
            current_noisy_action = (
                torch.randn((B, 1, self.act_dim), device=self.device)
                * self.noise_scheduler.init_noise_sigma
            )

            for i, t in enumerate(self.noise_scheduler.timesteps):
                # 't' is the HF transformed timestep (0.25 * log(sigma))
                # 'raw_sigma' is the actual standard deviation we need for DiffusionGPT
                raw_sigma = self.noise_scheduler.sigmas[i].to(self.device).expand(B)

                # Assemble the sequence: [Clean History (7), Noisy Current (1)]
                combined_act_seq = torch.cat([clean_past_actions, current_noisy_action], dim=1)

                # Precondition the full sequence (Diffusers expects 't')
                scaled_sample = self.noise_scheduler.scale_model_input(combined_act_seq, t)

                # Get raw network prediction (DiffusionGPT expects 'raw_sigma')
                model_pred = self.network(
                    sample=scaled_sample, timestep=raw_sigma, obs=network_cond
                )

                # We only extract and step the derivative for the final noisy token
                current_noisy_action_pred = model_pred[:, -1:, :]

                # Step the scheduler using the raw predicted action (Diffusers expects 't')
                output = self.noise_scheduler.step(
                    model_output=current_noisy_action_pred,
                    timestep=t,
                    sample=current_noisy_action,
                    s_churn=self.sigma_churn,
                    return_dict=False,
                )
                current_noisy_action = output[0]

        self.ema.restore(self.network.parameters())

        # Extract the actions to execute
        denoised_action = current_noisy_action

        if clamp_range is not None:
            low, high = clamp_range
            denoised_action = torch.clamp(denoised_action, low, high)

        # Save the executed action to the history queue for the next step
        self.action_history.append(denoised_action)

        return denoised_action

    def _prepare_network_cond(self, obs_seq: dict | torch.Tensor) -> torch.Tensor:
        """Prepares the observation sequence for the BESO network conditioning (keeps sequence
        format)."""
        return concat_leaf_tensors(obs_seq, device=self.device)
