import math
from collections import deque

import hydra_zen
import torch
import torch.nn.functional as F

from policy.algorithms import DiffusionPolicy
from policy.utils import get_batch_size


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
        sigma_churn: float = 0.0,
        sigma_min: float = 0.005,
        sigma_max: float = 1.0,
        pred_last_action_only: bool = False,
        **kwargs,
    ):

        # NOTE: we implement our own custom DDIM scheduler on continuous sigmas
        # so we can ignore the noise_scheduler argument from the base class

        kwargs.pop("noise_scheduler", None)
        super().__init__(*args, noise_scheduler=None, **kwargs)  # type: ignore

        self.noise_scheduler = None

        self.sigma_data = sigma_data
        self.sigma_churn = sigma_churn

        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.pred_last_action_only = pred_last_action_only
        self.action_history = deque(maxlen=self.obs_horizon - 1)

        if "DiffusionGPT" not in self.network_config.get("_target_", None):
            raise ValueError(
                f"BesoPolicy requires a DiffusionGPT network to run. Found: {type(self.network)}"
            )

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
        """BESO Loss formulation using Karras preconditioning and sampling from a continuous
        distribution of noise levels (i.e., do not use Diffusers)."""

        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        B = obs_seq.shape[0]

        # Sample continuous noise levels
        sigmas = self._sample_noise_distribution(B)
        sigma_bd = sigmas.view(B, 1, 1)

        # Add noise
        noise = torch.randn_like(act_seq)
        if self.pred_last_action_only:
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
        if self.pred_last_action_only:
            loss = F.mse_loss(model_output[:, -1:, :], target[:, -1:, :])
        else:
            loss = F.mse_loss(model_output, target)

        return loss

    def _sample_noise_distribution(self, batch_size: int) -> torch.Tensor:
        """Draws training sigmas from a log-logistic distribution (alpha=0.5, beta=0.5).

        As recommended by Reuss et al. (2023) for BESO action diffusion.
        """
        u = torch.rand(batch_size, device=self.device)
        # Log-Logistic inverse CDF: sigma = alpha * (u / (1 - u)) ^ (1/beta)
        return 0.5 * ((u / (1.0 - u)) ** 2)

    def _get_karras_scalings(self, sigma: torch.Tensor):
        """Computes the Karras preconditioning factors."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def _run_diffusion_loop(
        self,
        network_cond: torch.Tensor,
        num_inference_timesteps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        """Runs the BESO diffusion loop to generate the next action.

        Implemented as a custom continuous DDIM scheduler with Karras preconditioning as in Reuss
        et al. (2023).
        """

        # TODO: review with original code

        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        if num_inference_timesteps is None:
            raise ValueError(
                "num_inference_timesteps must be manually provided for BESO inference."
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

        sigmas = self._get_sigmas_exponential(
            num_inference_timesteps, self.sigma_min, self.sigma_max
        )

        with torch.no_grad():
            current_noisy_action = (
                torch.randn((B, 1, self.act_dim), device=self.device) * sigmas[0]
            )

            running_seq = torch.cat([clean_past_actions, current_noisy_action], dim=1)

            # Custom Continuous DDIM Loop
            for i in range(len(sigmas) - 1):
                sigma_t = sigmas[i].expand(B)
                sigma_next = sigmas[i + 1].expand(B)

                # Get Karras scalings for the current sigma
                sigma_bd = sigma_t.view(B, 1, 1)
                c_skip, c_out, c_in = self._get_karras_scalings(sigma_bd)

                # Scale input and predict
                scaled_sample = running_seq * c_in
                model_pred = self.network(sample=scaled_sample, timestep=sigma_t, obs=network_cond)

                if self.pred_last_action_only:
                    current_pred = model_pred[:, -1:, :]
                    current_noisy = running_seq[:, -1:, :]

                    # Reverse the Karras preconditioning to get clean x_0
                    denoised_x0 = current_pred * c_out + current_noisy * c_skip

                    # Continuous DDIM Update Step
                    t = self._t_fn(sigma_t).view(B, 1, 1)
                    t_next = self._t_fn(sigma_next).view(B, 1, 1)
                    h = t_next - t

                    updated_action = (
                        self._sigma_fn(t_next) / self._sigma_fn(t)
                    ) * current_noisy - (-h).expm1() * denoised_x0
                    running_seq[:, -1:, :] = updated_action

                else:
                    # Reverse Karras preconditioning for the full chunk
                    denoised_x0 = model_pred * c_out + running_seq * c_skip

                    # Continuous DDIM Update Step for full chunk
                    t = self._t_fn(sigma_t).view(B, 1, 1)
                    t_next = self._t_fn(sigma_next).view(B, 1, 1)
                    h = t_next - t

                    updated_action = (self._sigma_fn(t_next) / self._sigma_fn(t)) * running_seq - (
                        -h
                    ).expm1() * denoised_x0
                    running_seq = updated_action

        self.ema.restore(self.network.parameters())

        # Extract the actions to execute
        denoised_action = running_seq[:, -1:, :]

        if clamp_range is not None:
            low, high = clamp_range
            denoised_action = torch.clamp(denoised_action, low, high)

        # Save the executed action to the history queue for the next step
        self.action_history.append(denoised_action)

        return denoised_action

    def _get_sigmas_exponential(self, n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(
            math.log(sigma_max), math.log(sigma_min), n, device=self.device
        ).exp()

        # Append 0.0 for the final step
        return torch.cat([sigmas, torch.zeros(1, device=self.device)])

    def _sigma_fn(self, t):
        return t.neg().exp()

    def _t_fn(self, sigma):
        return sigma.log().neg()
