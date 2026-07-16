import math
from collections import deque
from typing import Any

import hydra_zen
import torch
import torch.nn.functional as F

from policy.algorithms import DiffusionPolicy
from policy.utils import concat_leaf_tensors, get_batch_size

# TODO: Since we are not using Diffusers (which directly provided a clipping option)
#       at every denoising step, they added such logic on their own. Note that this is NOT the same as clamping the final action
#       which we already do in the _run_diffusion_loop method. This is a more aggressive clipping that is done at every denoising step.
#       However implementing this could be tricky as I need to understand what range of clipping is right, even in function of which
#       action normalizer we are using.


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
        alpha: float = 0.5,
        beta: float = 0.5,
        sigma_data: float = 0.5,
        sigma_min: float = 0.005,
        sigma_max: float = 1.0,
        pred_last_action_only: bool = False,
        goal_seq_len: int = 0,
        num_parallel_samples: int = 1,
        **kwargs,
    ):

        # NOTE: we implement our own custom DDIM scheduler on continuous sigmas
        # so we can ignore the noise_scheduler argument from the base class

        kwargs.pop("noise_scheduler", None)
        super().__init__(*args, noise_scheduler=None, **kwargs)  # type: ignore

        self.noise_scheduler = None

        # Training
        self.alpha = alpha
        self.beta = beta

        # Inference
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        # Training and Inference
        self.sigma_data = sigma_data

        self.pred_last_action_only = pred_last_action_only
        self.action_history = deque(maxlen=self.obs_horizon - 1)

        self.goal_seq_len = goal_seq_len
        self.goal_conditioned = goal_seq_len > 0
        self.num_parallel_samples = num_parallel_samples

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

    def _prepare_goal(self, goal: dict | torch.Tensor) -> torch.Tensor:
        """Prepares the goal conditioning for the network."""
        return concat_leaf_tensors(goal, device=self.device)

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        action_seq = batch["act_seq"]
        goal = batch.get("goal", None)

        if self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq)
            if goal is not None:
                goal = self.normalizer.normalize(goal)

        if self.action_normalizer is not None:
            action_seq = self.action_normalizer.normalize(action_seq)

        network_cond = self._prepare_network_cond(obs_seq)
        goal_cond = self._prepare_goal(goal) if goal is not None else None

        loss = self._compute_loss(network_cond, action_seq, goal=goal_cond)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _compute_loss(
        self, obs_seq: torch.Tensor, act_seq: torch.Tensor, goal: torch.Tensor | None = None
    ) -> torch.Tensor:
        """BESO Loss formulation using Karras preconditioning and sampling from a continuous
        distribution of noise levels (i.e., do not use Diffusers)."""

        if self.network is None:
            raise ValueError(
                "Network not initialized. Call configure_model() before getting action."
            )

        B = obs_seq.shape[0]

        # Sample continuous noise levels
        sigmas = self._sample_noise_distribution(B, alpha=self.alpha, beta=self.beta)
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
        model_output = self.network(
            sample=scaled_noisy_act, timestep=sigmas, obs=obs_seq, goal=goal
        )

        # Compute the Karras target (reversing the skip connection)
        target = (act_seq - c_skip * noisy_act_seq) / c_out

        # Compute MSE loss
        if self.pred_last_action_only:
            loss = F.mse_loss(model_output[:, -1:, :], target[:, -1:, :])
        else:
            loss = F.mse_loss(model_output, target)

        return loss

    def get_action(
        self,
        obs_seq: torch.Tensor | dict,
        goal: torch.Tensor | dict | None = None,
        num_inference_timesteps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        if self.normalizer is not None:
            obs_seq = self.normalizer.normalize(obs_seq)
            if goal is not None:
                goal = self.normalizer.normalize(goal)

        obs_seq = self._prepare_network_cond(obs_seq)
        goal_cond = self._prepare_goal(goal) if goal is not None else None

        return self._run_diffusion_loop(obs_seq, goal_cond, num_inference_timesteps, clamp_range)

    def _run_diffusion_loop(
        self,
        network_cond: torch.Tensor,
        goal_cond: torch.Tensor | None = None,
        num_inference_timesteps: int | None = None,
        clamp_range: tuple | None = None,
    ):
        """Runs the BESO diffusion loop to generate the next action.

        Implemented as a custom continuous DDIM scheduler with Karras preconditioning as in Reuss
        et al. (2023).
        """

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

        # Determine the current history length
        H = len(self.action_history)
        cur_obs_len = H + 1

        # Combine the deque into a single [B, H, act_dim] tensor
        if H > 0:
            clean_past_actions = torch.cat(list(self.action_history), dim=1)
        else:
            clean_past_actions = torch.zeros((B, 0, self.act_dim), device=self.device)

        # Slice network_cond to the current observation sequence length
        if network_cond.ndim == 3:
            sliced_network_cond = network_cond[:, -cur_obs_len:, :]
        else:
            sliced_network_cond = network_cond.view(B, self.obs_horizon, -1)[:, -cur_obs_len:, :]

        if self.num_parallel_samples > 1:
            clean_past_actions = clean_past_actions.repeat_interleave(
                self.num_parallel_samples, dim=0
            )
            sliced_network_cond = sliced_network_cond.repeat_interleave(
                self.num_parallel_samples, dim=0
            )
            if goal_cond is not None:
                goal_cond = goal_cond.repeat_interleave(self.num_parallel_samples, dim=0)
            B_expanded = B * self.num_parallel_samples
        else:
            B_expanded = B

        self.ema.store(self.network.parameters())
        self.ema.copy_to(self.network.parameters())

        sigmas = self._get_sigmas_exponential(
            num_inference_timesteps, self.sigma_min, self.sigma_max
        )

        # NOTE: they provided many custom solvers in the original BESO code
        # however in the paper they indicated DDIM as their best choice for
        # (goal-conditioned) action diffusion, so we only implemented that one here.
        # Tho, one could add other solvers as well, e.g., DPM++, Heun, LMS, Ancestral, etc.
        # see https://github.com/intuitive-robots/beso/blob/main/beso/agents/diffusion_agents/beso_agent.py#L437

        with torch.no_grad():
            current_noisy_action = (
                torch.randn((B_expanded, 1, self.act_dim), device=self.device) * sigmas[0]
            )

            running_seq = torch.cat([clean_past_actions, current_noisy_action], dim=1)

            # Custom Continuous DDIM Loop
            for i in range(len(sigmas) - 1):
                sigma_t = sigmas[i].expand(B_expanded)
                sigma_next = sigmas[i + 1].expand(B_expanded)

                # Get Karras scalings for the current sigma
                sigma_bd = sigma_t.view(B_expanded, 1, 1)
                c_skip, c_out, c_in = self._get_karras_scalings(sigma_bd)

                # Scale input and predict
                scaled_sample = running_seq * c_in
                model_pred = self.network(
                    sample=scaled_sample, timestep=sigma_t, obs=sliced_network_cond, goal=goal_cond
                )

                if self.pred_last_action_only:
                    current_pred = model_pred[:, -1:, :]
                    current_noisy = running_seq[:, -1:, :]
                else:
                    current_pred = model_pred
                    current_noisy = running_seq

                # Reverse Karras preconditioning
                scaled_pred = current_pred * c_out + current_noisy * c_skip

                # Continuous-DDIM step
                # as in https://github.com/intuitive-robots/beso/blob/main/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L896
                t = self._t_fn(sigma_t).view(B_expanded, 1, 1)
                t_next = self._t_fn(sigma_next).view(B_expanded, 1, 1)

                h = t_next - t
                sigma_ratio = self._sigma_fn(t_next) / self._sigma_fn(t)

                updated_action = sigma_ratio * current_noisy - (-h).expm1() * scaled_pred

                # Update the running sequence with the new action
                if self.pred_last_action_only:
                    running_seq[:, -1:, :] = updated_action
                else:
                    running_seq = updated_action

        self.ema.restore(self.network.parameters())

        # Extract the actions to execute
        denoised_action = running_seq[:, -1:, :]

        if self.num_parallel_samples > 1:
            denoised_action = denoised_action.view(B, self.num_parallel_samples, 1, self.act_dim)
            denoised_action = denoised_action.mean(dim=1)

        if self.action_normalizer is not None:
            physical_action = self.action_normalizer.unnormalize(denoised_action)
            if clamp_range is not None:
                low, high = clamp_range
                physical_action = torch.clamp(physical_action, low, high)
            executed_action_normalized = self.action_normalizer.normalize(physical_action)
            self.action_history.append(executed_action_normalized)
            denoised_action = physical_action
        else:
            if clamp_range is not None:
                low, high = clamp_range
                denoised_action = torch.clamp(denoised_action, low, high)
            self.action_history.append(denoised_action)

        return denoised_action

    def _get_karras_scalings(self, sigma: torch.Tensor):
        """Computes the Karras preconditioning factors."""
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def _sample_noise_distribution(
        self, batch_size: int, alpha: float = 0.5, beta: float = 0.5
    ) -> torch.Tensor:
        """Draws training sigmas from a log-logistic distribution (alpha=0.5, beta=0.5).

        As recommended by Reuss et al. (2023) for BESO action diffusion.
        """
        u = torch.rand(batch_size, device=self.device)
        # Log-Logistic inverse CDF: sigma = alpha * (u / (1 - u)) ^ (1/beta)
        return torch.clamp(alpha * ((u / (1.0 - u)) ** (1 / beta)), min=1e-5, max=1e5)

    def _get_sigmas_exponential(self, n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
        """Constructs an exponential noise schedule."""
        sigmas = torch.linspace(
            math.log(sigma_max), math.log(sigma_min), n, device=self.device
        ).exp()

        # Append 0.0 for the final step
        return torch.cat([sigmas, torch.zeros(1, device=self.device)])

    def _t_fn(self, sigma):
        return sigma.log().neg()

    def _sigma_fn(self, t):
        return t.neg().exp()
