import math
from collections import deque
from collections.abc import Mapping
from typing import Any, cast

import hydra_zen
import torch
import torch.nn.functional as F

from policy.algorithms.base_diffusion_agent import BaseDiffusionAgent
from policy.algorithms.networks.utils import as_task_only
from policy.utils import (
    concat_leaf_tensors,
    get_batch_size,
    get_total_dim,
    map_leaves,
    merge_dicts,
    pack_obs_goal,
    pop_leaf_key,
    subtract_leaves,
)
from policy.utils.typing_utils import (
    DimSpec,
    GoalConditionedPolicyProtocol,
    TensorTree,
)


class BesoPolicy(BaseDiffusionAgent, GoalConditionedPolicyProtocol):
    """Trains a BESO diffusion model to predict action sequences from observation histories.

    BESO as in Goal-Conditioned Imitation Learning using Score-based Diffusion Policies, Reuss et al. (RSS, 2023)

    Reference:
        - Arxiv: https://arxiv.org/abs/2304.02532
        - Paper website: https://intuitive-robots.github.io/beso-website/

    NOTE: Running BESO with `as_dict=true` vs `as_dict=false` (when no keys are dropped) processes
        identical data batches, normalized features, and network weights. However, minor
        initial validation loss variations occur because BESO's continuous noise sampling
        (sigmas and noise vectors) is drawn after setup-phase PRNG ticks from ModuleDict initialization.
    """

    def __init__(
        self,
        *args,
        alpha: float = 0.5,
        beta: float = 0.5,
        sigma_min: float = 0.005,
        sigma_max: float = 1.0,
        sigma_data: float = 0.5,
        pred_last_action_only: bool = False,
        num_parallel_samples: int = 1,
        goal_drop_prob: float = 0.1,
        cfg_lambda: float | None = None,
        use_proprio_token: bool = False,
        **kwargs,
    ):

        # TODO: looking at this code, it seems like BESO is actually using some form of
        # (degenerate, checks and reshaping only) encoder. Value later if we could make one for it too.

        super().__init__(*args, **kwargs)

        if "DiffusionGPT" not in self.decoder_config.get("_target_", None):
            raise ValueError(
                f"BesoPolicy requires a DiffusionGPT decoder to run. Found: {type(self.decoder)}"
            )

        if self.noise_scheduler_config is not None:
            raise ValueError(
                "BesoPolicy does not support noise_schedulers as it implements its own custom one. "
                f"Got noise_scheduler={self.noise_scheduler_config}. Please remove it."
            )

        # For BESO++ (relative_goal is a BaseDiffusionAgent-level option)
        self.use_proprio_token = use_proprio_token

        # Training
        self.alpha = alpha
        self.beta = beta
        self.goal_drop_prob = goal_drop_prob

        # Inference
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.cfg_lambda = cfg_lambda

        # Training and Inference
        self.sigma_data = sigma_data

        # Validation and extra options
        self._validate_goal_parameters()

        self.pred_last_action_only = pred_last_action_only
        self.action_history = deque(maxlen=self.obs_horizon - 1)

        self.num_parallel_samples = num_parallel_samples

    def _validate_goal_parameters(self):
        if self.relative_goal and not self.goal_conditioned:
            raise ValueError(
                "relative_goal=True requires goal_horizon > 0: there is no goal to "
                "difference the observations against when goal-conditioning is disabled."
            )
        if self.relative_goal:
            if self.goal_horizon != 1:
                raise ValueError(
                    f"relative_goal=True requires goal_horizon=1 (got goal_horizon={self.goal_horizon}): "
                    "a single goal frame is differenced against every obs timestep; multi-frame "
                    "goals would need an explicit broadcasting rule that isn't defined here."
                )
            if self.goal_drop_prob > 0.0 or self.cfg_lambda is not None:
                raise ValueError(
                    f"relative_goal=True is mutually exclusive with classifier-free guidance "
                    f"(goal_drop_prob={self.goal_drop_prob}, cfg_lambda={self.cfg_lambda}): CFG's "
                    "'unconditional' goal-zeroing has no well-defined meaning once there's no "
                    "standalone goal tensor to zero. Leave goal_drop_prob=0.0 and cfg_lambda=None."
                )

    def configure_optimizers(self):
        """BESO custom optimizer configuration with weight decay handling for DiffusionGPT."""
        if self.decoder is None:
            raise ValueError(
                "Decoder not initialized. Call configure_model() before configure_optimizers."
            )

        # Separate parameters into decay/no_decay sets
        decay = set()
        no_decay = set()
        # nn.MultiheadAttention owns its packed in-projection (in_proj_weight/in_proj_bias)
        # directly -- it isn't an nn.Linear -- so it needs its own whitelist entry; its
        # out_proj sub-module is an nn.Linear and is already covered by that branch.
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        # Iterate over the DiffusionGPT decoder
        for mn, m in self.decoder.named_modules():
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
        if hasattr(self.decoder, "pos_emb"):
            no_decay.add("pos_emb")

        # Validate that we categorized every parameter
        param_dict = {pn: p for pn, p in self.decoder.named_parameters() if p.requires_grad}
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

    def _get_cond_dims(self) -> DimSpec:
        """Reports the per-timestep conditioning dimensionality passed to the network."""
        cond_dims = cast(Mapping[str, DimSpec], super()._get_cond_dims())

        # No "goal" key when relative_goal=True, the goal is folded into obs
        if self.goal_horizon > 0 and not self.relative_goal:
            cond_dims = {**cond_dims, "goal": get_total_dim(cond_dims["obs"])}
        return cond_dims

    def _decoder_extra_kwargs(self) -> dict[str, Any]:
        kwargs = super()._decoder_extra_kwargs()  # {"cond_dims": self._get_cond_dims()}
        if self.use_proprio_token:
            kwargs["proprio_dim"] = self.proprio_dim
            kwargs["use_proprio_token"] = True
        if self.relative_goal:
            # No "goal" key when relative_goal=True, the goal is folded into obs
            kwargs["goal_horizon"] = 0
        return kwargs

    def get_action(
        self,
        obs_seq: torch.Tensor | Mapping[str, Any],
        goal: torch.Tensor | Mapping[str, Any] | None = None,
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        external_cond = self._build_external_cond(obs_seq, goal)

        return self._run_diffusion_loop(
            external_cond=external_cond,
            num_inference_steps=num_inference_steps,
            output_clip_range=output_clip_range,
        )

    def _shared_step(self, batch: dict[str, Any], batch_idx: int, phase: str) -> torch.Tensor:
        obs_seq = batch["obs_seq"]
        action_seq = batch["act_seq"]
        goal = batch.get("goal", None)

        action_seq = self._normalize_act(action_seq)

        external_cond = self._build_external_cond(obs_seq, goal)

        loss = self._compute_loss(external_cond, action_seq)

        self.log(f"{phase}/loss", loss, prog_bar=True, sync_dist=(phase == "val"))
        return loss

    def _compute_loss(
        self, external_cond: Mapping[str, TensorTree], act_seq: torch.Tensor
    ) -> torch.Tensor:
        """BESO Loss formulation using Karras preconditioning and sampling from a continuous
        distribution of noise levels (i.e., do not use Diffusers)."""

        if self.decoder is None:
            raise ValueError(
                "Decoder not initialized. Call configure_model() before getting action."
            )

        B = get_batch_size(external_cond)

        # Since we need to mask the goal for CFG
        # we unpack the external_cond and pack it back later

        # Unpack data
        obs_seq = external_cond["obs"]
        goal = external_cond.get("goal", None)

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

        # Predict raw model output (decoder is just DiffusionGPT)
        scaled_noisy_act = noisy_act_seq * c_in

        # Goal dropout for Classifier-Free Guidance
        if self.training and goal is not None and self.goal_drop_prob > 0.0:
            # One drop decision per batch item, shared across every leaf of the goal tree.
            drop_mask = torch.rand(B, device=self.device) < self.goal_drop_prob
            goal = map_leaves(
                lambda leaf: torch.where(
                    drop_mask.view([B] + [1] * (leaf.ndim - 1)), torch.zeros_like(leaf), leaf
                ),
                goal,
            )

        # Repack data back into the external_cond dictionary and predict.
        external_cond = pack_obs_goal(obs_seq, goal)
        model_output = self.decoder(
            sample=scaled_noisy_act, timestep=sigmas, external_cond=external_cond
        )

        # Compute the Karras target (reversing the skip connection)
        target = (act_seq - c_skip * noisy_act_seq) / c_out

        # Compute MSE loss
        if self.pred_last_action_only:
            loss = F.mse_loss(model_output[:, -1:, :], target[:, -1:, :])
        else:
            loss = F.mse_loss(model_output, target)

        return loss

    def _build_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        if self.goal_conditioned and goal is None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} > 0, "
                "but received goal=None."
            )
        if not self.goal_conditioned and goal is not None:
            raise ValueError(
                f"{type(self).__name__} is configured with goal_horizon={self.goal_horizon} "
                "(not goal-conditioned), but received a non-None goal. Pass goal=None, or "
                "construct this policy with goal_horizon > 0."
            )

        obs = self._normalize_obs(obs)
        if goal is not None:
            goal = self._normalize_obs(goal)

        if self.relative_goal:
            assert goal is not None
            return self._build_delta_external_cond(obs, goal)
        else:
            return self._build_abs_external_cond(obs, goal)

    def _build_abs_external_cond(
        self, obs: TensorTree, goal: TensorTree | None
    ) -> dict[str, TensorTree]:
        """Build the external condition dictionary for the absolute goal variant (original
        BESO)."""
        obs_cond = self._build_obs_external_cond(obs)
        goal_cond = self._build_goal_external_cond(goal) if goal is not None else {}

        return merge_dicts([obs_cond, goal_cond])

    def _build_obs_external_cond(self, obs: TensorTree) -> dict[str, TensorTree]:
        return {"obs": obs}

    def _build_goal_external_cond(self, goal: TensorTree) -> dict[str, TensorTree]:
        return {"goal": goal}

    def _build_delta_external_cond(
        self, obs: TensorTree, goal: TensorTree
    ) -> dict[str, TensorTree]:
        """Builds the external condition dictionary for the goal-delta variant."""
        obs_proprio, obs_task = pop_leaf_key(obs, "proprio", self.proprio_dim)
        if obs_proprio is None:
            raise ValueError(
                f"{type(self).__name__} requires external_cond['obs'] to carry a 'proprio' key."
            )

        goal_task = self._extract_goal_task(goal)
        task_delta = subtract_leaves(goal_task, obs_task)

        if isinstance(task_delta, Mapping):
            return {"obs": {"proprio": obs_proprio, **task_delta}}

        return {"obs": {"proprio": obs_proprio, "task": task_delta}}

    def _extract_goal_task(self, goal: TensorTree) -> TensorTree:
        """Extracts the task-only portion of `goal`."""
        if isinstance(goal, Mapping):
            _, goal_task = pop_leaf_key(goal, "proprio", self.proprio_dim)
            return goal_task
        elif isinstance(goal, torch.Tensor):
            return as_task_only(goal, self.proprio_dim, self.task_dim)

    def _run_diffusion_loop(
        self,
        external_cond: Mapping[str, TensorTree],
        num_inference_steps: int | None = None,
        output_clip_range: tuple | None = None,
    ):
        """Runs the BESO diffusion loop to generate the next action.

        Implemented as a custom continuous DDIM scheduler with Karras preconditioning as in Reuss
        et al. (2023).
        """

        if self.decoder is None:
            raise ValueError(
                "Decoder not initialized. Call configure_model() before getting action."
            )

        if num_inference_steps is None:
            raise ValueError("num_inference_steps must be manually provided for BESO inference.")

        if self.ema is None:
            raise ValueError(
                "EMA Model not initialized. Call configure_model() before getting action."
            )

        B = get_batch_size(external_cond)

        # Unpack data
        obs_cond = external_cond["obs"]
        goal_cond = external_cond.get("goal", None)

        # Determine the current history length
        H = len(self.action_history)
        cur_obs_len = H + 1

        # Combine the deque into a single [B, H, act_dim] tensor
        if H > 0:
            clean_past_actions = torch.cat(list(self.action_history), dim=1)
        else:
            clean_past_actions = torch.zeros((B, 0, self.act_dim), device=self.device)

        # NOTE: Slicing logic below needs a flat Tensor so we need
        # to flatten it before and work on it; kind of dirty but same
        # flattening would have happened inside DiffusionGPT anyway.

        if isinstance(obs_cond, Mapping):
            obs_cond = concat_leaf_tensors(obs_cond, dim=-1)

        # Slice network_cond to the current observation sequence length
        if obs_cond.ndim == 3:
            sliced_obs_cond = obs_cond[:, -cur_obs_len:, :]
        else:
            sliced_obs_cond = obs_cond.view(B, self.obs_horizon, -1)[:, -cur_obs_len:, :]

        if self.num_parallel_samples > 1:
            clean_past_actions = clean_past_actions.repeat_interleave(
                self.num_parallel_samples, dim=0
            )
            sliced_obs_cond = sliced_obs_cond.repeat_interleave(self.num_parallel_samples, dim=0)
            if goal_cond is not None:
                goal_cond = map_leaves(
                    lambda t: t.repeat_interleave(self.num_parallel_samples, dim=0), goal_cond
                )
            B_expanded = B * self.num_parallel_samples
        else:
            B_expanded = B

        self.ema.store(self._ema_parameters())
        self.ema.copy_to(self._ema_parameters())

        sigmas = self._get_sigmas_exponential(num_inference_steps, self.sigma_min, self.sigma_max)

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

                # Repack and send it to the decoder
                cond_pred = self.decoder(
                    sample=scaled_sample,
                    timestep=sigma_t,
                    external_cond=pack_obs_goal(sliced_obs_cond, goal_cond),
                )

                # Unconditional Prediction (goal zeroed out)
                # cfg_lambda=None (default): CFG disabled, single network call, model_pred=cond_pred.
                #   Mathematically identical to lambda=1 below, just without paying for the extra call.
                # cfg_lambda=0.0: two network calls, blend collapses to uncond_pred -- the policy
                #   ignores the goal entirely (paper's Fig. 5 diagnostic for a well-formed unconditional
                #   policy). NOT the same as leaving cfg_lambda unset.
                # cfg_lambda=0.5: blends toward uncond_pred, i.e. weaker goal influence than plain
                #   conditioning (paper's tuned value for Block-Push, action_dim=2).
                # cfg_lambda=1.0: blend collapses to cond_pred -- same output as disabling CFG, but
                #   costs the redundant unconditional forward pass. Useful only to sanity-check the
                #   two branches agree.
                # cfg_lambda=1.25: extrapolates past cond_pred, amplifying goal influence beyond plain
                #   conditioning (paper's tuned value for Kitchen/CALVIN, action_dim=9).
                # cfg_lambda>1 in general: stronger extrapolation/guidance; the paper reports
                #   instability at higher values in higher-dimensional action spaces.
                if self.cfg_lambda is not None and goal_cond is not None:
                    uncond_goal = map_leaves(torch.zeros_like, goal_cond)
                    uncond_pred = self.decoder(
                        sample=scaled_sample,
                        timestep=sigma_t,
                        external_cond=pack_obs_goal(sliced_obs_cond, uncond_goal),
                    )

                    model_pred = uncond_pred + self.cfg_lambda * (cond_pred - uncond_pred)
                else:
                    model_pred = cond_pred

                if self.pred_last_action_only:
                    current_pred = model_pred[:, -1:, :]
                    current_noisy = running_seq[:, -1:, :]
                else:
                    current_pred = model_pred
                    current_noisy = running_seq

                # Reverse Karras preconditioning
                scaled_pred = current_pred * c_out + current_noisy * c_skip

                # Continuous-DDIM step, as in:
                # https://github.com/intuitive-robots/beso/blob/main/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L896
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

        self.ema.restore(self._ema_parameters())

        # Extract the actions to execute
        denoised_action = running_seq[:, -1:, :]

        if self.num_parallel_samples > 1:
            denoised_action = denoised_action.view(B, self.num_parallel_samples, 1, self.act_dim)
            denoised_action = denoised_action.mean(dim=1)

        # NOTE: The original BESO implementation incorrectly inverts the order of normalization and clipping:
        # it first clips the denoised actions in the normalized space (to [-1, 1]), and only then unnormalizes
        # them to the physical space. This is mathematically sloppy:
        # 1. With MinMaxNormalizer, it assumes the training dataset's min/max bounds perfectly align with the
        #    physical environment boundaries.
        # 2. With ZScoreNormalizer, it restricts the generated actions to within exactly one standard deviation
        #    from the mean, preventing the policy from ever outputting extreme actions.
        # By unnormalizing first and then clipping in physical space using output_clip_range, we apply
        # the bounds correctly in physical/environment coordinates, regardless of the normalization.

        if self.act_normalizer is not None:
            physical_action = self.act_normalizer.unnormalize(denoised_action)
            if output_clip_range is not None:
                low, high = output_clip_range
                physical_action = torch.clamp(physical_action, low, high)
            executed_action_normalized = self.act_normalizer.normalize(physical_action)
            self.action_history.append(executed_action_normalized)
            denoised_action = physical_action
        else:
            if output_clip_range is not None:
                low, high = output_clip_range
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
