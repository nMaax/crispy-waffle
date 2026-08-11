"""Cross-attention-conditioned 1D UNet -- Stable-Diffusion-style conditioning for the diffusion
policy noise-prediction network.

Reuses ``ConditionalUnet1D``'s building blocks unmodified (residual FiLM conditioning stays the
mechanism for proprioception and the diffusion timestep) and adds one new block,
``CrossAttentionBlock1D``, after each pair of residual blocks -- the analogue of Stable Diffusion's
``BasicTransformerBlock`` (its cross-attention + feed-forward sublayers; the self-attention sublayer
is omitted since the surrounding convolutions already do local temporal mixing). The UNet attends
over a token sequence carried as ``external_cond["context"]`` instead of having it flattened into
the FiLM vector -- see
:class:`policy.algorithms.cross_attention_goal_conditioned_diffusion_policy.CrossAttentionGoalConditionedDiffusionPolicy`
for how that sequence is produced.
"""

from collections.abc import Mapping

import torch
import torch.nn as nn

from policy.algorithms.networks.unet1d import (
    ConditionalResidualBlock1D,
    Conv1dBlock,
    Downsample1d,
    SinusoidalPosEmb,
    Upsample1d,
)
from policy.utils import flatten_and_concat_leaf_tensors, get_total_dim
from policy.utils.typing_utils import DimSpec, TensorTree
from policy.utils.typing_utils.protocols import DiffusionNetworkProtocol


class CrossAttentionBlock1D(nn.Module):
    """Cross-attends a UNet feature map over a conditioning token sequence.

    The analogue of Stable Diffusion's ``BasicTransformerBlock.attn2`` + ``.ff``: pre-norm ->
    cross-attention (query=features, key=value=projected context) -> residual -> pre-norm ->
    position-wise feed-forward -> residual.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        n_groups: int = 8,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(n_groups, channels)
        self.context_proj = nn.Linear(context_dim, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.GroupNorm(n_groups, channels)
        self.ff = nn.Sequential(
            nn.Linear(channels, 4 * channels),
            nn.Mish(),
            nn.Linear(4 * channels, channels),
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            x: [B, channels, horizon]
            context: [B, S, context_dim]
            returns: [B, channels, horizon]
        """
        h = self.norm1(x).moveaxis(-1, -2)  # [B, horizon, channels]
        ctx = self.context_proj(context)  # [B, S, channels]
        attn_out, _ = self.attn(h, ctx, ctx, need_weights=False)
        x = x + attn_out.moveaxis(-1, -2)

        h = self.norm2(x).moveaxis(-1, -2)  # [B, horizon, channels]
        x = x + self.ff(h).moveaxis(-1, -2)
        return x


class CrossAttentionConditionalUnet1D(nn.Module, DiffusionNetworkProtocol):
    def __init__(
        self,
        act_dim,
        cond_dims: DimSpec,
        obs_horizon: int,
        diffusion_step_embed_dim=256,
        down_dims=[256, 512, 1024],
        kernel_size=5,
        n_groups=8,
        cross_attn_num_heads=4,
        cross_attn_dropout=0.0,
    ):
        """UNet with FiLM conditioning (proprioception + diffusion timestep, exactly as in
        ``ConditionalUnet1D``) plus Stable-Diffusion-style cross-attention conditioning on a token
        sequence, carried in ``external_cond["context"]``.

        Operates: Downsample (residual blocks + cross-attention) --> Middle (residual blocks +
        cross-attention) --> Upsample (residual blocks + cross-attention).
        """
        super().__init__()
        if not isinstance(cond_dims, Mapping) or "context" not in cond_dims:
            raise ValueError(
                "CrossAttentionConditionalUnet1D requires cond_dims to be a mapping with a "
                f"'context' entry (the cross-attention token width); got {cond_dims!r}."
            )
        cond_dims_map = dict(cond_dims)
        context_dim = get_total_dim(cond_dims_map.pop("context"))

        all_dims = [act_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        # The remaining cond_dims (everything but "context", handled above via cross-attention)
        # report per-timestep widths (see BaseDiffusionAgent._get_cond_dims); only "obs" repeats
        # every timestep, so only its width gets multiplied by obs_horizon -- mirrors
        # ConditionalUnet1D exactly.
        total_cond_dim = sum(
            get_total_dim(spec) * (obs_horizon if key == "obs" else 1)
            for key, spec in cond_dims_map.items()
        )
        obs_dim = dsed + total_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        def make_cross_attn(channels: int) -> CrossAttentionBlock1D:
            return CrossAttentionBlock1D(
                channels,
                context_dim,
                num_heads=cross_attn_num_heads,
                dropout=cross_attn_dropout,
                n_groups=n_groups,
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    obs_dim=obs_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    obs_dim=obs_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                ),
            ]
        )
        self.mid_cross_attn = make_cross_attn(mid_dim)

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            obs_dim=obs_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            obs_dim=obs_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        make_cross_attn(dim_out),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            obs_dim=obs_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            obs_dim=obs_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        make_cross_attn(dim_in),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, act_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        external_cond: Mapping[str, TensorTree] | None = None,
    ) -> torch.Tensor:
        """Predicts the noise residual for a given noisy sample, timestep, and conditioning.

        Shapes:
            sample: [B, pred_horizon, input_dim] (noisy action sequence)
            timestep: [B,] or int
            external_cond: conditioning tensor tree, e.g. ``{"obs": {"proprio": ...}, "context":
                ...}``. The ``"context"`` entry (``[B, S, context_dim]``) is cross-attended over;
                everything else is flattened and concatenated for FiLM, exactly as in
                ``ConditionalUnet1D``.
            returns: [B, horizon, input_dim] (predicted noise)
        """
        if external_cond is None or "context" not in external_cond:
            raise ValueError(
                "CrossAttentionConditionalUnet1D requires external_cond to contain a 'context' "
                "entry (the cross-attention token sequence)."
            )
        external_cond = dict(external_cond)
        context = external_cond.pop("context")
        if not isinstance(context, torch.Tensor):
            raise TypeError(f"Expected external_cond['context'] to be a Tensor, got {type(context)}.")

        # (B,T,C) -> (B,C,T)
        sample = sample.moveaxis(-1, -2)

        # Ensure time is a non-scalar tensor
        if not isinstance(timestep, torch.Tensor):
            # this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        else:
            timesteps = timestep
            if len(timesteps.shape) == 0:
                timesteps = timesteps[None]

        # Broadcast time to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        # Encode time as an embedding
        global_feature = self.diffusion_step_encoder(timesteps)

        # Concatenate time embedding with the remaining (FiLM) global conditioning, if any
        if external_cond:
            cond_flat = flatten_and_concat_leaf_tensors(external_cond, device=sample.device)
            global_feature = torch.cat([global_feature, cond_flat], dim=-1)

        # Prepare variables to pass and track Unet features for skip connections
        x = sample  # Working variable that we will pass through the UNet
        h = []  # Storage for features for skip connections between down and up modules

        # Downsample
        for modules in self.down_modules:
            resnet, resnet2, cross_attn, downsample = modules  # type: ignore[misc]
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = cross_attn(x, context)
            h.append(x)
            x = downsample(x)

        # Middle
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        x = self.mid_cross_attn(x, context)

        # Upsample
        for modules in self.up_modules:
            resnet, resnet2, cross_attn, upsample = modules  # type: ignore[misc]
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = cross_attn(x, context)
            x = upsample(x)

        # Final layer
        x = self.final_conv(x)

        # (B,C,T) -> (B,T,C)
        x = x.moveaxis(-1, -2)

        return x
