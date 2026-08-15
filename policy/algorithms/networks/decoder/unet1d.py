# Reference: https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/diffusion_policy/diffusion_policy/conditional_unet1d.py

import math
from collections.abc import Mapping, Sequence

import torch
import torch.nn as nn

from policy.algorithms.networks.encoder import ConditioningContract
from policy.utils import flatten_and_concat_leaf_tensors
from policy.utils.typing_utils import TensorTree
from policy.utils.typing_utils.protocols import DiffusionNetworkProtocol


class SinusoidalPosEmb(nn.Module):
    """Positional embedding for diffusion (time) step k.

    Similar to the one used in the original transformer paper.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Embeds the timestep tensor.

        shapes:
            x: [B,] (timesteps)
            returns: [B, dim]
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)  # type: ignore
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # type: ignore
        return emb


class Downsample1d(nn.Module):
    """Downsamples the input by a factor of 2 with a strided convolution."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        """
        shapes:
            x: [B, in_channels, horizon]
            returns: [B, in_channels, horizon // 2]
        """
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsamples the input by a factor of 2 with a transposed convolution."""

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        """
        shapes:
            x: [B, in_channels, horizon]
            returns: [B, in_channels, horizon * 2]
        """
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """A basic block for convolutional processing in the UNet.

    Operates: Conv1d --> GroupNorm --> Mish.
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        """
        shapes:
            x: [B, in_channels, horizon]
            returns: [B, out_channels, horizon]
        """
        return self.block(x)


class FiLMResidualBlock1D(nn.Module):
    """A wrapper of the basic block that applies FiLM conditioning and a residual connection.

    Operates: Conv1dBlock --> FiLM conditioning --> Conv1dBlock with residual connection.
    """

    def __init__(self, in_channels, out_channels, obs_dim, kernel_size=3, n_groups=8):
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        obs_channels = out_channels * 2
        self.out_channels = out_channels
        self.obs_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(obs_dim, obs_channels), nn.Unflatten(-1, (-1, 1))
        )

        # Make sure dimensions are compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, cond):
        """
        shapes:
            x : [ B, in_channels, horizon ]
            cond : [ B, extended_obs_dim (obs_dim + time_embed_dim)]
            returns: [ B, out_channels, horizon ]
        """
        # First Conv1d Block
        out = self.blocks[0](x)

        # FiLM conditioning
        embed = self.obs_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:, 0, ...]
        bias = embed[:, 1, ...]
        out = scale * out + bias

        # Second Conv1d Block + residual
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class CrossAttentionBlock1D(nn.Module):
    """Cross-attends a UNet feature map over a conditioning token sequence, like Stable- Diffusion.

    Operates: GroupNorm --> MultiheadAttention over context + residual connection -->
    GroupNorm --> FeedForward with residual connection.
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
        shapes:
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


class FiLMDecoder1D(nn.Module, DiffusionNetworkProtocol):
    """1D UNet decoder with FiLM conditioning for noise prediction.

    Operates: Downsample residual blocks --> Middle residual blocks --> Upsample residual
    blocks with skip connections, using FiLM (Feature-wise Linear Modulation) conditioning.

    Takes ``cond_dims`` directly (computed by a :class:`~policy.algorithms.networks.encoder.
    ConditioningEncoder` the owning algorithm builds separately) rather than owning any
    conditioning logic itself. Its ``external_cond`` is the encoder's already-encoded payload,
    never a raw obs/goal tree -- the name is kept for consistency with `DiffusionNetworkProtocol`
    and `DiffusionGPT`'s calling convention, even though nothing "external" happens here anymore.
    """

    def __init__(
        self,
        act_dim: int,
        cond_dims: ConditioningContract,
        obs_horizon: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
    ):
        super().__init__()

        all_dims = [act_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        film_dim = dsed + cond_dims.get_film_width(obs_horizon)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            downsample = Downsample1d(dim_out) if not is_last else nn.Identity()
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock1D(
                            dim_in,
                            dim_out,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        FiLMResidualBlock1D(
                            dim_out,
                            dim_out,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        downsample,
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                FiLMResidualBlock1D(
                    mid_dim, mid_dim, obs_dim=film_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                FiLMResidualBlock1D(
                    mid_dim, mid_dim, obs_dim=film_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            upsample = Upsample1d(dim_in) if not is_last else nn.Identity()
            self.up_modules.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        FiLMResidualBlock1D(
                            dim_in,
                            dim_in,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        upsample,
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, act_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        external_cond: Mapping[str, TensorTree],
    ) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)

        if not isinstance(timestep, torch.Tensor):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        else:
            timesteps = timestep
            if len(timesteps.shape) == 0:
                timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        cond_flat = flatten_and_concat_leaf_tensors(external_cond, device=sample.device)
        global_feature = torch.cat([global_feature, cond_flat], dim=-1)

        x = sample
        h = []

        for stage in self.down_modules:
            resnet1, resnet2, downsample = stage  # type: ignore[misc]
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for stage in self.up_modules:
            resnet1, resnet2, upsample = stage  # type: ignore[misc]
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)


class CrossAttentionDecoder1D(nn.Module, DiffusionNetworkProtocol):
    """1D UNet decoder with cross-attention over tokens.

    Operates: Downsample residual blocks --> Middle residual blocks --> Upsample residual
    blocks with skip connections, using cross-attention conditioning.
    """

    def __init__(
        self,
        act_dim: int,
        cond_dims: ConditioningContract,
        obs_horizon: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cross_attn_num_heads: int = 4,
        cross_attn_dropout: float = 0.0,
    ):
        super().__init__()

        all_dims = [act_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )

        if cond_dims.context_dim is None:
            raise ValueError(
                "CrossAttentionDecoder1D requires context_dim to be set in ConditioningContract."
            )
        self.context_key = cond_dims.context_key
        context_dim = cond_dims.context_dim
        film_dim = dsed + cond_dims.get_film_width(obs_horizon)

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

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            downsample = Downsample1d(dim_out) if not is_last else nn.Identity()
            self.down_modules.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock1D(
                            dim_in,
                            dim_out,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        FiLMResidualBlock1D(
                            dim_out,
                            dim_out,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        make_cross_attn(dim_out),
                        downsample,
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                FiLMResidualBlock1D(
                    mid_dim, mid_dim, obs_dim=film_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
                FiLMResidualBlock1D(
                    mid_dim, mid_dim, obs_dim=film_dim, kernel_size=kernel_size, n_groups=n_groups
                ),
            ]
        )
        self.mid_cross_attn = make_cross_attn(mid_dim)

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            upsample = Upsample1d(dim_in) if not is_last else nn.Identity()
            self.up_modules.append(
                nn.ModuleList(
                    [
                        FiLMResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        FiLMResidualBlock1D(
                            dim_in,
                            dim_in,
                            obs_dim=film_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        make_cross_attn(dim_in),
                        upsample,
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, act_dim, 1),
        )

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        external_cond: Mapping[str, TensorTree],
    ) -> torch.Tensor:
        sample = sample.moveaxis(-1, -2)

        if not isinstance(timestep, torch.Tensor):
            timesteps = torch.tensor([timestep], dtype=torch.long, device=sample.device)
        else:
            timesteps = timestep
            if len(timesteps.shape) == 0:
                timesteps = timesteps[None]
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)
        encoded_cond = dict(external_cond)
        context = encoded_cond.pop(self.context_key, None)
        if context is None:
            raise ValueError(
                f"CrossAttentionDecoder1D expected a {self.context_key!r} entry in external_cond."
            )

        cond_flat = flatten_and_concat_leaf_tensors(encoded_cond, device=sample.device)
        global_feature = torch.cat([global_feature, cond_flat], dim=-1)

        x = sample
        h = []

        for stage in self.down_modules:
            resnet1, resnet2, cross_attn, downsample = stage  # type: ignore[misc]
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = cross_attn(x, context)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        x = self.mid_cross_attn(x, context)

        for stage in self.up_modules:
            resnet1, resnet2, cross_attn, upsample = stage  # type: ignore[misc]
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet1(x, global_feature)
            x = resnet2(x, global_feature)
            x = cross_attn(x, context)
            x = upsample(x)

        x = self.final_conv(x)
        return x.moveaxis(-1, -2)
