"""PyTorch U-Net models for CFO experiments."""

from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(
    t: torch.Tensor,
    embed_dim: int,
) -> torch.Tensor:
    """Create sinusoidal time embeddings.

    Parameters
    ----------
    t : torch.Tensor
        Time values, shape (B,).
    embed_dim : int
        Dimension of the time embedding.

    Returns
    -------
    torch.Tensor
        Time embedding of shape (B, embed_dim).
    """
    half_dim = embed_dim // 2

    freqs = torch.exp(
        torch.linspace(
            0.0,
            math.log(10000.0),
            half_dim,
            device=t.device,
            dtype=t.dtype,
        )
    )

    angles = t[:, None] * freqs[None, :]

    return torch.cat(
        [torch.sin(angles), torch.cos(angles)],
        dim=-1,
    )


class TimeEmbeddingMLP(nn.Module):
    """Project sinusoidal time features."""

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        x = sinusoidal_time_embedding(t, self.embed_dim)
        return self.net(x)


class DownsampleBlock2D(nn.Module):
    """2D residual downsampling block."""

    def __init__(
        self,
        in_channels: int,
        features: int,
        time_emb_dim: int,
        use_time_emb: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()

        self.use_time_emb = use_time_emb

        self.conv1 = nn.Conv2d(
            in_channels,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=features,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=features,
        )

        # Time embedding -> feature channels
        if use_time_emb:
            self.time_proj = nn.Linear(
                time_emb_dim,
                features,
            )

        # Downsampling convolution
        self.downsample = nn.Conv2d(
            features,
            features,
            kernel_size=4,
            stride=2,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)

        if self.use_time_emb and t_emb is not None:
            t_out = self.time_proj(t_emb)

            # (B, C) -> (B, C, 1, 1)
            x = x + t_out[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)

        # Save before downsampling
        skip = x

        x_down = self.downsample(x)

        return x_down, skip


class UpsampleBlock2D(nn.Module):
    """2D residual upsampling block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        features: int,
        time_emb_dim: int,
        use_time_emb: bool = True,
        num_groups: int = 8,
    ):
        super().__init__()

        self.use_time_emb = use_time_emb

        # Corresponds to Flax ConvTranspose
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            features,
            kernel_size=4,
            stride=2,
            padding=1,
        )

        # After concatenating x and skip
        self.conv1 = nn.Conv2d(
            features + skip_channels,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=features,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=features,
        )

        if use_time_emb:
            self.time_proj = nn.Linear(
                time_emb_dim,
                features,
            )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
        t_emb: Optional[torch.Tensor],
    ) -> torch.Tensor:

        x = self.upsample(x)

        # Usually the spatial dimensions match exactly.
        # The interpolation below makes the model robust to odd sizes.
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [x, skip],
            dim=1,
        )

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.silu(x)

        if self.use_time_emb and t_emb is not None:
            t_out = self.time_proj(t_emb)
            x = x + t_out[:, :, None, None]

        x = self.conv2(x)
        x = self.norm2(x)
        x = F.silu(x)

        return x


class UNet2D(nn.Module):
    """2D U-Net with optional time and condition inputs.

    Input convention:
        x: (B, C, H, W)

    Optional condition:
        c: (B, C_condition, H, W)

    Time:
        t: (B,)

    Output:
        (B, C_out, H, W)
    """

    def __init__(
        self,
        input_channels: int = 1,
        channel_mult: Tuple[int, ...] = (64, 128, 256),
        time_emb_dim: int = 256,
        use_condition: bool = False,
        condition_channels: int = 0,
        num_groups: int = 8,
    ):
        super().__init__()

        self.channel_mult = channel_mult
        self.time_emb_dim = time_emb_dim
        self.use_condition = use_condition

        # ---------------------------------------------------------
        # Input channels
        # ---------------------------------------------------------

        if use_condition:
            if condition_channels <= 0:
                raise ValueError(
                    "condition_channels must be > 0 when "
                    "use_condition=True"
                )

            in_channels = input_channels + condition_channels
        else:
            in_channels = input_channels

        # ---------------------------------------------------------
        # Time embedding
        # ---------------------------------------------------------

        self.time_embedding = TimeEmbeddingMLP(
            embed_dim=time_emb_dim,
            hidden_dim=128,
        )

        # ---------------------------------------------------------
        # Downsampling path
        # ---------------------------------------------------------

        self.down_blocks = nn.ModuleList()

        current_channels = in_channels

        for features in channel_mult:

            block = DownsampleBlock2D(
                in_channels=current_channels,
                features=features,
                time_emb_dim=time_emb_dim,
                use_time_emb=True,
                num_groups=num_groups,
            )

            self.down_blocks.append(block)

            current_channels = features

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        bottleneck_channels = channel_mult[-1]

        self.bottleneck_conv1 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
        )

        self.bottleneck_norm1 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=bottleneck_channels,
        )

        self.bottleneck_time = nn.Linear(
            time_emb_dim,
            bottleneck_channels,
        )

        self.bottleneck_conv2 = nn.Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1,
        )

        self.bottleneck_norm2 = nn.GroupNorm(
            num_groups=num_groups,
            num_channels=bottleneck_channels,
        )

        # ---------------------------------------------------------
        # Upsampling path
        # ---------------------------------------------------------

        self.up_blocks = nn.ModuleList()

        current_channels = bottleneck_channels

        for features in reversed(channel_mult):

            block = UpsampleBlock2D(
                in_channels=current_channels,
                skip_channels=features,
                features=features,
                time_emb_dim=time_emb_dim,
                use_time_emb=True,
                num_groups=num_groups,
            )

            self.up_blocks.append(block)

            current_channels = features

        # ---------------------------------------------------------
        # Output
        # ---------------------------------------------------------

        self.output_conv = nn.Conv2d(
            current_channels,
            input_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        c: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # ---------------------------------------------------------
        # Conditioning
        # ---------------------------------------------------------

        if self.use_condition and c is not None:
            x = torch.cat(
                [x, c],
                dim=1,
            )

        # ---------------------------------------------------------
        # Time embedding
        # ---------------------------------------------------------

        if t is not None:
            t_emb = self.time_embedding(t)
        else:
            t_emb = None

        # ---------------------------------------------------------
        # Downsampling
        # ---------------------------------------------------------

        skips = []

        h = x

        for block in self.down_blocks:
            h, skip = block(h, t_emb)
            skips.append(skip)

        # ---------------------------------------------------------
        # Bottleneck
        # ---------------------------------------------------------

        h = self.bottleneck_conv1(h)
        h = self.bottleneck_norm1(h)
        h = F.silu(h)

        if t_emb is not None:
            t_out = self.bottleneck_time(t_emb)
            h = h + t_out[:, :, None, None]

        h = self.bottleneck_conv2(h)
        h = self.bottleneck_norm2(h)
        h = F.silu(h)

        # ---------------------------------------------------------
        # Upsampling
        # ---------------------------------------------------------

        for block in self.up_blocks:

            skip = skips.pop()

            h = block(
                h,
                skip,
                t_emb,
            )

        # ---------------------------------------------------------
        # Output
        # ---------------------------------------------------------

        h = self.output_conv(h)

        return h


__all__ = [
    "sinusoidal_time_embedding",
    "TimeEmbeddingMLP",
    "UNet2D",
]