from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(1, 3, 3),
        stride=(1, 1, 1),
        padding=(0, 1, 1),
    ):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm3d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32, num_levels: int = 5, out_channels: int = 64):
        super().__init__()
        self.num_levels = num_levels
        self.initial_conv = ResidualBlock3D(in_channels, base_channels)

        self.down_convs = nn.ModuleList()
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        channels = base_channels
        for _ in range(num_levels):
            block = ResidualBlock3D(channels, channels * 2)
            self.down_convs.append(block)
            channels *= 2

        self.bottleneck = ResidualBlock3D(channels, channels)

        self.up_transposes = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for _ in range(num_levels):
            self.up_transposes.append(
                nn.ConvTranspose3d(channels, channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
            self.up_convs.append(ResidualBlock3D(channels, channels // 2))
            channels //= 2

        self.final_conv = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []
        out = self.initial_conv(x)

        for down in self.down_convs:
            skip_connections.append(out)
            out = self.pool(out)
            out = down(out)

        out = self.bottleneck(out)

        for i in range(self.num_levels):
            out = self.up_transposes[i](out)
            skip = skip_connections[-(i + 1)]
            if out.shape != skip.shape:
                diff_t = skip.size(2) - out.size(2)
                diff_h = skip.size(3) - out.size(3)
                diff_w = skip.size(4) - out.size(4)
                out = F.pad(
                    out,
                    [
                        diff_w // 2,
                        diff_w - diff_w // 2,
                        diff_h // 2,
                        diff_h - diff_h // 2,
                        diff_t // 2,
                        diff_t - diff_t // 2,
                    ],
                )
            out = torch.cat([out, skip], dim=1)
            out = self.up_convs[i](out)

        return self.final_conv(out)


class GatedConv3DBlock(nn.Module):
    """Gated Conv3D block, faithful to the VONA-DL paper Sec. 3.3.2 (Eq. 4-6).

    A single Conv3D maps the layer input to 3*hidden_dim channels, split into
    (i, o, g). Applied as a purely feed-forward gated activation on the
    current volume (no recurrent cell state carried across time steps, per
    the paper: "Unlike a recurrent cell that maintains a separate cell state
    across time steps, our implementation utilizes these components to
    perform a gated activation on the current volume locally"):
        c~ = sigmoid(i) * tanh(g)
        Z  = sigmoid(o) * tanh(c~)
    """

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv3d(input_dim, 3 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = self.conv(x)
        i, o, g = torch.chunk(gates, 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_tilde = i * g
        return o * torch.tanh(c_tilde)


class StackedConv3D(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], kernel_size: int = 3, padding: int = 1):
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims must be a non-empty list")

        self.layers = nn.ModuleList()
        current_dim = input_dim
        for hd in hidden_dims:
            self.layers.append(GatedConv3DBlock(current_dim, hd, kernel_size, padding))
            current_dim = hd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x
        for layer in self.layers:
            current = layer(current)
        return current


class Model(nn.Module):
    def __init__(
        self,
        unet_in_channels: int = 3,
        unet_base_channels: int = 32,
        unet_num_levels: int = 5,
        unet_out_channels: int = 64,
        stackedconv_hidden_dims: List[int] | None = None,
        stackedconv_kernel_size: int = 3,
        stackedconv_padding: int = 1,
        final_out_channels: int = 3,
    ):
        super().__init__()
        if stackedconv_hidden_dims is None:
            stackedconv_hidden_dims = [64, 128, 256, 512]
        stacked_out_dim = stackedconv_hidden_dims[-1]

        self.unet = UNet3D(
            in_channels=unet_in_channels,
            base_channels=unet_base_channels,
            num_levels=unet_num_levels,
            out_channels=unet_out_channels,
        )
        self.stackedConv = StackedConv3D(
            input_dim=unet_out_channels,
            hidden_dims=stackedconv_hidden_dims,
            kernel_size=stackedconv_kernel_size,
            padding=stackedconv_padding,
        )
        self.final_conv = nn.Conv3d(stacked_out_dim, final_out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.unet(x)
        stacked_out = self.stackedConv(features)
        return self.final_conv(stacked_out)


def build_model(cfg: dict) -> Model:
    unet_cfg = cfg["unet"]
    stacked_cfg = cfg["stacked_conv"]
    return Model(
        unet_in_channels=int(unet_cfg["in_channels"]),
        unet_base_channels=int(unet_cfg["base_channels"]),
        unet_num_levels=int(unet_cfg["num_levels"]),
        unet_out_channels=int(unet_cfg["out_channels"]),
        stackedconv_hidden_dims=list(stacked_cfg["hidden_dims"]),
        stackedconv_kernel_size=int(stacked_cfg["kernel_size"]),
        stackedconv_padding=int(stacked_cfg["padding"]),
        final_out_channels=int(cfg["final_out_channels"]),
    )

