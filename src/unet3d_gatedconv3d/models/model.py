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
    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv3d(input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        combined = torch.cat([x, h_prev], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_hidden(self, batch_size: int, time_steps: int, height: int, width: int, device):
        h = torch.zeros(batch_size, self.hidden_dim, time_steps, height, width, device=device)
        c = torch.zeros(batch_size, self.hidden_dim, time_steps, height, width, device=device)
        return h, c


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
        batch_size, _channels, time_steps, height, width = x.size()
        device = x.device

        h_states = []
        c_states = []
        for layer in self.layers:
            hi, ci = layer.init_hidden(batch_size, time_steps, height, width, device)
            h_states.append(hi)
            c_states.append(ci)

        current = x
        for i, layer in enumerate(self.layers):
            hi, ci = layer(current, h_states[i], c_states[i])
            current = hi
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

