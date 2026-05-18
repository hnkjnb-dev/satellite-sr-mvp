from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_coord_grid(height: int, width: int, device: torch.device) -> torch.Tensor:
    ys = torch.linspace(
        -1 + 1 / height, 1 - 1 / height, steps=height, device=device, dtype=torch.float32
    )
    xs = torch.linspace(
        -1 + 1 / width, 1 - 1 / width, steps=width, device=device, dtype=torch.float32
    )
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).view(-1, 2)


class ResidualDenseBlock(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(
            channels + growth_channels * 2, growth_channels, 3, 1, 1
        )
        self.conv4 = nn.Conv2d(
            channels + growth_channels * 3, growth_channels, 3, 1, 1
        )
        self.conv5 = nn.Conv2d(channels + growth_channels * 4, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.act(self.conv1(x))
        x2 = self.act(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.act(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.act(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))
        x5 = self.conv5(torch.cat([x, x1, x2, x3, x4], dim=1))
        return x + x5 * 0.2


class RRDB(nn.Module):
    def __init__(self, channels: int, growth_channels: int = 32) -> None:
        super().__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return x + out * 0.2


class FiLMBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        meta_dim: int,
        hidden_dim: int = 64,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.film_scale = float(film_scale)
        self.meta_mlp = nn.Sequential(
            nn.Linear(meta_dim, hidden_dim),
            nn.GELU(),
        )
        self.to_gamma_beta = nn.Linear(hidden_dim, channels * 2)
        # True identity start: gamma=0, beta=0.
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if meta.ndim != 2:
            raise ValueError(f"Expected encoder meta [B, D], got {tuple(meta.shape)}")
        h = self.meta_mlp(meta)
        gamma_beta = self.to_gamma_beta(h)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1.0 + self.film_scale * gamma) + self.film_scale * beta


class RRDBEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        feat_channels: int = 48,
        num_blocks: int = 12,
        growth_channels: int = 24,
        encoder_meta_dim: int = 6,
        film_hidden_dim: int = 64,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.conv_first = nn.Conv2d(in_channels, feat_channels, 3, 1, 1)
        self.trunk = nn.Sequential(
            *[RRDB(feat_channels, growth_channels) for _ in range(num_blocks)]
        )
        self.trunk_conv = nn.Conv2d(feat_channels, feat_channels, 3, 1, 1)
        self.film_after_conv_first = FiLMBlock(
            channels=feat_channels,
            meta_dim=encoder_meta_dim,
            hidden_dim=film_hidden_dim,
            film_scale=film_scale,
        )
        self.film_after_trunk = FiLMBlock(
            channels=feat_channels,
            meta_dim=encoder_meta_dim,
            hidden_dim=film_hidden_dim,
            film_scale=film_scale,
        )

    def forward(self, x: torch.Tensor, encoder_meta: torch.Tensor) -> torch.Tensor:
        feat = self.conv_first(x)
        feat = self.film_after_conv_first(feat, encoder_meta)
        trunk = self.trunk_conv(self.trunk(feat))
        trunk = self.film_after_trunk(trunk, encoder_meta)
        return feat + trunk


class ImplicitDecoder3x3(nn.Module):
    def __init__(
        self,
        feat_channels: int = 48,
        meta_dim: int = 5,
        hidden_dim: int = 256,
        num_layers: int = 4,
        out_channels: int = 3,
        sampling_temperature: float = 0.0,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        in_dim = feat_channels * 9 + 2 + meta_dim
        self.sampling_temperature = float(sampling_temperature)
        self.register_buffer(
            "offset_grid",
            torch.tensor(
                [
                    (-1.0, -1.0),
                    (0.0, -1.0),
                    (1.0, -1.0),
                    (-1.0, 0.0),
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (-1.0, 1.0),
                    (0.0, 1.0),
                    (1.0, 1.0),
                ],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        layers = []
        dim = in_dim
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(dim, hidden_dim), nn.GELU()])
            dim = hidden_dim
        # Predict residual mean and log_sigma per channel.
        layers.append(nn.Linear(dim, self.out_channels * 2))
        self.mlp = nn.Sequential(*layers)

    def _sample_3x3(
        self, feat: torch.Tensor, coord: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, channels, feat_h, feat_w = feat.shape
        num_query = coord.shape[1]
        eps = 1e-6

        feat_coord = make_coord_grid(feat_h, feat_w, feat.device)
        feat_coord = feat_coord.view(1, feat_h, feat_w, 2).permute(0, 3, 1, 2)
        feat_coord = feat_coord.expand(batch, -1, -1, -1)

        q_coord = F.grid_sample(
            feat_coord,
            coord.view(batch, num_query, 1, 2),
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        )
        q_coord = q_coord.squeeze(-1).permute(0, 2, 1)

        rel_coord = coord - q_coord
        rel_coord[..., 0] *= feat_w
        rel_coord[..., 1] *= feat_h

        step_x = 2.0 / feat_w
        step_y = 2.0 / feat_h

        sampled = []
        for offset_y in (-1, 0, 1):
            for offset_x in (-1, 0, 1):
                shifted = coord.clone()
                shifted[..., 0] = (shifted[..., 0] + step_x * offset_x).clamp(
                    -1 + eps, 1 - eps
                )
                shifted[..., 1] = (shifted[..., 1] + step_y * offset_y).clamp(
                    -1 + eps, 1 - eps
                )
                val = F.grid_sample(
                    feat,
                    shifted.view(batch, num_query, 1, 2),
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=False,
                )
                val = val.squeeze(-1).permute(0, 2, 1)
                sampled.append(val)

        sampled_stack = torch.stack(sampled, dim=2)  # [B, Q, 9, C]
        if self.sampling_temperature > 0.0:
            offset_grid = self.offset_grid.to(device=rel_coord.device, dtype=rel_coord.dtype)
            offset_grid = offset_grid.view(1, 1, 9, 2)
            dist2 = ((rel_coord.unsqueeze(2) - offset_grid) ** 2).sum(dim=-1)
            logits = -dist2 / max(self.sampling_temperature, 1e-6)
            weights = torch.softmax(logits, dim=-1).unsqueeze(-1)
            sampled_stack = sampled_stack * weights

        sampled_flat = sampled_stack.reshape(batch, num_query, -1)
        return sampled_flat, rel_coord

    def forward(
        self, feat: torch.Tensor, coord: torch.Tensor, meta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sampled, rel_coord = self._sample_3x3(feat, coord)
        decoder_inp = torch.cat([sampled, rel_coord, meta], dim=-1)
        out = self.mlp(decoder_inp)
        residual, log_sigma = torch.split(out, self.out_channels, dim=-1)
        return residual, log_sigma


class RRDBLINF(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        feat_channels: int = 48,
        num_blocks: int = 12,
        growth_channels: int = 24,
        decoder_hidden_dim: int = 256,
        decoder_layers: int = 4,
        meta_dim: int = 8,
        encoder_meta_dim: int = 6,
        sampling_temperature: float = 0.0,
        log_sigma_min: float = -3.0,
        log_sigma_max: float = 1.0,
        film_hidden_dim: int = 64,
        film_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder_meta_dim = int(encoder_meta_dim)
        self.log_sigma_min = float(log_sigma_min)
        self.log_sigma_max = float(log_sigma_max)
        self.encoder = RRDBEncoder(
            in_channels=in_channels,
            feat_channels=feat_channels,
            num_blocks=num_blocks,
            growth_channels=growth_channels,
            encoder_meta_dim=encoder_meta_dim,
            film_hidden_dim=film_hidden_dim,
            film_scale=film_scale,
        )
        self.decoder = ImplicitDecoder3x3(
            feat_channels=feat_channels,
            meta_dim=meta_dim,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
            out_channels=out_channels,
            sampling_temperature=sampling_temperature,
        )

    def forward(
        self,
        lr: torch.Tensor,
        coord: torch.Tensor,
        meta: torch.Tensor,
        encoder_meta: torch.Tensor | None = None,
        query_chunk_size: int = 8192,
        return_log_sigma: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if encoder_meta is None:
            if meta.ndim != 3 or meta.shape[-1] < (2 + self.encoder_meta_dim):
                raise ValueError(
                    "encoder_meta missing and decoder meta cannot infer it."
                )
            encoder_meta = meta[:, 0, 2 : 2 + self.encoder_meta_dim]
        feat = self.encoder(lr, encoder_meta)
        batch, num_query, _ = coord.shape
        preds = []
        log_sigmas = []
        for start in range(0, num_query, query_chunk_size):
            end = min(start + query_chunk_size, num_query)
            coord_chunk = coord[:, start:end]
            residual, log_sigma = self.decoder(feat, coord_chunk, meta[:, start:end])
            base = F.grid_sample(
                lr,
                coord_chunk.view(batch, end - start, 1, 2),
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            base = base.squeeze(-1).permute(0, 2, 1)
            pred = base + residual
            preds.append(pred)
            log_sigmas.append(log_sigma.clamp(self.log_sigma_min, self.log_sigma_max))

        pred_out = torch.cat(preds, dim=1).view(batch, num_query, -1)
        if return_log_sigma:
            log_sigma_out = torch.cat(log_sigmas, dim=1).view(batch, num_query, -1)
            return pred_out, log_sigma_out
        return pred_out
