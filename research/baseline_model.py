"""Resolution-preserving U-Net; source transfers only the spatial encoder."""
import torch
from torch import nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(1, outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(1, outputs), nn.SiLU(),
        )

    def forward(self, x):
        return self.layers(x)


class SpatialEncoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.blocks = nn.ModuleList([Block(2, channels), Block(channels, 2 * channels),
                                     Block(2 * channels, 4 * channels), Block(4 * channels, 8 * channels)])

    def forward(self, x):
        features = []
        for index, block in enumerate(self.blocks):
            if index:
                x = F.max_pool2d(x, 2)
            x = block(x)
            features.append(x)
        return features


class AntennaCondition(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pattern = nn.Sequential(nn.Conv1d(2, 16, 5, padding=2), nn.SiLU(),
                                     nn.Conv1d(16, 32, 5, padding=2), nn.SiLU(),
                                     nn.AdaptiveAvgPool1d(36), nn.Flatten(),
                                     nn.Linear(32 * 36, 32), nn.SiLU())
        self.scalar = nn.Sequential(nn.Linear(7, 32), nn.SiLU())
        self.film = nn.Linear(64, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, features, condition, pattern):
        vector = torch.cat((self.scalar(condition), self.pattern(pattern)), dim=-1)
        scale, bias = self.film(vector).chunk(2, dim=-1)
        return features * (1 + scale[:, :, None, None]) + bias[:, :, None, None]


class RadioUNetBaseline(nn.Module):
    def __init__(self, *, conditioned=False, base_channels=16):
        super().__init__()
        if base_channels < 1:
            raise ValueError('base_channels must be positive')
        c = base_channels
        # Construction order keeps encoder/decoder initialization paired for A/B.
        self.encoder = SpatialEncoder(c)
        self.decoder = nn.ModuleList([Block(12 * c, 4 * c), Block(6 * c, 2 * c), Block(3 * c, c)])
        self.head = nn.Conv2d(c, 1, 1)
        self.conditioning = AntennaCondition(8 * c) if conditioned else None

    def forward(self, spatial, condition=None, pattern=None):
        height, width = spatial.shape[-2:]
        spatial = F.pad(spatial, (0, (-width) % 8, 0, (-height) % 8))
        features = self.encoder(spatial)
        x = features[-1]
        if self.conditioning is not None:
            if condition is None or pattern is None:
                raise ValueError('Conditioned model requires scalar and pattern')
            x = self.conditioning(x, condition, pattern)
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return torch.sigmoid(self.head(x))[:, :, :height, :width]
