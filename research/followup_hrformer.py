"""Paper-mechanism HRFormer-S adapter; audited B/T inputs and normalized loss.

This is not a reproduction of2606.27671's outdoor/random-split result.
Local MIT HRFormer backbone lineage, stages32/64/128/256, scalar FiLM at
all four stages,90 pattern tokens and two8-head decoder attention layers.
"""
import json
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from baseline_model import Block
from vendor.hrformer.backbone import HRFormer


class StageFiLM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.project = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 2 * sum(channels)))
        nn.init.zeros_(self.project[-1].weight)
        nn.init.zeros_(self.project[-1].bias)

    def forward(self, features, scalar):
        scale, bias = self.project(scalar).chunk(2, dim=-1)
        scales, biases = scale.split(self.channels, dim=1), bias.split(self.channels, dim=1)
        return [x * (1 + s[:, :, None, None]) + b[:, :, None, None]
                for x, s, b in zip(features, scales, biases)]


class HRFormerConditioned(nn.Module):
    def __init__(self, conditioned=True):
        super().__init__()
        config = json.loads((Path(__file__).parent / 'vendor/hrformer/config.json').read_text())
        self.encoder = HRFormer(config, in_channels=2)
        # Decoder construction precedes conditioning for paired source/target initialization.
        self.decoder = nn.ModuleList([Block(256 + 128, 128), Block(128 + 64, 64), Block(64 + 32, 32)])
        self.refine = nn.ModuleList([Block(32, 16), Block(16, 16)])
        self.head = nn.Conv2d(16, 1, 1)
        self.films = nn.ModuleList([StageFiLM([256]), StageFiLM([32, 64]),
                                    StageFiLM([32, 64, 128]), StageFiLM([32, 64, 128, 256])]) if conditioned else None
        self.pattern = nn.Sequential(nn.Conv1d(2, 256, 4, stride=4), nn.GELU()) if conditioned else None
        self.cross_attention = nn.TransformerDecoder(nn.TransformerDecoderLayer(
            d_model=256, nhead=8, dim_feedforward=1024, dropout=0., batch_first=True), num_layers=2) if conditioned else None

    def features(self, x, scalar):
        e = self.encoder
        x = e.relu(e.bn1(e.conv1(x)))
        x = e.relu(e.bn2(e.conv2(x)))
        x = e.layer1(x)
        if self.films is not None: x = self.films[0]([x], scalar)[0]
        inputs = [layer(x) if layer is not None else x for layer in e.transition1]
        features = e.stage2(inputs)
        if self.films is not None: features = self.films[1](features, scalar)
        inputs = [layer(features[-1]) if layer is not None else features[i] for i, layer in enumerate(e.transition2)]
        features = e.stage3(inputs)
        if self.films is not None: features = self.films[2](features, scalar)
        inputs = [layer(features[-1]) if layer is not None else features[i] for i, layer in enumerate(e.transition3)]
        features = e.stage4(inputs)
        return self.films[3](features, scalar) if self.films is not None else features

    def forward(self, spatial, condition=None, pattern=None):
        height, width = spatial.shape[-2:]
        spatial = F.pad(spatial, (0, (-width) % 32, 0, (-height) % 32))
        if self.films is not None and (condition is None or pattern is None):
            raise ValueError('Conditioned HRT requires scalar and pattern')
        features = self.features(spatial, condition)
        x = features[-1]
        if self.pattern is not None:
            tokens = self.pattern(pattern).transpose(1, 2)
            assert tokens.shape[1] == 90
            batch, channels, h, w = x.shape
            query = x.flatten(2).transpose(1, 2)
            x = self.cross_attention(query, tokens).transpose(1, 2).reshape(batch, channels, h, w)
        for block, skip in zip(self.decoder, reversed(features[:-1])):
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            x = block(torch.cat([x, skip], dim=1))
        for block in self.refine:
            x = block(F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False))
        return torch.sigmoid(self.head(x))[:, :, :height, :width]
