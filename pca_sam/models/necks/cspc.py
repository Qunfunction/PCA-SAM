"""Cross-scale photonic semantic calibration (CSPC) for PCA-SAM."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from mmengine.model import BaseModule
from mmdet.models.necks import PAFPN
from mmdet.registry import MODELS

from ..sam.common import LayerNorm2d


class _Projection(nn.Conv2d):
    """Keep the experiment's convolution and normalization parameter layout."""

    def __init__(self, in_channels, out_channels, norm):
        super().__init__(in_channels, out_channels, 1, bias=not norm)
        if norm == "LN":
            self.norm = LayerNorm2d(out_channels)
        elif not norm:
            self.norm = None
        else:
            raise ValueError("The released CSPC configuration supports LN or no normalization.")

    def forward(self, x):
        x = super().forward(x)
        return self.norm(x) if self.norm is not None else x


@MODELS.register_module()
class CSPCNeck(BaseModule):
    """Calibrate four SAM encoder depths on four spatial grids, then build P2-P6.

    Shallow encoder features supply queries; pooled deep features supply keys
    and values. Attention output is concatenated with unpooled deep features.
    Module attribute names retain compatibility with experiment checkpoints.
    """

    def __init__(self, patch_size, embed_dim, out_channels=256,
                 scale_factors=(4.0, 2.0, 1.0, 0.5), top_block=None,
                 norm="LN", square_pad=1024,
                 target_sizes=((256, 256), (128, 128), (64, 64), (32, 32))):
        super().__init__()
        if len(scale_factors) != 4 or len(target_sizes) != 4:
            raise ValueError("CSPC requires four target grids.")
        self.fusion = PAFPN(in_channels=[256] * 4,
                            out_channels=out_channels, num_outs=5)
        self.scale_factors = scale_factors
        self.size = list(target_sizes)
        strides = [int(patch_size / scale) for scale in scale_factors]
        if any(b != 2 * a for a, b in zip(strides, strides[1:])):
            raise ValueError("Feature strides must be contiguous powers of two.")
        self.fpn = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(_Projection(embed_dim, 64, norm))
                for _ in scale_factors
            ]) for _ in range(4)
        ])
        self.top_block = top_block
        self._out_feature_strides = {
            "p{}".format(int(math.log2(stride))): stride for stride in strides
        }
        if top_block is not None:
            last_stage = int(math.log2(strides[-1]))
            for stage in range(last_stage, last_stage + top_block.num_levels):
                self._out_feature_strides["p{}".format(stage + 1)] = 2 ** (stage + 1)
        self._out_features = list(self._out_feature_strides)
        self._out_feature_channels = {name: out_channels for name in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad
        self.cross_attention = nn.MultiheadAttention(128, 4, batch_first=True)
        self.kv_pool = nn.AdaptiveAvgPool2d((32, 32))

    @property
    def padding_constraints(self):
        return {"size_divisiblity": self._size_divisibility,
                "square_size": self._square_pad}

    def _align_depth(self, feature, stages):
        aligned = [
            stage(F.interpolate(feature, size, mode="bilinear", align_corners=False))
            for size, stage in zip(self.size, stages)
        ]
        if self.top_block is not None:
            index = self._out_features.index(self.top_block.in_feature)
            aligned.extend(self.top_block(aligned[index]))
        return aligned

    def forward(self, features):
        if len(features) < 4:
            raise ValueError("CSPC needs four encoder-depth feature maps.")
        aligned = [self._align_depth(features[i], self.fpn[i]) for i in range(4)]
        calibrated = []
        for j in range(4):
            query = torch.cat([aligned[0][j], aligned[1][j]], dim=1)
            context = torch.cat([aligned[2][j], aligned[3][j]], dim=1)
            batch, channels, height, width = query.shape
            pooled = self.kv_pool(context) if height > 32 else context
            query_tokens = query.view(batch, channels, -1).permute(0, 2, 1)
            context_tokens = pooled.view(batch, channels, -1).permute(0, 2, 1)
            attention, _ = self.cross_attention(
                query=query_tokens, key=context_tokens, value=context_tokens)
            attention = attention.permute(0, 2, 1).view(batch, channels, height, width)
            calibrated.append(torch.cat([attention, context], dim=1))
        return tuple(self.fusion(calibrated))
