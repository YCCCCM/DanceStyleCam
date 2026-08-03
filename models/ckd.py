"""Camera Keyframe Detection model construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.nn import functional as F

from .backbone import MusicDanceCameraKeyframeDecoder


def build_ckd_model(config: Mapping[str, Any]):
    values = config.get("model", config)
    return MusicDanceCameraKeyframeDecoder(
        nfeats=1,
        history_len=int(values.get("history_len", 60)),
        inference_len=int(values.get("inference_len", 60)),
        latent_dim=int(values.get("latent_dim", 512)),
        ff_size=int(values.get("ff_size", 1024)),
        num_layers=int(values.get("num_layers", 8)),
        num_heads=int(values.get("num_heads", 8)),
        dropout=float(values.get("dropout", 0.1)),
        m_cond_feature_dim=35,
        p_cond_dim=180,
        activation=F.gelu,
    )
