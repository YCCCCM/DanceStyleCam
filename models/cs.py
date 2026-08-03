"""Style-conditioned baseline Camera Synthesis model construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from torch.nn import functional as F

from .backbone import EditableDanceCameraDecoder_ForVelocity


def build_cs_model(config: Mapping[str, Any]):
    values = config.get("model", config)
    camera_format = str(values.get("camera_format", "polar"))
    if camera_format not in {"polar", "centric"}:
        raise ValueError(f"Unsupported camera format: {camera_format}")
    camera_features = 8 if camera_format == "polar" else 7
    return EditableDanceCameraDecoder_ForVelocity(
        nfeats=camera_features,
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
        style_feat_dim=16,
    )
