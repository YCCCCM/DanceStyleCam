import pytest

from infer.pipeline import checkpoint_uses_ema


def test_inference_defaults_to_original_non_ema_weights() -> None:
    assert checkpoint_uses_ema({}) is False
    assert checkpoint_uses_ema({"checkpoint_weights": "model"}) is False


def test_inference_can_select_ema_weights() -> None:
    assert checkpoint_uses_ema({"checkpoint_weights": "ema"}) is True


def test_inference_rejects_unknown_checkpoint_weights() -> None:
    with pytest.raises(ValueError, match="checkpoint_weights"):
        checkpoint_uses_ema({"checkpoint_weights": "best"})
