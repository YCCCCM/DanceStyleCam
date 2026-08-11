from common.config import load_config
from common.paths import PROJECT_ROOT


def test_public_metadata_is_present() -> None:
    required = (
        "README.md",
        "MIGRATION_SCOPE.md",
        "DCM_data/music_style_16cat.json",
        "DCM_data/split/long2short.json",
        "DCM_data/split/train_pre.json",
        "DCM_data/split/test_pre.json",
        "configs/data/dcm_style_pp.yaml",
        "configs/train/ckd.yaml",
        "configs/train/cs.yaml",
        "configs/train/cs_nogan.yaml",
        "configs/infer/default.yaml",
        "configs/infer/controlled_test.yaml",
        "configs/infer/custom.yaml",
        "train/train_ckd.py",
        "train/train_cs.py",
        "infer/generate.py",
        "infer/generate_test_controlled.py",
        "infer/generate_custom.py",
    )
    assert all((PROJECT_ROOT / path).is_file() for path in required)


def test_runtime_roots_are_separate() -> None:
    assert (PROJECT_ROOT / "runs/README.md").is_file()
    assert (PROJECT_ROOT / "generation/README.md").is_file()
    assert (PROJECT_ROOT / "DCM-style++/README.md").is_file()


def test_anchor_dance_style_top_level_modules() -> None:
    assert not (PROJECT_ROOT / "dancestylecam").exists()
    for directory in ("common", "data", "models", "train", "infer", "metric", "tools"):
        assert (PROJECT_ROOT / directory).is_dir()


def test_cs_gan_switch_is_config_only() -> None:
    gan = load_config(PROJECT_ROOT / "configs/train/cs.yaml")
    no_gan = load_config(PROJECT_ROOT / "configs/train/cs_nogan.yaml")
    assert gan["training"]["use_gan"] is True
    assert no_gan["training"]["use_gan"] is False
    assert gan["model"] == no_gan["model"]


def test_inference_matches_original_checkpoint_weight_selection() -> None:
    config = load_config(PROJECT_ROOT / "configs/infer/default.yaml")
    assert config["generation"]["checkpoint_weights"] == "model"
