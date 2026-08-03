import numpy as np

from metric.evaluate import _style_metrics


def test_style_metrics_include_all_legacy_summary_fields() -> None:
    reference = [
        ("A", np.arange(30, dtype=np.float32)),
        ("B", np.arange(30, dtype=np.float32)[::-1]),
    ]
    generated = [
        ("A", np.arange(30, dtype=np.float32)),
        ("B", np.arange(30, dtype=np.float32)[::-1]),
    ]

    result = _style_metrics(generated, reference)

    assert result["strict_accuracy"] == 1.0
    assert result["max_target_similarity"] >= result["median_target_similarity"]
    assert result["median_target_similarity"] >= result["min_target_similarity"]
    assert result["confusion_matrix"] == {"A-A": 1, "B-B": 1}
