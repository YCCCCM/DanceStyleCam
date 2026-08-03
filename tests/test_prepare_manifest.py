from data.prepare import _empty_manifest, _load_or_create_manifest, _write_json_atomic


def test_written_manifest_can_be_resumed(tmp_path) -> None:
    manifest = _empty_manifest()
    _write_json_atomic(tmp_path / "manifest.json", manifest)
    assert _load_or_create_manifest(tmp_path) == manifest

