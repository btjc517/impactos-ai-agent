from ir_planner import _gather_catalog_snippets


def test_catalog_snippets_empty_ok(tmp_path, monkeypatch):
    # Create an empty metrics dir; gathering should not fail
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    out = _gather_catalog_snippets(str(metrics_dir), "volunteering")
    assert isinstance(out, list)


