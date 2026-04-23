import pytest

from scd_agent.utils.config import Config, ensure_dataset_paths, load_config


def _write_yaml(tmp_path, text: str):
    p = tmp_path / "config.yaml"
    p.write_text(text, encoding="utf-8")
    return p


def test_config_dot_and_get(tmp_path):
    p = _write_yaml(tmp_path, """
data:
  mitbih_dir: "/tmp/anywhere"
  list: [1, 2]
nested:
  inner:
    a: 1
""")
    cfg = load_config(str(p))
    assert cfg.data.mitbih_dir == "/tmp/anywhere"
    assert cfg.data.list == [1, 2]
    assert cfg.nested.inner.a == 1
    assert cfg.data.get("missing", "x") == "x"
    assert "data" in cfg


def test_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "nope.yaml"))


def test_ensure_dataset_paths_rejects_placeholder(tmp_path):
    p = _write_yaml(tmp_path, """
data:
  mitbih_dir: "/path/to/mitbih"
""")
    cfg = load_config(str(p))
    with pytest.raises(RuntimeError):
        ensure_dataset_paths(cfg)


def test_ensure_dataset_paths_rejects_missing_dir(tmp_path):
    p = _write_yaml(tmp_path, f"""
data:
  mitbih_dir: "{tmp_path / 'does_not_exist'}"
""")
    cfg = load_config(str(p))
    with pytest.raises(FileNotFoundError):
        ensure_dataset_paths(cfg)


def test_ensure_dataset_paths_accepts_existing(tmp_path):
    real = tmp_path / "mitbih"
    real.mkdir()
    p = _write_yaml(tmp_path, f"""
data:
  mitbih_dir: "{real}"
""")
    cfg = load_config(str(p))
    # 不抛错即通过
    ensure_dataset_paths(cfg)
