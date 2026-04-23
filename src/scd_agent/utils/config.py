"""配置加载工具。

`config.yaml` 是系统运行的唯一入口配置，包括数据路径、预处理参数、
模型结构、训练超参、风险阈值以及 Agent 决策模板等。本模块将 YAML
文件映射为带点访问的 `Config` 对象，方便在各模块中直接使用。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


# YAML 默认写的占位符；出现这个值说明用户还没填真实数据路径
_PLACEHOLDER_PREFIXES = ("/path/to/",)

# 项目根目录：<repo>/src/scd_agent/utils/config.py → parents[3] = <repo>
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Config:
    """用字典 + 点访问包装配置，保留原始 dict 以便序列化。"""

    raw: Dict[str, Any]

    def __getattr__(self, key: str) -> Any:
        if key == "raw":
            raise AttributeError(key)
        if key in self.raw:
            value = self.raw[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no key '{key}'")

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def __contains__(self, key: str) -> bool:
        return key in self.raw

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        return self.raw


def resolve_project_path(path: str | Path) -> Path:
    """把相对路径解析到项目根目录（便于 cwd 不在 repo 根时仍可运行）。"""
    p = Path(path)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def load_config(path: str | Path) -> Config:
    """读取 YAML 配置文件并返回 `Config`。

    若传入相对路径，会自动解析到项目根目录下查找。
    """
    resolved = resolve_project_path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"配置文件未找到: {resolved}\n"
            f"请确认路径，或从项目根目录运行（默认使用 config/config.yaml）。"
        )
    with resolved.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data)}")
    return Config(data)


def ensure_dataset_paths(cfg: Config, require_ptbxl: bool = False) -> None:
    """校验数据路径是否已填写（非占位符）且真实存在。

    在训练 / 推理入口处调用，把"占位符没改"这类低级错误尽早拦下。
    """
    mitbih_dir = str(cfg.data.mitbih_dir)
    if any(mitbih_dir.startswith(p) for p in _PLACEHOLDER_PREFIXES):
        raise RuntimeError(
            "检测到 data.mitbih_dir 仍为占位符路径。\n"
            "请编辑 config/config.yaml 将其改为本地 MIT-BIH 解压目录的绝对路径。\n"
            "MIT-BIH 可从 PhysioNet 获取：https://physionet.org/content/mitdb/"
        )
    if not Path(mitbih_dir).exists():
        raise FileNotFoundError(
            f"MIT-BIH 目录不存在: {mitbih_dir}\n"
            f"请确认本地数据已下载并解压，并在 config.yaml 中填写正确路径。"
        )

    if require_ptbxl:
        ptbxl_dir = str(cfg.data.ptbxl_dir)
        if any(ptbxl_dir.startswith(p) for p in _PLACEHOLDER_PREFIXES):
            raise RuntimeError("PTB-XL 路径仍为占位符，请先在 config.yaml 中配置。")
        if not Path(ptbxl_dir).exists():
            raise FileNotFoundError(f"PTB-XL 目录不存在: {ptbxl_dir}")
