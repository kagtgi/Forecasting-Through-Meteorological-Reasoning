"""Lightweight YAML config loader with dotted access and CLI overrides."""
from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - yaml always present in the env
    yaml = None


class Config(dict):
    """A dict that also supports attribute access and nested dotted get/set."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def __setattr__(self, k: str, v: Any) -> None:
        self[k] = v

    def get_path(self, dotted: str, default: Any = None) -> Any:
        cur: Any = self
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def set_path(self, dotted: str, value: Any) -> None:
        cur: Dict = self
        parts = dotted.split(".")
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = _coerce(value)


def _coerce(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    # JSON list/dict overrides, e.g. eval.lead_times_min=[5,15,30]
    if s[:1] in ("[", "{"):
        import json
        try:
            return json.loads(s)
        except Exception:
            try:
                import ast
                return ast.literal_eval(s)
            except Exception:
                return v
    for caster in (int, float):
        try:
            return caster(s)
        except ValueError:
            pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("none", "null"):
        return None
    return v


def load_config(path: str, overrides: Optional[List[str]] = None) -> Config:
    """Load a YAML config and apply `key.subkey=value` overrides.

    If `paths.root` is overridden (e.g. to a Google Drive / bucket mount on Colab),
    every other `paths.*` entry that still points under the *original* default root is
    automatically rebased onto the new root. This makes a single
    `--override paths.root=/content/drive/MyDrive/asgwm` redirect all caches and
    checkpoints to persistent storage — essential for the resumable <12 h A100 sessions.
    Explicit per-path overrides (e.g. `paths.cache=...`) still win.
    """
    if yaml is None:
        raise RuntimeError("PyYAML is required to load configs")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(copy.deepcopy(data))

    orig_root = str(cfg.get_path("paths.root", "")).rstrip("/\\")
    explicit_path_keys = set()
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must be key=value, got {ov!r}")
        k, v = ov.split("=", 1)
        k = k.strip()
        if k.startswith("paths.") and k != "paths.root":
            explicit_path_keys.add(k)
        cfg.set_path(k, v.strip())

    _rebase_paths(cfg, orig_root, explicit_path_keys)
    return cfg


def _rebase_paths(cfg: "Config", orig_root: str, explicit_path_keys: set) -> None:
    """Rebase paths.* under a newly-overridden paths.root (see load_config)."""
    paths = cfg.get("paths")
    if not isinstance(paths, dict):
        return
    new_root = str(paths.get("root", "")).rstrip("/\\")
    if not orig_root or new_root == orig_root:
        return
    import os
    for key, val in list(paths.items()):
        if key == "root" or f"paths.{key}" in explicit_path_keys:
            continue
        if not isinstance(val, str):
            continue
        norm = val.rstrip("/\\")
        if norm == orig_root or norm.startswith(orig_root + "/") or norm.startswith(orig_root + "\\"):
            rel = os.path.relpath(norm, orig_root)
            paths[key] = new_root if rel == "." else os.path.join(new_root, rel).replace("\\", "/")
