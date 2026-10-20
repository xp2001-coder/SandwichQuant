"""Resolve locally supplied quantization backends without downloading code."""

from __future__ import annotations

import os
from pathlib import Path


def backend_root(name: str, project_root: Path | None = None) -> Path:
    settings = {"rescomp": ("SQ_RESCOMP_ROOT", "ResComp"),
                "qep": ("SQ_QEP_ROOT", "qep")}
    if name not in settings:
        raise ValueError(f"Unknown external backend: {name}")
    variable, directory = settings[name]
    project = project_root or Path(__file__).resolve().parents[1]
    configured = os.environ.get(variable, "").strip()
    root = Path(configured).expanduser() if configured else project / "third_party" / directory
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"External {name} backend not found. Set {variable} to a compatible "
            "local checkout; see docs/backends.md. No backend is downloaded automatically."
        )
    return root
