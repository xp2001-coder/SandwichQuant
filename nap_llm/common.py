from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import os
import random
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


TASKS = ("piqa", "arc_easy", "arc_challenge", "hellaswag", "winogrande", "boolq")
PPL_DATASETS = ("wikitext2", "c4")


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name, suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_hash(payload: Any) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def supported_kwargs(callable_obj: Any, values: dict[str, Any]) -> dict[str, Any]:
    """Keep only arguments accepted by the installed backend version."""
    signature = inspect.signature(callable_obj)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return values
    return {key: value for key, value in values.items() if key in signature.parameters}


@dataclass(frozen=True)
class Job:
    job_id: str
    priority: int
    lane: str
    enabled: bool
    model_key: str
    model_path: str
    method: str
    bits: int
    group_size: int
    seed: int
    run_nap: bool
    external_artifact: str = ""

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Job":
        def truth(value: str) -> bool:
            return str(value).strip().lower() in {"1", "true", "yes", "y"}

        return cls(
            job_id=row["job_id"].strip(),
            priority=int(row.get("priority", 999)),
            lane=row.get("lane", "main").strip(),
            enabled=truth(row.get("enabled", "1")),
            model_key=row["model_key"].strip(),
            model_path=os.path.expandvars(row["model_path"].strip()),
            method=row["method"].strip().lower(),
            bits=int(row["bits"]),
            group_size=int(row.get("group_size", 128)),
            seed=int(row.get("seed", 0)),
            run_nap=truth(row.get("run_nap", "1")),
            external_artifact=os.path.expandvars(row.get("external_artifact", "").strip()),
        )

    def fingerprint(self) -> str:
        return stable_hash(asdict(self))


@dataclass
class Deadline:
    end_time: float

    @classmethod
    def after_minutes(cls, minutes: float) -> "Deadline":
        return cls(time.time() + minutes * 60.0)

    def remaining_minutes(self) -> float:
        return max(0.0, (self.end_time - time.time()) / 60.0)

    def can_start(self, reserve_minutes: float = 12.0) -> bool:
        return self.remaining_minutes() > reserve_minutes


@contextlib.contextmanager
def stage_record(run_dir: Path, name: str, metadata: dict[str, Any] | None = None) -> Iterator[None]:
    state_dir = run_dir / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    running = state_dir / f"{name}.running.json"
    done = state_dir / f"{name}.done.json"
    started = time.time()
    atomic_json(running, {"stage": name, "started": started, "metadata": metadata or {}})
    try:
        yield
    except BaseException as exc:
        atomic_json(
            state_dir / f"{name}.failed.json",
            {"stage": name, "started": started, "failed": time.time(), "error": repr(exc)},
        )
        raise
    else:
        atomic_json(done, {"stage": name, "started": started, "finished": time.time(), "metadata": metadata or {}})
    finally:
        running.unlink(missing_ok=True)


def stage_done(run_dir: Path, name: str) -> bool:
    return (run_dir / "state" / f"{name}.done.json").exists()
