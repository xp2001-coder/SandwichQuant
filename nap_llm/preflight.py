from __future__ import annotations

import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch

from .common import Job, atomic_json


def inspect_model_path(path: Path) -> dict:
    return {
        "exists": path.is_dir(),
        "config_json": (path / "config.json").is_file(),
        "hf_safetensors": bool(list(path.glob("*.safetensors"))),
        "hf_pytorch_bin": bool(list(path.glob("pytorch_model*.bin"))),
        "meta_consolidated": bool(list(path.glob("consolidated.*.pth"))),
        "tokenizer": any((path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")),
    }


def run_preflight(job: Job, data_root: Path, output: Path) -> dict:
    model = inspect_model_path(Path(job.model_path))
    errors: list[str] = []
    if not model["exists"]:
        errors.append(f"model directory not found: {job.model_path}")
    if model["meta_consolidated"] and not (model["hf_safetensors"] or model["hf_pytorch_bin"]):
        errors.append("Llama checkpoint is Meta-original format; convert it to a Hugging Face checkpoint first")
    required = [
        data_root / "raw/c4/en/c4-train.00000-of-01024.json.gz",
        data_root / "raw/c4/en/c4-validation.00000-of-00008.json.gz",
        data_root / "raw/c4_parquet/train.parquet",
        data_root / "raw/c4_parquet/validation.parquet",
        data_root / "raw/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        data_root / "hf_home",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"missing data/cache path: {path}")
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is false; a compatible accelerator runtime is required")
    packages = {}
    for name in ("torch", "transformers", "datasets", "lm_eval", "gptqmodel", "safetensors"):
        try:
            packages[name] = importlib.metadata.version(name.replace("_", "-"))
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    if os.environ.get("NAP_QUANT_BACKEND", "official").lower() == "gptqmodel" and job.method in {"gptq", "awq", "gptaq"} and packages["gptqmodel"] is None:
        errors.append("gptqmodel is required for GPTQ/AWQ/GPTAQ jobs")
    if job.method in {"awq", "qep_gptq", "qep_awq"}:
        for package in ("primefac", "scipy", "portalocker"):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = None
                errors.append(f"{package} is required by the external QEP/AWQ backend")
    result = {
        "ok": not errors,
        "errors": errors,
        "job": job.job_id,
        "model": model,
        "packages": packages,
        "python": platform.python_version(),
        "torch_cuda": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2) if torch.cuda.is_available() else None,
        "environment": {key: os.environ.get(key) for key in ("HF_HOME", "HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "CUDA_VISIBLE_DEVICES")},
    }
    atomic_json(output, result)
    if errors:
        raise RuntimeError("Preflight failed:\n- " + "\n- ".join(errors))
    return result
