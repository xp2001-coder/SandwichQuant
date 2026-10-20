from __future__ import annotations

import inspect
import json
import os
import subprocess
from pathlib import Path

import torch
from safetensors.torch import save_file

from .common import Job, atomic_json, log, supported_kwargs
from .data import ensure_author_c4_calibration_tokens
from .modeling import load_dense_model, load_tokenizer, unload


def group_symmetric_quantize(weight: torch.Tensor, bits: int, group_size: int) -> torch.Tensor:
    if bits not in {2, 3, 4, 8}:
        raise ValueError(f"Unsupported bit width: {bits}")
    original_shape = weight.shape
    matrix = weight.float().reshape(-1, original_shape[-1])
    pad = (-matrix.shape[1]) % group_size
    if pad:
        matrix = torch.nn.functional.pad(matrix, (0, pad))
    groups = matrix.reshape(matrix.shape[0], -1, group_size)
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quant = torch.round(groups / scale).clamp(qmin, qmax)
    dequant = (quant * scale).reshape(matrix.shape[0], -1)
    if pad:
        dequant = dequant[:, :-pad]
    return dequant.reshape(original_shape).to(weight.dtype)


def quantize_rtn(job: Job, artifact: Path, device: str = "cuda") -> None:
    model = load_dense_model(job.model_path, device=device)
    tensors: dict[str, torch.Tensor] = {}
    total = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, torch.nn.Linear) or name.endswith("lm_head"):
                continue
            module.weight.data.copy_(group_symmetric_quantize(module.weight.data, job.bits, job.group_size))
            tensors[f"{name}.weight"] = module.weight.detach().cpu().contiguous()
            total += module.weight.numel()
    artifact.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(artifact / "weights.safetensors"))
    (artifact / "artifact_kind.txt").write_text("dense_delta\n", encoding="utf-8")
    atomic_json(artifact / "quantization.json", {"method": "rtn", "bits": job.bits, "group_size": job.group_size, "quantized_parameters": total})
    unload(model)


def _import_gptqmodel_symbols():
    import gptqmodel

    symbols = {name: getattr(gptqmodel, name, None) for name in ("GPTQModel", "GPTQConfig", "AWQConfig", "RTNConfig", "QuantizeConfig", "GPTAQConfig", "BACKEND")}
    if symbols["GPTQModel"] is None:
        raise RuntimeError("Installed gptqmodel does not expose GPTQModel")
    return symbols


def quantize_gptqmodel(job: Job, artifact: Path, calibration: list[str]) -> None:
    symbols = _import_gptqmodel_symbols()
    config_cls = symbols["AWQConfig"] if job.method == "awq" else symbols["GPTQConfig"] or symbols["QuantizeConfig"]
    if config_cls is None:
        raise RuntimeError("This gptqmodel version lacks the requested quantization config")
    kwargs = {"bits": job.bits, "group_size": job.group_size, "sym": True, "desc_act": True, "device": "auto"}
    if job.method == "gptaq":
        gptaq_cls = symbols["GPTAQConfig"]
        if gptaq_cls is None:
            raise RuntimeError("This gptqmodel version does not provide GPTAQConfig")
        kwargs["gptaq"] = gptaq_cls(alpha=0.25, device="auto")
    quant_config = config_cls(**supported_kwargs(config_cls, kwargs))
    wrapper = symbols["GPTQModel"].load(job.model_path, quant_config)
    quantize_kwargs = {"batch_size": 1, "calibration_dataset_concat_size": 2048, "auto_gc": True, "buffered_fwd": True}
    wrapper.quantize(calibration, **supported_kwargs(wrapper.quantize, quantize_kwargs))
    packed = artifact / "packed"
    packed.mkdir(parents=True, exist_ok=True)
    wrapper.save(str(packed))
    (artifact / "artifact_kind.txt").write_text("gptqmodel\n", encoding="utf-8")
    atomic_json(artifact / "quantization.json", {"method": job.method, "bits": job.bits, "group_size": job.group_size, "backend": "gptqmodel"})
    unload(wrapper)


def import_official_artifact(job: Job, artifact: Path) -> None:
    source = Path(job.external_artifact)
    if not source.is_file():
        raise FileNotFoundError(
            f"{job.method} requires the official state dict at {source}. "
            "Run scripts/run_official_qep.sh or scripts/run_official_rescomp.sh first."
        )
    artifact.mkdir(parents=True, exist_ok=True)
    destination = artifact / "model.pt"
    if destination.resolve() != source.resolve():
        import shutil

        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    (artifact / "artifact_kind.txt").write_text("official_state_dict\n", encoding="utf-8")
    atomic_json(artifact / "quantization.json", {"method": job.method, "bits": job.bits, "group_size": job.group_size, "source": str(source)})


def quantize_official(job: Job, artifact: Path, data_root: Path, calibration: list[str]) -> None:
    project_root = Path(__file__).resolve().parents[1]
    if job.external_artifact:
        state_path = Path(job.external_artifact)
    else:
        state_path = artifact / "model.pt"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    if not state_path.exists():
        tokenizer = load_tokenizer(job.model_path, use_fast=False)
        calibration_path = ensure_author_c4_calibration_tokens(
            tokenizer, data_root, job.model_key, job.seed, count=128, seq_len=2048
        )
        del tokenizer
        model_name = job.model_key.lower()
        if "qwen" in model_name and job.method in {"qep_gptq", "qep_awq"}:
            raise RuntimeError(
                "The QEP release exposes Qwen2 pieces but its main sequential GPTQ path is not verified for Qwen3. "
                "For Qwen3 use the validated ResComp sequential GPTQ/GPTAQ path or AWQ."
            )
        if job.method in {"awq", "qep_gptq", "qep_awq"}:
            script = project_root / "scripts/run_official_qep.sh"
            method = job.method
        elif job.method in {"gptq", "gptaq", "rescomp", "rescomp_gptaq"}:
            script = project_root / "scripts/run_official_rescomp.sh"
            method = job.method
        else:
            raise ValueError(job.method)
        environment = os.environ.copy()
        environment["DATA_ROOT"] = str(data_root)
        environment["NAP_CALIBRATION_TOKENS"] = str(calibration_path)
        command = ["bash", str(script), job.model_path, str(state_path), method, str(job.bits), str(job.seed)]
        log("Official quantization command: " + " ".join(command))
        subprocess.run(command, check=True, env=environment)
    patched_job = Job(**{**job.__dict__, "external_artifact": str(state_path)})
    import_official_artifact(patched_job, artifact)


def run_quantization(job: Job, artifact: Path, calibration: list[str], data_root: Path) -> None:
    log(f"Quantizing {job.job_id}: {job.method} W{job.bits} g{job.group_size}")
    if job.method == "rtn":
        quantize_rtn(job, artifact)
    elif job.method in {"gptq", "awq", "gptaq", "qep_gptq", "qep_awq", "rescomp", "rescomp_gptaq"}:
        backend = os.environ.get("NAP_QUANT_BACKEND", "official").lower()
        if backend == "gptqmodel" and job.method in {"gptq", "awq", "gptaq"}:
            quantize_gptqmodel(job, artifact, calibration)
        else:
            quantize_official(job, artifact, data_root, calibration)
    elif job.method == "fp16":
        artifact.mkdir(parents=True, exist_ok=True)
        (artifact / "artifact_kind.txt").write_text("fp16\n", encoding="utf-8")
    else:
        raise ValueError(f"Unsupported method: {job.method}")
