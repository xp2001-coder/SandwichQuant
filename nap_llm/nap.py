from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import atomic_json, log, seed_everything
from .modeling import (
    load_dense_model,
    load_norm_state,
    load_quant_model,
    materialize_quant_linears,
    save_norm_state,
    select_norm_parameters,
    select_relative_norm_parameters,
    unload,
)


def save_trainer_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def train_nap(
    model_path: str,
    artifact: Path,
    output_dir: Path,
    blocks: list[torch.Tensor],
    seed: int,
    max_steps: int = 200,
    lr: float = 5e-4,
    temperature: float = 1.0,
    ce_weight: float = 1.0,
    kd_weight: float = 1.0,
    grad_clip: float = 1.0,
    save_every: int = 10,
    deadline_end: float | None = None,
    parameterization: str = "direct",
    max_log_scale: float = 0.1,
) -> dict:
    seed_everything(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "norm_latest.safetensors"
    trainer_state = output_dir / "trainer_state.pt"

    student, wrapper = load_quant_model(artifact, model_path)
    artifact_kind = (artifact / "artifact_kind.txt").read_text(encoding="utf-8").strip()
    if artifact_kind == "gptqmodel":
        count = materialize_quant_linears(student)
        log(f"Materialized {count} canonical dequantized linears for NAP backward")
    teacher = load_dense_model(model_path)
    teacher.eval()
    student.train()
    if hasattr(student, "gradient_checkpointing_enable"):
        try:
            student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            student.gradient_checkpointing_enable()
    if parameterization == "relative":
        trainable, names = select_relative_norm_parameters(student, max_log_scale=max_log_scale)
    elif parameterization == "direct":
        trainable, names = select_norm_parameters(student)
    else:
        raise ValueError(f"Unknown NAP parameterization: {parameterization}")
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    start_step = 0
    history: list[dict] = []
    if checkpoint.exists() and trainer_state.exists():
        state = torch.load(trainer_state, map_location="cpu", weights_only=False)
        if parameterization == "relative":
            saved_parameters = state.get("parameter_state")
            if not isinstance(saved_parameters, dict):
                raise RuntimeError("Relative NAP checkpoint has no parameter_state; use a new job_id")
            for name, parameter in zip(names, trainable):
                parameter.data.copy_(saved_parameters[name].to(device=parameter.device, dtype=parameter.dtype))
        else:
            load_norm_state(student, checkpoint)
        optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(trainable[0].device)
        start_step = int(state["step"])
        history = list(state.get("history", []))
        log(f"Resuming NAP from step {start_step}")

    for step in range(start_step, max_steps):
        if deadline_end is not None and time.time() > deadline_end - 8 * 60:
            log("NAP reached the reservation safety margin; checkpointing")
            break
        input_ids = blocks[step % len(blocks)].unsqueeze(0).to("cuda")
        labels = input_ids.clone()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=next(teacher.parameters()).dtype):
            teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits
        with torch.autocast(device_type="cuda", dtype=next(student.parameters()).dtype):
            student_logits = student(input_ids=input_ids, use_cache=False).logits
            shift_student = student_logits[:, :-1].float()
            shift_teacher = teacher_logits[:, :-1].float()
            shift_labels = labels[:, 1:]
            ce = F.cross_entropy(shift_student.reshape(-1, shift_student.size(-1)), shift_labels.reshape(-1))
            kd = F.kl_div(
                F.log_softmax(shift_student / temperature, dim=-1),
                F.softmax(shift_teacher / temperature, dim=-1),
                reduction="none",
            ).sum(dim=-1).mean() * (temperature * temperature)
            loss = ce_weight * ce + kd_weight * kd
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"NAP produced a non-finite loss at step {step + 1}: "
                f"loss={float(loss.detach())}, ce={float(ce.detach())}, kd={float(kd.detach())}"
            )
        try:
            loss.backward()
        except RuntimeError as exc:
            if "backward" in str(exc).lower() or "grad" in str(exc).lower():
                raise RuntimeError(
                    "The selected packed quantization kernel has no input-gradient path. "
                    "Load this artifact with the GPTQ_TORCH reference backend or use an official dense fake-quant state dict."
                ) from exc
            raise
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(f"NAP produced a non-finite gradient norm at step {step + 1}")
        optimizer.step()
        if parameterization == "relative":
            for parameter in trainable:
                parameter.data.clamp_(-max_log_scale, max_log_scale)
        if any(not torch.isfinite(parameter).all() for parameter in trainable):
            raise FloatingPointError(f"NAP produced non-finite normalization parameters at step {step + 1}")
        row = {"step": step + 1, "loss": float(loss.detach()), "ce": float(ce.detach()), "kd": float(kd.detach()), "grad_norm": float(grad_norm)}
        if parameterization == "relative":
            row["max_abs_log_scale"] = max(float(parameter.detach().abs().max()) for parameter in trainable)
        history.append(row)
        if (step + 1) % save_every == 0 or step + 1 == max_steps:
            save_norm_state(student, checkpoint)
            save_trainer_state(
                trainer_state,
                {
                    "step": step + 1,
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    "parameterization": parameterization,
                    "parameter_state": {name: parameter.detach().cpu() for name, parameter in zip(names, trainable)},
                },
            )
            atomic_json(output_dir / "history.json", history)
            log(f"NAP step {step + 1}/{max_steps}: loss={row['loss']:.5f}")

    final_step = len(history)
    completed = final_step >= max_steps
    save_norm_state(student, checkpoint)
    save_trainer_state(
        trainer_state,
        {
            "step": final_step,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "parameterization": parameterization,
            "parameter_state": {name: parameter.detach().cpu() for name, parameter in zip(names, trainable)},
        },
    )
    summary = {
        "completed": completed,
        "steps": final_step,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "parameter_names": names,
        "parameterization": parameterization,
        "lr": lr,
        "max_steps": max_steps,
        "max_log_scale": max_log_scale if parameterization == "relative" else None,
    }
    atomic_json(output_dir / "summary.json", summary)
    unload(teacher, student, wrapper)
    return summary
