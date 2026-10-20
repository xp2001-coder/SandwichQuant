from __future__ import annotations

import gc
import sys
from pathlib import Path
from typing import Any

import torch
from torch.nn.utils import parametrize
from safetensors.torch import load_file, save_file

from .common import log
from .backends import backend_root


def resolve_dtype(name: str = "auto") -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name != "auto":
        raise ValueError(name)
    return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16


def load_tokenizer(model_path: str, use_fast: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True, use_fast=use_fast)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_dense_model(model_path: str, device: str = "cuda", dtype: str = "auto"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=resolve_dtype(dtype),
        # Each worker owns one complete device; no device-map dispatch is used.
        low_cpu_mem_usage=False,
    )
    model.config.use_cache = False
    return model.to(device)


def load_quant_model(artifact: Path, model_path: str, device: str = "cuda"):
    kind_file = artifact / "artifact_kind.txt"
    kind = kind_file.read_text(encoding="utf-8").strip() if kind_file.exists() else "dense_delta"
    if kind == "gptqmodel":
        from gptqmodel import BACKEND, GPTQModel

        backend = getattr(BACKEND, "GPTQ_TORCH", None)
        wrapper = GPTQModel.load(str(artifact / "packed"), backend=backend) if backend is not None else GPTQModel.load(str(artifact / "packed"))
        model = getattr(wrapper, "model", wrapper)
        model.config.use_cache = False
        return model, wrapper
    if kind == "fp16":
        return load_dense_model(model_path, device=device), None
    if kind in {"dense_delta", "official_state_dict"}:
        model = load_dense_model(model_path, device=device)
        if kind == "dense_delta":
            state = load_file(str(artifact / "weights.safetensors"), device="cpu")
        else:
            # ResComp/GPTQ/GPTAQ checkpoints include author-code quantizer
            # objects alongside the dense fake-quantized model state.  Their
            # pickle module is named simply ``quant_utils`` because the
            # official entry point runs from fake_quant/.  Make that source
            # directory importable when resuming evaluation from our project
            # root; the tensors are still loaded on CPU below.
            rescomp_source = backend_root("rescomp") / "fake_quant"
            source_text = str(rescomp_source)
            if rescomp_source.is_dir() and source_text not in sys.path:
                sys.path.insert(0, source_text)
            payload = torch.load(artifact / "model.pt", map_location="cpu", weights_only=False)
            source_state = payload.get("model", payload) if isinstance(payload, dict) else payload
            target_state = model.state_dict()
            state = {}
            ignored = []
            for source_name, tensor in source_state.items():
                candidates = [source_name, source_name.replace(".module.", ".")]
                target_name = next((name for name in candidates if name in target_state and target_state[name].shape == tensor.shape), None)
                if target_name is None:
                    ignored.append(source_name)
                    continue
                state[target_name] = tensor
            if not state:
                raise RuntimeError("No compatible tensors found in the official fake-quant checkpoint")
            log(f"Mapped {len(state)} official tensors; ignored {len(ignored)} wrapper/qparam tensors")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected keys after checkpoint mapping: {unexpected[:8]}")
        log(f"Loaded quantized state; {len(missing)} original tensors were unchanged")
        return model, None
    raise ValueError(f"Unknown artifact kind: {kind}")


def unload(*objects: Any) -> None:
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_norm_module(module: torch.nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return "rmsnorm" in class_name or isinstance(module, torch.nn.LayerNorm)


def select_norm_parameters(model: torch.nn.Module) -> tuple[list[torch.nn.Parameter], list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[torch.nn.Parameter] = []
    names: list[str] = []
    for module_name, module in model.named_modules():
        if not is_norm_module(module):
            continue
        weight = getattr(module, "weight", None)
        if isinstance(weight, torch.nn.Parameter):
            weight.requires_grad_(True)
            selected.append(weight)
            names.append(f"{module_name}.weight")
    if not selected:
        raise RuntimeError("No RMSNorm/LayerNorm affine weights found")
    return selected, names


class RelativeNormScale(torch.nn.Module):
    """Scale an existing norm weight through a bounded log-space delta.

    AWQ folds activation-aware channel scales into adjacent normalization
    weights.  Optimizing those folded weights additively makes one global
    learning rate depend on their absolute magnitudes.  A log-space delta is
    dimensionless, preserves the AWQ scale pattern at initialization, and
    bounds the maximum relative change made by NAP.
    """

    def __init__(self, weight: torch.Tensor, max_log_scale: float) -> None:
        super().__init__()
        self.register_buffer("base", weight.detach().clone())
        self.delta = torch.nn.Parameter(torch.zeros_like(weight, dtype=torch.float32))
        self.max_log_scale = float(max_log_scale)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        del original
        scale = torch.exp(self.delta.clamp(-self.max_log_scale, self.max_log_scale))
        return (self.base.float() * scale).to(dtype=self.base.dtype)


def select_relative_norm_parameters(
    model: torch.nn.Module,
    max_log_scale: float = 0.1,
) -> tuple[list[torch.nn.Parameter], list[str]]:
    """Attach multiplicative NAP deltas to every affine norm weight."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    selected: list[torch.nn.Parameter] = []
    names: list[str] = []
    for module_name, module in list(model.named_modules()):
        if not is_norm_module(module):
            continue
        weight = getattr(module, "weight", None)
        if not isinstance(weight, torch.nn.Parameter):
            continue
        parametrize.register_parametrization(
            module,
            "weight",
            RelativeNormScale(weight, max_log_scale=max_log_scale),
        )
        module.parametrizations.weight.original.requires_grad_(False)
        delta = module.parametrizations.weight[0].delta
        selected.append(delta)
        names.append(f"{module_name}.log_scale_delta")
    if not selected:
        raise RuntimeError("No RMSNorm/LayerNorm affine weights found")
    return selected, names


def norm_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for module_name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if is_norm_module(module) and torch.is_tensor(weight):
            state[f"{module_name}.weight"] = weight.detach().cpu().contiguous()
    return state


def save_norm_state(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(norm_state(model), str(temporary))
    temporary.replace(path)


def load_norm_state(model: torch.nn.Module, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    modules = dict(model.named_modules())
    for full_name, tensor in state.items():
        module_name, parameter_name = full_name.rsplit(".", 1)
        module = modules[module_name]
        parameter = getattr(module, parameter_name)
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


def materialize_quant_linears(model: torch.nn.Module) -> int:
    """Replace packed GPTQModel linears with dense frozen linears for NAP backward.

    Dequantization comes from the installed canonical backend. No hand-written
    unpacking logic is used, which avoids silent GPTQ/AWQ layout mistakes.
    """
    replacements: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if not name or isinstance(module, torch.nn.Linear):
            continue
        dequantize = getattr(module, "dequantize_weight", None)
        if not callable(dequantize):
            continue
        weight = dequantize()
        if weight.ndim != 2:
            raise RuntimeError(f"{name}.dequantize_weight returned shape {tuple(weight.shape)}")
        in_features = int(getattr(module, "in_features", weight.shape[0]))
        out_features = int(getattr(module, "out_features", weight.shape[1]))
        if tuple(weight.shape) == (in_features, out_features):
            weight = weight.T
        elif tuple(weight.shape) != (out_features, in_features):
            raise RuntimeError(
                f"Cannot infer orientation for {name}: weight={tuple(weight.shape)}, "
                f"in={in_features}, out={out_features}"
            )
        bias = getattr(module, "bias", None)
        dense = torch.nn.Linear(
            in_features,
            out_features,
            bias=bias is not None,
            device=weight.device,
            dtype=weight.dtype,
        )
        dense.weight.data.copy_(weight)
        dense.weight.requires_grad_(False)
        if bias is not None:
            dense.bias.data.copy_(bias.to(device=dense.bias.device, dtype=dense.bias.dtype))
            dense.bias.requires_grad_(False)
        replacements.append((name, dense))
    for name, dense in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child_name, dense)
    if not replacements:
        raise RuntimeError(
            "Packed quantized linears expose no canonical dequantize_weight() method. "
            "Use the official fake-quant state-dict adapter for NAP."
        )
    return len(replacements)
