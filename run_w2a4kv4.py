#!/usr/bin/env python3
"""Resumable QuaRot W2A4KV4 evaluation with paired NAP adaptation.

This runner intentionally stays separate from the W3A16 pipeline.  It keeps
activation and KV-cache fake quantization, online Hadamard transforms, and the
post-rotation affine response parameters alive for quantization, NAP, and all
evaluations.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file

from nap_llm.common import TASKS, atomic_json, log, seed_everything
from nap_llm.backends import backend_root
from nap_llm.data import build_ppl_ids
from nap_llm.modeling import load_dense_model, load_tokenizer, unload


METRIC_BY_TASK = {
    "piqa": "acc_norm,none",
    "arc_easy": "acc_norm,none",
    "arc_challenge": "acc_norm,none",
    "hellaswag": "acc_norm,none",
    "winogrande": "acc,none",
    "boolq": "acc,none",
}
PROTOCOL = "nap-quarot-w2a4kv4-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--model-key",
        required=True,
        choices=("llama2_7b", "llama3_8b", "qwen3_8b"),
    )
    parser.add_argument("--method", required=True, choices=("gptq", "gptaq", "rescomp_gptaq"))
    parser.add_argument(
        "--rotation",
        choices=("quarot", "spinquant"),
        default="quarot",
        help="Fixed QuaRot Hadamard rotation or a learned SpinQuant rotation.",
    )
    parser.add_argument(
        "--optimized-rotation-path",
        type=Path,
        help="Official SpinQuant R.bin (required when --rotation spinquant).",
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--minutes", type=float, default=350.0)
    parser.add_argument("--quant-reserve-minutes", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--calibration-seq-len", type=int, default=2048)
    parser.add_argument("--nap-steps", type=int, default=200)
    parser.add_argument("--nap-seq-len", type=int, default=256)
    parser.add_argument("--nap-lr", type=float, default=5e-4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--eval-start-margin", type=float, default=5.0)
    parser.add_argument("--nap-checkpoint-margin", type=float, default=5.0)
    parser.add_argument("--smoke-layers", type=int, default=0)
    parser.add_argument(
        "--baseline-ppl-only",
        action="store_true",
        help="Evaluate only WikiText-2/C4 PPL for the frozen baseline; NAP still runs all tasks.",
    )
    parser.add_argument(
        "--skip-tasks",
        action="store_true",
        help="Skip downstream tasks for both variants (intended only for smoke tests).",
    )
    return parser.parse_args()


def remaining_minutes(deadline: float) -> float:
    return max(0.0, (deadline - time.time()) / 60.0)


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    return str(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def install_official_imports(project_root: Path) -> Path:
    source = backend_root("rescomp", project_root) / "fake_quant"
    if not source.is_dir():
        raise FileNotFoundError(f"ResComp source is missing: {source}")
    source_text = str(source)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    try:
        import fast_hadamard_transform  # noqa: F401
        log("Using the native fast_hadamard_transform module")
    except ImportError:
        from nap_llm import torch_hadamard_fallback

        sys.modules["fast_hadamard_transform"] = torch_hadamard_fallback
        log("Native fast_hadamard_transform is absent; using the PyTorch fallback")
    return source


def official_modules(project_root: Path):
    install_official_imports(project_root)
    import data_utils
    import gptaq_utils
    import gptaq_utils_r
    import gptq_utils
    import hadamard_utils
    import model_utils
    import quant_utils
    import rotation_utils

    modules = SimpleNamespace(
        data=data_utils,
        gptaq=gptaq_utils,
        rescomp=gptaq_utils_r,
        gptq=gptq_utils,
        hadamard=hadamard_utils,
        model=model_utils,
        quant=quant_utils,
        rotation=rotation_utils,
    )
    patch_gqa_safe_qk_wrapper(modules)
    patch_qwen3_support(modules)
    return modules


def patch_qwen3_support(modules) -> None:
    """Extend the released Llama-only QuaRot path to dense Qwen3 models.

    Qwen3 is structurally close to Llama, but it has two additional per-head
    Q/K RMSNorms and its decoder reads ``attention_type`` from every layer.
    Treating every Qwen3 RMSNorm as a residual-path norm would silently change
    the attention computation. We therefore fuse and expose an affine only
    for the input, post-attention, and final residual-path norms, while leaving
    ``q_norm`` and ``k_norm`` untouched.
    """

    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    except (ImportError, AttributeError):
        return

    original_get_model = modules.model.get_model
    original_model_type_extractor = modules.model.model_type_extractor
    original_get_model_type = modules.model.get_model_type
    original_get_rope_function_name = modules.model.get_rope_function_name
    original_get_layers = modules.model.get_layers
    original_get_pre_head_layernorm = modules.model.get_pre_head_layernorm
    original_fuse_layer_norms = modules.rotation.fuse_layer_norms

    def is_qwen3(model) -> bool:
        return isinstance(model, Qwen3ForCausalLM)

    def get_model(model_name, hf_token=None):
        if "qwen3" not in str(model_name).lower():
            return original_get_model(model_name, hf_token)
        model = Qwen3ForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            use_auth_token=hf_token,
            low_cpu_mem_usage=True,
        )
        model.seqlen = 2048
        log(f"---> Loading {model_name} as dense Qwen3 with seq_len=2048")
        return model

    def model_type_extractor(model):
        if is_qwen3(model):
            return modules.model.LLAMA_MODEL
        return original_model_type_extractor(model)

    def get_model_type(model):
        if is_qwen3(model):
            return modules.model.LLAMA_MODEL
        return original_get_model_type(model)

    def get_rope_function_name(model):
        if is_qwen3(model):
            return "apply_rotary_pos_emb"
        return original_get_rope_function_name(model)

    def get_layers(model):
        if is_qwen3(model):
            return model.model.layers
        return original_get_layers(model)

    def get_pre_head_layernorm(model, model_type):
        if is_qwen3(model):
            return model.model.norm
        return original_get_pre_head_layernorm(model, model_type)

    def residual_affine(norm):
        eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-6)))
        return modules.model.RMSN(int(norm.weight.numel()), eps=eps)

    def fuse_layer_norms(model):
        if not is_qwen3(model):
            return original_fuse_layer_norms(model)

        embedding = model.model.embed_tokens
        embedding_weight = embedding.weight.data.double()
        embedding.weight.data = (
            embedding_weight - embedding_weight.mean(dim=-1, keepdim=True)
        ).to(embedding.weight.data.dtype)

        for layer in model.model.layers:
            modules.rotation.fuse_ln_linear(
                layer.input_layernorm,
                [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj],
            )
            modules.rotation.fuse_ln_linear(
                layer.post_attention_layernorm,
                [layer.mlp.up_proj, layer.mlp.gate_proj],
            )
            layer.input_layernorm = residual_affine(layer.input_layernorm)
            layer.post_attention_layernorm = residual_affine(layer.post_attention_layernorm)

        modules.rotation.fuse_ln_linear(model.model.norm, [model.lm_head])
        model.model.norm = residual_affine(model.model.norm)
        log("Qwen3 QuaRot compatibility: fused residual RMSNorms; preserved Q/K RMSNorms")

    modules.model.get_model = get_model
    modules.model.model_type_extractor = model_type_extractor
    modules.model.get_model_type = get_model_type
    modules.model.get_rope_function_name = get_rope_function_name
    modules.model.get_layers = get_layers
    modules.model.get_pre_head_layernorm = get_pre_head_layernorm
    modules.rotation.fuse_layer_norms = fuse_layer_norms


@contextlib.contextmanager
def qwen3_catcher_compatibility(enabled: bool):
    """Forward Qwen3 layer metadata through GPTAQ's temporary Catcher."""

    if not enabled:
        yield
        return
    original_getattr = torch.nn.Module.__getattr__

    def compatible_getattr(module, name):
        if name == "attention_type":
            wrapped = module.__dict__.get("_modules", {}).get("module")
            if wrapped is not None:
                return getattr(wrapped, name)
        return original_getattr(module, name)

    torch.nn.Module.__getattr__ = compatible_getattr
    try:
        yield
    finally:
        torch.nn.Module.__getattr__ = original_getattr


def patch_gqa_safe_qk_wrapper(modules) -> None:
    """Fix upstream KV quantization for grouped-query attention.

    The released wrapper reshapes K with ``config.hidden_size`` and therefore
    assumes num_key_value_heads == num_attention_heads.  Llama-3-8B uses GQA.
    Deriving the width from the runtime K tensor preserves both Llama-2 MHA and
    Llama-3 GQA without changing their attention semantics.
    """

    from fast_hadamard_transform import hadamard_transform

    class GQASafeQKRotationWrapper(torch.nn.Module):
        def __init__(self, func, config, *unused, **kwargs) -> None:
            super().__init__()
            self.func = func
            self.config = config
            self.k_groupsize = int(kwargs["k_groupsize"])
            self.k_quantizer = modules.quant.ActQuantizer()
            self.k_quantizer.configure(
                bits=int(kwargs["k_bits"]),
                groupsize=-1,
                sym=bool(kwargs["k_sym"]),
                clip_ratio=float(kwargs["k_clip_ratio"]),
            )

        def forward(self, *args, **kwargs):
            query, key = self.func(*args, **kwargs)
            query_dtype = query.dtype
            query = hadamard_transform(
                query.float(), scale=1 / math.sqrt(query.shape[-1])
            ).to(query_dtype)
            key = hadamard_transform(
                key.float(), scale=1 / math.sqrt(key.shape[-1])
            ).to(query_dtype)
            batch, kv_heads, seq_len, head_dim = key.shape
            if self.k_groupsize == -1:
                token_key = key.transpose(1, 2).reshape(-1, kv_heads * head_dim)
                self.k_quantizer.find_params(token_key)
                key = self.k_quantizer(token_key).reshape(
                    batch, seq_len, kv_heads, head_dim
                ).transpose(1, 2).to(query)
            elif self.k_groupsize == head_dim:
                per_head_key = key.reshape(-1, head_dim)
                self.k_quantizer.find_params(per_head_key)
                key = self.k_quantizer(per_head_key).reshape(
                    batch, kv_heads, seq_len, head_dim
                ).to(query)
            else:
                raise ValueError(
                    f"K groupsize must be -1 or head_dim={head_dim}, got {self.k_groupsize}"
                )
            self.k_quantizer.free()
            return query, key

    modules.rotation.QKRotationWrapper = GQASafeQKRotationWrapper


class AffineRMSN(torch.nn.Module):
    """QuaRot RMS normalization with an identity-initialized channel affine."""

    def __init__(self, mean_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.mean_dim = int(mean_dim)
        self.weight = torch.nn.Parameter(torch.ones(mean_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        normalized = x.float()
        variance = normalized.pow(2).sum(-1, keepdim=True) / self.mean_dim
        normalized = normalized * torch.rsqrt(variance + self.eps)
        return (normalized.to(input_dtype) * self.weight.to(dtype=input_dtype))


def quant_args(args: argparse.Namespace, run_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        model=args.model_path,
        seed=args.seed,
        hf_token=None,
        w_bits=2,
        w_groupsize=128,
        w_clip=True,
        w_asym=False,
        w_rtn=False,
        a_bits=4,
        a_groupsize=-1,
        a_asym=True,
        a_clip_ratio=0.9,
        v_bits=4,
        v_groupsize=-1,
        v_asym=True,
        v_clip_ratio=0.95,
        k_bits=4,
        k_groupsize=-1,
        k_asym=True,
        k_clip_ratio=0.95,
        k_pre_rope=False,
        rotate=True,
        rotate_mode="hadamard",
        rotation_seed=-1,
        fp32_had=False,
        enable_aq_calibration=True,
        int8_down_proj=False,
        cal_dataset="wikitext2",
        nsamples=args.calibration_samples,
        percdamp=0.01,
        act_order=True,
        static_groups=False,
        bsz=1,
        capture_layer_io=False,
        layer_idx=10,
        save_path=str(run_dir / "author_logs"),
    )


def add_activation_and_kv_quant(model, qargs: SimpleNamespace, modules) -> None:
    qlayers = modules.quant.find_qlayers(model, layers=[modules.quant.ActQuantWrapper])
    for name, layer in qlayers.items():
        input_bits = 16 if "lm_head" in name else qargs.a_bits
        layer.quantizer.configure(
            bits=input_bits,
            groupsize=qargs.a_groupsize,
            sym=not qargs.a_asym,
            clip_ratio=qargs.a_clip_ratio,
        )
        if "v_proj" in name:
            layer.out_quantizer.configure(
                bits=qargs.v_bits,
                groupsize=qargs.v_groupsize,
                sym=not qargs.v_asym,
                clip_ratio=qargs.v_clip_ratio,
            )
    rope_name = modules.model.get_rope_function_name(model)
    k_config = {
        "k_bits": qargs.k_bits,
        "k_groupsize": qargs.k_groupsize,
        "k_sym": not qargs.k_asym,
        "k_clip_ratio": qargs.k_clip_ratio,
    }
    for layer in modules.model.get_layers(model):
        modules.rotation.add_qk_rotation_wrapper_after_function_call_in_forward(
            layer.self_attn,
            rope_name,
            config=model.config,
            **k_config,
        )


def _spinquant_linear_right(linear, rotation: torch.Tensor) -> None:
    dtype = linear.weight.dtype
    weight = linear.weight.data.to(device="cuda", dtype=torch.float64)
    linear.weight.data = (weight @ rotation).to(device="cpu", dtype=dtype)


def _spinquant_linear_left_transpose(linear, rotation: torch.Tensor) -> None:
    dtype = linear.weight.dtype
    weight = linear.weight.data.to(device="cuda", dtype=torch.float64)
    linear.weight.data = (rotation.T @ weight).to(device="cpu", dtype=dtype)
    if linear.bias is not None:
        bias = linear.bias.data.to(device="cuda", dtype=torch.float64)
        linear.bias.data = (rotation.T @ bias).to(device="cpu", dtype=dtype)


def _spinquant_apply_r2(linear, rotation: torch.Tensor, *, output: bool) -> None:
    """Bake SpinQuant's learned per-head R2 into V/O projection weights."""

    dtype = linear.weight.dtype
    original = linear.weight.data.to(device="cuda", dtype=torch.float64)
    head_dim = rotation.shape[0]
    if output:
        transposed = original.T
        shape = transposed.shape
        transformed = transposed.reshape(-1, shape[-1] // head_dim, head_dim) @ rotation
        transformed = transformed.reshape(shape).T
    else:
        shape = original.shape
        transformed = original.reshape(-1, shape[-1] // head_dim, head_dim) @ rotation
        transformed = transformed.reshape(shape)
    linear.weight.data = transformed.to(device="cpu", dtype=dtype)


@torch.inference_mode()
def rotate_spinquant_model(model, rotation_path: Path, modules) -> None:
    """Apply the released SpinQuant R1/R2 transform to a fused Llama graph."""

    state = torch.load(rotation_path, map_location="cpu", weights_only=False)
    if "R1" not in state:
        raise KeyError(f"SpinQuant checkpoint has no R1 tensor: {rotation_path}")
    hidden_size = model.config.hidden_size
    r1_cpu = state["R1"]
    if tuple(r1_cpu.shape) != (hidden_size, hidden_size):
        raise ValueError(
            f"SpinQuant R1 shape {tuple(r1_cpu.shape)} does not match hidden size {hidden_size}"
        )
    r1 = r1_cpu.to(device="cuda", dtype=torch.float64)
    model_type = modules.model.model_type_extractor(model)

    for embedding in modules.model.get_embeddings(model, model_type=model_type):
        _spinquant_linear_right(embedding, r1)
    _spinquant_linear_right(modules.model.get_lm_head(model, model_type=model_type), r1)

    layers = modules.model.get_layers(model)
    head_dim = hidden_size // model.config.num_attention_heads
    for index, layer in enumerate(layers):
        key = f"model.layers.{index}.self_attn.R2"
        if key not in state:
            raise KeyError(f"SpinQuant checkpoint has no tensor {key}")
        r2 = state[key].to(device="cuda", dtype=torch.float64)
        if tuple(r2.shape) != (head_dim, head_dim):
            raise ValueError(
                f"SpinQuant {key} shape {tuple(r2.shape)} does not match head dim {head_dim}"
            )
        for linear in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
            _spinquant_linear_right(linear, r1)
        _spinquant_linear_left_transpose(layer.self_attn.o_proj, r1)
        for linear in (layer.mlp.up_proj, layer.mlp.gate_proj):
            _spinquant_linear_right(linear, r1)
        _spinquant_linear_left_transpose(layer.mlp.down_proj, r1)
        modules.hadamard.apply_exact_had_to_linear(
            layer.mlp.down_proj, had_dim=-1, output=False
        )
        _spinquant_apply_r2(layer.self_attn.v_proj, r2, output=True)
        _spinquant_apply_r2(layer.self_attn.o_proj, r2, output=False)
        del r2
        if (index + 1) % 4 == 0 or index + 1 == len(layers):
            log(f"Applied SpinQuant rotation {index + 1}/{len(layers)}")
    del r1, state
    torch.cuda.empty_cache()


def build_quarot_model(args: argparse.Namespace, run_dir: Path, modules):
    qargs = quant_args(args, run_dir)
    modules.model.RMSN = AffineRMSN
    model = modules.model.get_model(args.model_path, None)
    model.eval()
    model.config.use_cache = False
    modules.rotation.fuse_layer_norms(model)
    if args.rotation == "spinquant":
        rotate_spinquant_model(model, args.optimized_rotation_path, modules)
    else:
        modules.rotation.rotate_model(model, qargs)
    modules.quant.add_actquant(model)
    qlayers = modules.quant.find_qlayers(model)
    for name, layer in qlayers.items():
        if "down_proj" in name:
            had_k, k_value = modules.hadamard.get_hadK(model.config.intermediate_size)
            layer.online_full_had = True
            layer.had_K = had_k
            layer.K = k_value
            layer.fp32_had = qargs.fp32_had
        if args.rotation == "quarot" and "o_proj" in name:
            had_k, k_value = modules.hadamard.get_hadK(model.config.num_attention_heads)
            layer.online_partial_had = True
            layer.had_K = had_k
            layer.K = k_value
            layer.had_dim = model.config.hidden_size // model.config.num_attention_heads
            layer.fp32_had = qargs.fp32_had
    add_activation_and_kv_quant(model, qargs, modules)
    return model, qargs


def save_quant_checkpoint(model, path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    torch.save({"model": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)


def load_or_quantize(
    args: argparse.Namespace,
    run_dir: Path,
    modules,
    deadline: float,
):
    checkpoint = run_dir / "artifact" / "model.pt"
    model, qargs = build_quarot_model(args, run_dir, modules)
    if checkpoint.is_file():
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(payload["model"], strict=False)
        if unexpected:
            raise RuntimeError(f"Unexpected checkpoint keys: {unexpected[:8]}")
        log(f"Restored full W2A4KV4 graph; missing keys={len(missing)}")
        return model, qargs
    if remaining_minutes(deadline) <= args.quant_reserve_minutes:
        log(
            f"Not starting a new non-resumable quantization: remaining={remaining_minutes(deadline):.1f}m, "
            f"required={args.quant_reserve_minutes:.1f}m"
        )
        raise SystemExit(75)
    if args.smoke_layers > 0:
        os.environ["NAP_MAX_QUANT_LAYERS"] = str(args.smoke_layers)
    else:
        os.environ.pop("NAP_MAX_QUANT_LAYERS", None)
    trainloader = modules.data.get_loaders(
        "wikitext2",
        nsamples=args.calibration_samples,
        seed=args.seed,
        model=args.model_path,
        seqlen=args.calibration_seq_len,
        hf_token=None,
        eval_mode=False,
    )
    rotation_name = "SpinQuant" if args.rotation == "spinquant" else "QuaRot"
    log(f"Quantizing {args.model_key} with {rotation_name}+{args.method}, W2A4KV4")
    with qwen3_catcher_compatibility(args.model_key == "qwen3_8b"):
        if args.method == "gptq":
            modules.gptq.gptq_fwrd(model, trainloader, torch.device("cuda:0"), qargs)
        elif args.method == "gptaq":
            modules.gptaq.gptaq_fwrd(model, trainloader, torch.device("cuda:0"), qargs)
        else:
            modules.rescomp.gptaq_fwrd(model, trainloader, torch.device("cuda:0"), qargs)
    save_quant_checkpoint(
        model,
        checkpoint,
        {
            "protocol": PROTOCOL,
            "method": args.method,
            "rotation": args.rotation,
            "optimized_rotation_path": (
                str(args.optimized_rotation_path) if args.optimized_rotation_path else None
            ),
            "model_key": args.model_key,
            "seed": args.seed,
            "w_bits": 2,
            "a_bits": 4,
            "k_bits": 4,
            "v_bits": 4,
            "group_size": 128,
            "calibration": "wikitext2",
            "calibration_samples": args.calibration_samples,
            "calibration_seq_len": args.calibration_seq_len,
            "smoke_layers": args.smoke_layers,
        },
    )
    atomic_json(run_dir / "state" / "quantize.done.json", {"finished": time.time()})
    log(f"Saved quantized graph to {checkpoint}")
    return model, qargs


@torch.inference_mode()
def evaluate_ppl_on_model(
    model,
    tokenizer,
    dataset: str,
    output: Path,
    data_root: Path,
) -> None:
    wiki = data_root / "raw/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet"
    c4 = data_root / "raw/c4/en/c4-validation.00000-of-00008.json.gz"
    ids = build_ppl_ids(dataset, tokenizer, wiki, c4)
    seq_len = 2048
    available = ids.numel() // seq_len
    blocks = min(256 if dataset == "c4" else available, available)
    total_nll = torch.zeros((), device="cuda", dtype=torch.float64)
    total_tokens = 0
    for index in range(blocks):
        batch = ids[:, index * seq_len : (index + 1) * seq_len].to("cuda")
        logits = model(input_ids=batch, use_cache=False).logits[:, :-1].float()
        labels = batch[:, 1:]
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1), reduction="sum"
        ).double()
        total_tokens += labels.numel()
        if (index + 1) % 16 == 0:
            log(f"{dataset} PPL blocks {index + 1}/{blocks}")
    mean_nll = float((total_nll / total_tokens).cpu())
    atomic_json(
        output,
        {
            "dataset": dataset,
            "ppl": math.exp(mean_nll),
            "mean_nll": mean_nll,
            "tokens": total_tokens,
            "blocks": blocks,
            "seq_len": seq_len,
            "protocol": PROTOCOL,
        },
    )


def evaluate_task_on_model(model, tokenizer, task: str, output: Path, batch_size: int) -> None:
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    harness = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, max_length=2048)
    raw = lm_eval.simple_evaluate(
        model=harness,
        tasks=[task],
        num_fewshot=0,
        batch_size=batch_size,
        log_samples=False,
    )
    task_result = raw["results"][task]
    metric = METRIC_BY_TASK[task]
    atomic_json(
        output,
        {
            "task": task,
            "metric": metric,
            "value": float(task_result[metric]),
            "stderr": float(task_result.get(metric.replace(",none", "_stderr,none"), float("nan"))),
            "lm_eval_version": getattr(lm_eval, "__version__", "unknown"),
            "num_fewshot": 0,
            "protocol": PROTOCOL,
            "raw_results": jsonable(raw.get("results", {})),
        },
    )
    del harness
    torch.cuda.empty_cache()


def evaluate_variant(
    args: argparse.Namespace,
    model,
    tokenizer,
    run_dir: Path,
    variant: str,
    deadline: float,
    evaluate_tasks: bool = True,
) -> bool:
    output_dir = run_dir / "results" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in ("wikitext2", "c4"):
        output = output_dir / f"ppl_{dataset}.json"
        if output.is_file():
            continue
        if remaining_minutes(deadline) <= args.eval_start_margin:
            return False
        evaluate_ppl_on_model(model, tokenizer, dataset, output, args.data_root)
    if not evaluate_tasks:
        atomic_json(
            output_dir / "TASKS_SKIPPED.json",
            {
                "variant": variant,
                "reason": "baseline_ppl_only",
                "tasks": list(TASKS),
                "protocol": PROTOCOL,
            },
        )
        log(f"Skipping downstream tasks for {variant}; PPL-only baseline policy is enabled")
        return True
    for task in TASKS:
        output = output_dir / f"task_{task}.json"
        if output.is_file():
            continue
        if remaining_minutes(deadline) <= args.eval_start_margin:
            return False
        evaluate_task_on_model(model, tokenizer, task, output, args.eval_batch_size)
    return True


def affine_parameters(model) -> tuple[list[torch.nn.Parameter], list[str]]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parameters: list[torch.nn.Parameter] = []
    names: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, AffineRMSN):
            module.weight.requires_grad_(True)
            parameters.append(module.weight)
            names.append(f"{name}.weight")
    if not parameters or any(parameter.numel() <= 1 for parameter in parameters):
        raise RuntimeError("The post-QuaRot channel-affine response subspace was not constructed")
    return parameters, names


def save_affine(model, path: Path) -> None:
    state = {
        f"{name}.weight": module.weight.detach().cpu().contiguous()
        for name, module in model.named_modules()
        if isinstance(module, AffineRMSN)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    save_file(state, str(temporary))
    temporary.replace(path)


def load_affine(model, path: Path) -> None:
    state = load_file(str(path), device="cpu")
    modules = dict(model.named_modules())
    for full_name, tensor in state.items():
        module_name, _ = full_name.rsplit(".", 1)
        parameter = modules[module_name].weight
        parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


@contextlib.contextmanager
def activation_ste(quant_module):
    original_asym = quant_module.asym_quant_dequant
    original_sym = quant_module.sym_quant_dequant

    def asym_ste(x, scale, zero, maxq):
        quantized = original_asym(x, scale, zero, maxq)
        return x + (quantized - x).detach()

    def sym_ste(x, scale, maxq):
        quantized = original_sym(x, scale, maxq)
        return x + (quantized - x).detach()

    quant_module.asym_quant_dequant = asym_ste
    quant_module.sym_quant_dequant = sym_ste
    try:
        yield
    finally:
        quant_module.asym_quant_dequant = original_asym
        quant_module.sym_quant_dequant = original_sym


def nap_blocks(args: argparse.Namespace, modules) -> list[torch.Tensor]:
    loader = modules.data.get_loaders(
        "wikitext2",
        nsamples=args.calibration_samples,
        seed=args.seed,
        model=args.model_path,
        seqlen=args.calibration_seq_len,
        hf_token=None,
        eval_mode=False,
    )
    sources = [item[0][0].cpu() for item in loader]
    rng = random.Random(args.seed + 100003)
    order = list(range(len(sources)))
    rng.shuffle(order)
    blocks: list[torch.Tensor] = []
    for step in range(args.nap_steps):
        source = sources[order[step % len(order)]]
        start = rng.randint(0, source.numel() - args.nap_seq_len)
        blocks.append(source[start : start + args.nap_seq_len].clone())
    return blocks


def train_nap_full_quant(
    args: argparse.Namespace,
    model,
    modules,
    run_dir: Path,
    deadline: float,
) -> bool:
    nap_dir = run_dir / "nap"
    affine_path = nap_dir / "norm_latest.safetensors"
    trainer_path = nap_dir / "trainer_state.pt"
    nap_dir.mkdir(parents=True, exist_ok=True)
    parameters, names = affine_parameters(model)
    if affine_path.is_file():
        load_affine(model, affine_path)
    optimizer = torch.optim.AdamW(parameters, lr=args.nap_lr, weight_decay=0.0)
    start_step = 0
    history: list[dict[str, float]] = []
    if trainer_path.is_file():
        state = torch.load(trainer_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(parameters[0].device)
        start_step = int(state["step"])
        history = list(state.get("history", []))
        log(f"Resuming full-quant NAP from step {start_step}")
    if start_step >= args.nap_steps:
        return True
    teacher = load_dense_model(args.model_path)
    teacher.eval()
    teacher.config.use_cache = False
    model.train()
    blocks = nap_blocks(args, modules)
    dtype = next(model.parameters()).dtype
    with activation_ste(modules.quant):
        for step in range(start_step, args.nap_steps):
            if remaining_minutes(deadline) <= args.nap_checkpoint_margin:
                break
            input_ids = blocks[step].unsqueeze(0).to("cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=next(teacher.parameters()).dtype):
                teacher_logits = teacher(input_ids=input_ids, use_cache=False).logits
            with torch.autocast(device_type="cuda", dtype=dtype):
                student_logits = model(input_ids=input_ids, use_cache=False).logits
                student_shift = student_logits[:, :-1].float()
                teacher_shift = teacher_logits[:, :-1].float()
                labels = input_ids[:, 1:]
                ce = F.cross_entropy(student_shift.reshape(-1, student_shift.size(-1)), labels.reshape(-1))
                kd = F.kl_div(
                    F.log_softmax(student_shift, dim=-1),
                    F.softmax(teacher_shift, dim=-1),
                    reduction="none",
                ).sum(-1).mean()
                loss = ce + kd
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite NAP loss at step {step + 1}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            if step == start_step and float(grad_norm) <= 1e-12:
                raise RuntimeError("Zero affine gradient: activation STE is not connected to the model graph")
            optimizer.step()
            row = {
                "step": float(step + 1),
                "loss": float(loss.detach()),
                "ce": float(ce.detach()),
                "kd": float(kd.detach()),
                "grad_norm": float(grad_norm),
            }
            history.append(row)
            if (step + 1) % 10 == 0 or step + 1 == args.nap_steps:
                save_affine(model, affine_path)
                temporary = trainer_path.with_suffix(".pt.tmp")
                torch.save(
                    {
                        "step": step + 1,
                        "optimizer": optimizer.state_dict(),
                        "history": history,
                        "parameter_names": names,
                    },
                    temporary,
                )
                temporary.replace(trainer_path)
                atomic_json(nap_dir / "history.json", history)
                log(f"NAP step {step + 1}/{args.nap_steps}: loss={row['loss']:.5f}")
    save_affine(model, affine_path)
    final_step = int(history[-1]["step"]) if history else start_step
    temporary = trainer_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "step": final_step,
            "optimizer": optimizer.state_dict(),
            "history": history,
            "parameter_names": names,
        },
        temporary,
    )
    temporary.replace(trainer_path)
    atomic_json(
        nap_dir / "summary.json",
        {
            "completed": final_step >= args.nap_steps,
            "steps": final_step,
            "trainable_parameters": sum(parameter.numel() for parameter in parameters),
            "parameter_names": names,
            "protocol": PROTOCOL,
            "straight_through_activation_gradient": True,
        },
    )
    unload(teacher)
    model.eval()
    return final_step >= args.nap_steps


def expected_results(
    run_dir: Path,
    baseline_ppl_only: bool = False,
    skip_tasks: bool = False,
) -> list[Path]:
    outputs: list[Path] = []
    for variant in ("baseline", "nap"):
        outputs.extend(run_dir / "results" / variant / f"ppl_{name}.json" for name in ("wikitext2", "c4"))
        if not skip_tasks and (variant != "baseline" or not baseline_ppl_only):
            outputs.extend(run_dir / "results" / variant / f"task_{name}.json" for name in TASKS)
    return outputs


def preflight(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("A compatible CUDA-API accelerator is not available")
    required = [
        Path(args.model_path) / "config.json",
        args.data_root / "raw/wikitext2/wikitext-2-raw-v1/train-00000-of-00001.parquet",
        args.data_root / "raw/wikitext2/wikitext-2-raw-v1/test-00000-of-00001.parquet",
        args.data_root / "raw/c4/en/c4-validation.00000-of-00008.json.gz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if args.rotation == "spinquant":
        if args.model_key == "qwen3_8b":
            raise ValueError("The released SpinQuant rotations do not support Qwen3-8B")
        if args.optimized_rotation_path is None:
            raise ValueError("--optimized-rotation-path is required for SpinQuant")
        elif not args.optimized_rotation_path.is_file():
            missing.append(str(args.optimized_rotation_path))
    if missing:
        raise FileNotFoundError("Required offline inputs are missing:\n- " + "\n- ".join(missing))
    if not args.skip_tasks:
        try:
            import lm_eval  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Task evaluation requires the evaluation extra: pip install -e '.[evaluation]'") from exc
    log(
        f"Preflight OK: device={torch.cuda.get_device_name(0)}, "
        f"remaining protocol=W2A4KV4/{args.rotation}/WikiText2-calibration"
    )


def main() -> int:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.runs_root = args.runs_root.resolve()
    if args.optimized_rotation_path is not None:
        args.optimized_rotation_path = args.optimized_rotation_path.resolve()
    project_root = Path(__file__).resolve().parent
    os.environ["DATA_ROOT"] = str(args.data_root)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    seed_everything(args.seed)
    deadline = time.time() + args.minutes * 60.0
    method_tag = "rescomp" if args.method == "rescomp_gptaq" else args.method
    smoke_tag = f"_smoke{args.smoke_layers}" if args.smoke_layers else ""
    rotation_tag = "spinq" if args.rotation == "spinquant" else "quarot"
    job_id = f"{args.model_key}_{rotation_tag}_{method_tag}_w2a4kv4_s{args.seed}{smoke_tag}"
    run_dir = args.runs_root / job_id
    run_dir.mkdir(parents=True, exist_ok=True)
    protocol = {
        "protocol": PROTOCOL,
        "job_id": job_id,
        "model_path": args.model_path,
        "model_key": args.model_key,
        "method": args.method,
        "seed": args.seed,
        "calibration": "wikitext2",
        "calibration_samples": args.calibration_samples,
        "calibration_seq_len": args.calibration_seq_len,
        "w_bits": 2,
        "a_bits": 4,
        "k_bits": 4,
        "v_bits": 4,
        "group_size": 128,
        "rotation": "spinquant_learned" if args.rotation == "spinquant" else "quarot_hadamard",
        "a_asym": True,
        "k_asym": True,
        "v_asym": True,
        "a_clip_ratio": 0.9,
        "k_clip_ratio": 0.95,
        "v_clip_ratio": 0.95,
        "nap_steps": args.nap_steps,
        "nap_seq_len": args.nap_seq_len,
        "nap_lr": args.nap_lr,
        "smoke_layers": args.smoke_layers,
    }
    if args.rotation == "spinquant":
        protocol["optimized_rotation_path"] = str(args.optimized_rotation_path)
        protocol["optimized_rotation_sha256"] = sha256_file(args.optimized_rotation_path)
    protocol_file = run_dir / "protocol.json"
    if protocol_file.is_file() and json.loads(protocol_file.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError(f"Protocol changed for existing job {job_id}; use a new runs root")
    atomic_json(protocol_file, protocol)
    if (run_dir / "COMPLETE.json").is_file():
        log(f"Already complete: {job_id}")
        return 0
    preflight(args)
    modules = official_modules(project_root)
    model, _ = load_or_quantize(args, run_dir, modules, deadline)
    model.to("cuda")
    model.eval()
    model.config.use_cache = False
    tokenizer = load_tokenizer(args.model_path, use_fast=False)
    if not evaluate_variant(
        args,
        model,
        tokenizer,
        run_dir,
        "baseline",
        deadline,
        evaluate_tasks=not (args.baseline_ppl_only or args.skip_tasks),
    ):
        return 75
    if not train_nap_full_quant(args, model, modules, run_dir, deadline):
        log("NAP checkpoint saved; resume this job in the next reservation")
        return 75
    if not evaluate_variant(
        args,
        model,
        tokenizer,
        run_dir,
        "nap",
        deadline,
        evaluate_tasks=not args.skip_tasks,
    ):
        return 75
    if all(
        path.is_file()
        for path in expected_results(run_dir, args.baseline_ppl_only, args.skip_tasks)
    ):
        atomic_json(run_dir / "COMPLETE.json", {"job_id": job_id, "finished": time.time(), "protocol": PROTOCOL})
        log(f"COMPLETE: {job_id}")
        return 0
    return 75


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("Interrupted; completed metrics, quantized artifact, and NAP checkpoints are preserved")
        raise SystemExit(130)
