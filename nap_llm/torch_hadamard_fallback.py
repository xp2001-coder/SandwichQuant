"""Autograd-safe PyTorch fallback for fast_hadamard_transform.

Some accelerator environments do not ship the optional native extension.
This implementation is slower, but it keeps the QuaRot
protocol functional and differentiable for NAP.  The runner only installs it
when importing the native extension fails.
"""

from __future__ import annotations

import math

import torch


def hadamard_transform(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    size = int(x.shape[-1])
    if size < 1 or size & (size - 1):
        raise ValueError(f"Hadamard dimension must be a power of two, got {size}")
    original_shape = x.shape
    output = x.contiguous()
    stride = 1
    while stride < size:
        grouped = output.reshape(*original_shape[:-1], -1, 2, stride)
        left = grouped[..., 0, :]
        right = grouped[..., 1, :]
        output = torch.cat((left + right, left - right), dim=-1).reshape(original_shape)
        stride *= 2
    if scale != 1.0:
        output = output * scale
    return output


__all__ = ["hadamard_transform"]
