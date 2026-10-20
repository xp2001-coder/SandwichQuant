# External backend setup

This source release deliberately excludes vendored backend code.
Supply source checkouts separately after reviewing their terms.

| Path | Configuration | Expected contents |
| --- | --- | --- |
| ResComp-style adapter | `SQ_RESCOMP_ROOT` | A checkout containing `fake_quant/` |
| QEP/AWQ adapter | `SQ_QEP_ROOT` | A checkout containing `src/llama.py` |

For example:

```bash
export SQ_RESCOMP_ROOT=/path/to/compatible/ResComp
export SQ_QEP_ROOT=/path/to/compatible/qep
```

The default local locations are `third_party/ResComp` and `third_party/qep`;
the entire `third_party/` directory is ignored by Git. There is no automatic
download or installation.

## Compatibility contract

The external launchers preserve the existing integration interface. A stock
upstream checkout is not guaranteed to implement the local/offline interface:

- The ResComp adapter calls `fake_quant/main.py` or `main_r.py`, passes
  `--save_qmodel_path`, and uses `NAP_CALIBRATION_TOKENS` for a shared token
  tensor and `NAP_QUANTIZE_ONLY=1` for checkpoint-only execution.
- The weight-only artifact is a trusted local `model.pt` containing a dense
  fake-quantized model state. Extra backend objects may require their defining
  modules at load time.
- The rotation-aware path additionally expects `gptq_utils`,
  `gptaq_utils`, `gptaq_utils_r`, `hadamard_utils`, `rotation_utils`,
  `quant_utils`, `model_utils`, and `data_utils` under `fake_quant/`.
  Upstream helper modules must be installed or made importable separately.
- The QEP adapter expects a compatible checkpoint-saving `src/llama.py`
  interface. Its AWQ execution and calibration sampling must be checked
  against the supplied checkout.

Do not silently substitute an unrelated backend or relax checkpoint
compatibility checks. Keep a record of the external revision used for each
run. Never publish private local backend changes without reviewing their
redistribution terms.

The optional `gptqmodel` extra exposes a different adapter:
`NAP_QUANT_BACKEND=gptqmodel`. Its bit-width support and model support depend
on the installed version. The legacy four-order weight-only runner requires
the external dense-state GPTQ artifact and does not accept that packed format.
