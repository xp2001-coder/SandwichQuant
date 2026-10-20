# Local data and evaluation

Obtain models and datasets separately under their applicable terms.
The code expects:

```text
DATA_ROOT/
├── raw/
│   ├── c4/en/
│   │   ├── c4-train.00000-of-01024.json.gz
│   │   └── c4-validation.00000-of-00008.json.gz
│   ├── c4_parquet/
│   │   ├── train.parquet
│   │   └── validation.parquet
│   └── wikitext2/wikitext-2-raw-v1/
│       ├── train-00000-of-00001.parquet
│       └── test-00000-of-00001.parquet
└── hf_home/
```

`prepare_local_data.py` creates the two C4 parquet files from the local gzip
files. It does not download data. `hf_home/` is the local Hugging Face cache;
set `HF_HOME` to that path and prepare downstream-task caches before enabling
accuracy evaluation.

The default weight-only calibration uses 128 C4 training spans of length
2,048. Affine calibration takes shorter subblocks from the cached calibration
tensor. The rotation-aware runner calibrates on WikiText-2 training data; inspect its
recorded protocol instead of assuming the samplers are identical.

WikiText-2 evaluation concatenates test documents with blank lines and uses
complete 2,048-token blocks. C4 evaluation concatenates the first 1,100 local
validation documents with spaces and uses 256 blocks. Tokenizer defaults,
including special tokens, are retained. Full C4 evaluation requires enough
tokens for all 256 blocks; always check the reported block count.

## AVG6

The evaluation harness uses zero-shot tasks:

| Task | Metric |
| --- | --- |
| PIQA | `acc_norm,none` |
| ARC-Easy | `acc_norm,none` |
| ARC-Challenge | `acc_norm,none` |
| HellaSwag | `acc_norm,none` |
| WinoGrande | `acc,none` |
| BoolQ | `acc,none` |

AVG6 is the arithmetic mean of all six scores. A missing task is not zero and
must not be hidden by averaging only completed tasks. Accuracy values in CSV
are fractions; the order-reporting script displays percentages.

Do not select calibration settings on these evaluation splits.
