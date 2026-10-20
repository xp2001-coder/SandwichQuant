#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from nap_llm.common import Job, PPL_DATASETS, TASKS


parser = argparse.ArgumentParser()
parser.add_argument("--matrix", type=Path, required=True)
parser.add_argument("--runs-root", type=Path, required=True)
parser.add_argument("--output", type=Path)
args = parser.parse_args()

with args.matrix.open("r", encoding="utf-8-sig", newline="") as handle:
    jobs = [Job.from_row(row) for row in csv.DictReader(handle) if row.get("enabled", "1").lower() not in {"0", "false", "no"}]

rows = []
for job in jobs:
    run = args.runs_root / job.job_id
    for variant in (["baseline", "nap"] if job.run_nap else ["baseline"]):
        row = {"job_id": job.job_id, "model": job.model_key, "method": job.method + ("+Post" if variant == "nap" else ""), "bits": job.bits, "group_size": job.group_size, "seed": job.seed}
        for dataset in PPL_DATASETS:
            path = run / "results" / variant / f"ppl_{dataset}.json"
            row[f"ppl_{dataset}"] = json.loads(path.read_text(encoding="utf-8"))["ppl"] if path.exists() else ""
        values = []
        for task in TASKS:
            path = run / "results" / variant / f"task_{task}.json"
            value = json.loads(path.read_text(encoding="utf-8"))["value"] if path.exists() else ""
            row[task] = value
            if value != "":
                values.append(float(value))
        row["avg"] = sum(values) / len(values) if len(values) == len(TASKS) else ""
        row["complete"] = all(row[key] != "" for key in [*(f"ppl_{x}" for x in PPL_DATASETS), *TASKS])
        rows.append(row)

fields = ["job_id", "model", "method", "bits", "group_size", "seed", "ppl_wikitext2", "ppl_c4", *TASKS, "avg", "complete"]
output = args.output or args.runs_root / "summary.csv"
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print(output)
