"""Regenerate the RQ3 batch-size figure
(figures/acl_energy_per_doc_by_batch_size_vrdu_vllm_tesseract_unquant.pdf).

Pins the text models to temperature 0.6 / top_p 0.9 -- both decoding configs now
exist in the repo, so an implicit selection silently mixes them and produces a
series whose points were measured under different settings. Vision models were
always run at 0.7 / 0.8 and are unaffected.

Includes the Qwen3-VL-8B batch 40 and 60 runs that were added after the Bench360
timeout was raised, so the figure no longer has missing bars.

Run from the repository root:  python visualizations/make_batchsweep_acl.py
"""

import csv
import glob
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = (ROOT / "figures"
       / "acl_energy_per_doc_by_batch_size_vrdu_vllm_tesseract_unquant.pdf")

BATCHES = [1, 5, 10, 20, 40, 60]
TEMP_TEXT = "0.6"
N_DOCS = 500

NAME = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Ministral-3-3B-Instruct-2512": "Ministral-3-3B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen3-0.6B": "Qwen3-0.6B", "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B", "Qwen3-8B": "Qwen3-8B",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}
ORDER = ["Arctic-TILT", "Llama-3.2-1B", "Llama-3.2-3B", "Ministral-3-3B", "Mistral-7B",
         "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B",
         "Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B"]



def _check_precision_dir(path, backend):
    """A run belongs in a directory matching its precision.

    Misfiled files silently compete with the runs the paper reports -- an FP16
    Qwen3-VL-8B run filed under vrdu_vision_quant/ differed from its correctly
    filed twin by 59% in energy. Such files now live in superseded_runs/; this
    guard makes a recurrence fail loudly instead of changing a figure quietly.
    """
    import sys
    top = str(path).split("/")[-3] if "/run_report/" in str(path) else ""
    if top and (("quant" in backend) != top.endswith("_quant")):
        sys.exit(f"ERROR: {path} has backend={backend} but sits in {top}. "
                 f"Move it to the directory matching its precision, or to superseded_runs/.")

def load():
    e = {m: {} for m in ORDER}
    for path in glob.glob(str(RESULTS / "run_reports" / "*_noquant" / "*.csv")):
        for row in csv.DictReader(open(path)):
            _check_precision_dir(path, row["backend"])
            if "quant" in row["backend"]:
                continue
            task = row["task"]
            if task == "vrdu_tesseract":
                if row["temperature"] != TEMP_TEXT:
                    continue
            elif task != "vrdu_vision":
                continue
            m = NAME.get(row["model_name"])
            if not m:
                continue
            bs = int(row["batch_size"]) if row["scenario"] == "batch" else 1
            e[m][bs] = (float(row["total_energy_wh"]) / int(row["num_queries"])
                        / R.pages_per_doc("VRDU") * 1000)
    # Arctic-TILT sweep is profiled separately by CodeCarbon (kWh over 500 docs).
    p = RESULTS / "arctic_tilt" / "energy_batch_size.csv"
    if p.exists():
        for row in csv.DictReader(open(p)):
            e["Arctic-TILT"][int(row["batch_size"])] = (
                float(row["energy_consumed"]) * 1000 / N_DOCS
                / R.pages_per_doc("VRDU") * 1000)
    return e


energy = load()
models = [m for m in ORDER if energy[m]]
BLUES = plt.get_cmap("Blues")(np.linspace(0.32, 0.95, len(BATCHES)))

fig, ax = plt.subplots(figsize=(13.0, 3.3))
x = np.arange(len(models))
w = 0.82 / len(BATCHES)

for j, bs in enumerate(BATCHES):
    xs = x - 0.41 + w * (j + 0.5)
    ax.bar(xs, [energy[m].get(bs, np.nan) for m in models], w * 0.9,
           color=BLUES[j], edgecolor="black", linewidth=0.35, label=f"BS={bs}", zorder=3)

# Label only the per-model minimum and maximum, as the caption states.
for xi, m in zip(x, models):
    vals = {bs: v for bs, v in energy[m].items() if bs in BATCHES}
    if not vals:
        continue
    for bs in (max(vals, key=vals.get), min(vals, key=vals.get)):
        j = BATCHES.index(bs)
        ax.annotate(f"{vals[bs]:.1f}", xy=(xi - 0.41 + w * (j + 0.5), vals[bs]),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=5.9, rotation=90, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("Energy per page (mWh)", fontsize=9.5, fontweight="bold")
ax.set_ylim(0, max(max(v.values()) for v in energy.values() if v) * 1.24)
ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
# Upper left: the tallest bar (Qwen3-VL-8B at BS 1) is on the right, and a
# legend there hides its data label.
ax.legend(title="Batch Size", fontsize=7.5, title_fontsize=8, loc="upper left", ncol=3,
          frameon=True, framealpha=0.94, columnspacing=1.0, handlelength=1.2)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}\n")
for m in models:
    v = {b: energy[m][b] for b in BATCHES if b in energy[m]}
    if not v:
        continue
    mn, mx = min(v.values()), max(v.values())
    print(f"  {m:16s} " + " ".join(f"{energy[m].get(b, float('nan')):.1f}" for b in BATCHES)
          + f"   -{100*(1-mn/mx):.0f}%")
