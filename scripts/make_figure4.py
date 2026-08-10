"""Regenerate the RQ2 input-representation figure
(figures/acl_text_tasks_comparison_both_datasets.pdf).

Why a script rather than the notebook: both decoding configs now exist in the repo
for the text tasks, so anything that selects runs implicitly can silently mix them.
This pins every selector -- batch size 10, temperature 0.6 / top_p 0.9,
`avg_calculated_field_em` -- which is exactly what Tables 3 and 4 report, so the
figure and the tables cannot drift apart.

Keeps the layout and larger fonts of the current version; only the selection is fixed.

Run from the repository root:  python visualizations/make_ocr_impact_acl.py
"""

import csv
import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = (ROOT / "figures"
       / "acl_text_tasks_comparison_both_datasets.pdf")

METRIC = "avg_calculated_field_em"
BATCH = "10"
TEMP = "0.6"

NAME = {
    "Qwen3-0.6B": "Qwen3-0.6B", "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B", "Qwen3-8B": "Qwen3-8B",
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Ministral-3-3B-Instruct-2512": "Ministral-3-3B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
}
ORDER = ["Arctic-TILT", "Llama-3.2-1B", "Llama-3.2-3B", "Ministral-3-3B", "Mistral-7B",
         "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]

# Colour is bound to the parser, not to its position in a panel.
PARSER = [("default", "Baseline (Dataset)", "#d9d98c"),
          ("deepseek", "DeepSeek OCR", "#1f5fa8"),
          ("docling", "Docling", "#5fa8dc"),
          ("tesseract", "Tesseract", "#8a8a8a")]

# arctic_results/evaluation_<dataset>_<parser>.csv, avg_calculated_field_em
ARCTIC = {"VRDU": {"default": 59.06, "deepseek": 54.81, "docling": 57.93, "tesseract": 64.62},
          "Kleister-NDA": {"deepseek": 81.47, "docling": 91.91, "tesseract": 92.43}}



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
    out = {"VRDU": {}, "Kleister-NDA": {}}
    for path in glob.glob(str(RESULTS / "run_reports" / "*_noquant" / "*.csv")):
        for row in csv.DictReader(open(path)):
            _check_precision_dir(path, row["backend"])
            if row["scenario"] != "batch" or row["batch_size"] != BATCH:
                continue
            if row["temperature"] != TEMP or not row.get(METRIC):
                continue
            model = NAME.get(row["model_name"])
            if not model or "vision" in row["task"]:
                continue
            ds = "VRDU" if row["task"].startswith("vrdu") else "Kleister-NDA"
            out[ds].setdefault(model, {})[row["task"].split("_")[-1]] = float(row[METRIC]) * 100
    for ds, vals in ARCTIC.items():
        out[ds]["Arctic-TILT"] = dict(vals)
    return out


data = load()
# One ACL column wide, panels stacked. Drawn at roughly its final size so the
# type renders at its nominal point size instead of being shrunk by
# \includegraphics; the per-bar value labels do not survive at this width and
# are dropped, since Tables 3 and 4 carry the same numbers.
fig, axes = plt.subplots(2, 1, figsize=(3.15, 3.5), sharex=True)

for ax, (ds, title) in zip(axes, [("VRDU", "VRDU (scanned, layout-rich)"),
                                  ("Kleister-NDA", "Kleister-NDA (born-digital)")]):
    d = data[ds]
    models = [m for m in ORDER if m in d]
    parsers = [p for p in PARSER if any(p[0] in d[m] for m in models)]
    x = np.arange(len(models))
    w = 0.80 / len(parsers)

    for j, (key, label, color) in enumerate(parsers):
        xs = x - 0.40 + w * (j + 0.5)
        vals = [d[m].get(key, np.nan) for m in models]
        ax.bar(xs, vals, w * 0.88, color=color, edgecolor="black", linewidth=0.3,
               label=label, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=6.2)
    ax.set_ylabel("Avg. field EM (%)", fontsize=7, fontweight="bold")
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(title, fontsize=7.5, fontweight="bold", pad=3)
    ax.tick_params(axis="y", labelsize=6.5)
    ax.tick_params(axis="x", length=2)
    ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# VRDU carries all four parsers, so its handles cover both panels.
handles, labels = axes[0].get_legend_handles_labels()
# tight_layout first, reserving a band at the top, then place the legend into
# that band -- otherwise it lands on the upper panel's title.
fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=1.2)
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncol=2, fontsize=6.5, frameon=True, handlelength=1.3,
           columnspacing=1.2, handletextpad=0.5, borderpad=0.4)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
for ds in ("VRDU", "Kleister-NDA"):
    print(f"\n{ds}:")
    for m in ORDER:
        if m in data[ds]:
            print("   " + f"{m:16s}" + "  ".join(f"{k}={v:.1f}" for k, v in sorted(data[ds][m].items())))
