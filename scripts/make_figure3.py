"""Regenerate the RQ1 baselines figure (figures/rq1_baselines_compact.pdf).

Pins the text models to temperature 0.6 / top_p 0.9 and `avg_calculated_field_em`,
matching Tables 3-4 and the other figures. Both decoding configs exist in the repo
for the text tasks, so an implicit selection picks whichever run happens to be
present. Vision models were always run at 0.7 / 0.8.

Single-request inference (batch size 1), unquantized. Energy covers model
inference only; parsing energy is added in the end-to-end analysis.

Run from the repository root:  python visualizations/make_rq1_baselines_acl.py
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
OUT = ROOT / "figures" / "rq1_baselines_compact.pdf"

METRIC = "avg_calculated_field_em"
TEMP_TEXT = "0.6"

NAME = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Ministral-3-3B-Instruct-2512": "Ministral-3-3B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen3-0.6B": "Qwen3-0.6B", "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B", "Qwen3-8B": "Qwen3-8B",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}
TEXT = ["Llama-3.2-1B", "Llama-3.2-3B", "Ministral-3-3B", "Mistral-7B",
        "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
VIS = ["Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B"]
SPEC = ["Arctic-TILT", "NuExtract-2.0-4B"]

# Specialized models run through their own pipelines, profiled by CodeCarbon.
# Energy is Wh per document here; load() rescales it to mWh per page.
SPECIAL = {
    "VRDU": {"Arctic-TILT": (64.6, 0.0100), "NuExtract-2.0-4B": (78.3, 0.0327)},
    "Kleister-NDA": {"Arctic-TILT": (92.4, 0.0396), "NuExtract-2.0-4B": (52.1, 0.0666)},
}
COLOR = {"text": "#81b29a", "vis": "#f4a261", "spec": "#9b5de5"}



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
        for r in csv.DictReader(open(path)):
            _check_precision_dir(path, r["backend"])
            if r["scenario"] != "single" or "quant" in r["backend"] or not r.get(METRIC):
                continue
            kind = r["task"].split("_")[-1]
            if kind not in ("tesseract", "vision"):
                continue
            if kind == "tesseract" and r["temperature"] != TEMP_TEXT:
                continue
            m = NAME.get(r["model_name"])
            if not m:
                continue
            ds = "VRDU" if r["task"].startswith("vrdu") else "Kleister-NDA"
            out[ds][m] = (float(r[METRIC]) * 100,
                          float(r["total_energy_wh"]) / int(r["num_queries"])
                          / R.pages_per_doc(ds) * 1000)
    for ds, vals in SPECIAL.items():
        ppd = R.pages_per_doc(ds)
        out[ds].update({m: (em, wh / ppd * 1000) for m, (em, wh) in vals.items()})
    return out


data = load()
# Both panels share one energy scale so the two datasets can be read against
# each other -- which is the point of normalizing per page. Headroom leaves room
# for the rotated data labels above the tallest bar.
ENERGY_MAX = max(e for d in data.values() for _, e in d.values()) * 1.30

DATASETS = ["VRDU", "Kleister-NDA"]
# The panels share one x axis, so the model list is resolved once and both
# datasets must agree on it -- otherwise the shared tick labels would silently
# describe the wrong bars.
MODELS = [m for m in TEXT + VIS + SPEC if m in data[DATASETS[0]]]
for ds in DATASETS[1:]:
    other = [m for m in TEXT + VIS + SPEC if m in data[ds]]
    if other != MODELS:
        raise SystemExit(
            f"ERROR: panels disagree on the model list, so the x axis cannot be shared.\n"
            f"  {DATASETS[0]}: {MODELS}\n  {ds}: {other}")

fig, axes = plt.subplots(2, 1, figsize=(13.0, 5.8), sharex=True)

for ax1, ds in zip(axes, DATASETS):
    d = data[ds]
    models = MODELS
    colors = [COLOR["text"] if m in TEXT else COLOR["vis"] if m in VIS else COLOR["spec"]
              for m in models]
    ax2 = ax1.twinx()
    x = np.arange(len(models))
    w = 0.38

    em = [d[m][0] for m in models]
    en = [d[m][1] for m in models]
    r1 = ax1.bar(x - w / 2, em, w, color=colors, edgecolor="black", linewidth=0.5, zorder=3)
    r2 = ax2.bar(x + w / 2, en, w, color=colors, edgecolor="black", linewidth=0.5,
                 hatch="//", zorder=3)
    # Each series is labelled on its own axis, otherwise the energy values are
    # placed against the exact-match scale and end up clipped inside the bars.
    for ax, rects, vals, fmt in ((ax1, r1, em, "{:.1f}"), (ax2, r2, en, "{:.1f}")):
        for rect, v in zip(rects, vals):
            ax.annotate(fmt.format(v),
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", fontsize=7, rotation=90, zorder=5)
    ax1.set_ylabel("Exact Match (%)", fontweight="bold", fontsize=9)
    ax1.set_ylim(0, 125)
    ax2.set_ylabel("Energy (mWh/page)", fontweight="bold", fontsize=9)
    ax2.set_ylim(0, ENERGY_MAX)
    ax1.set_xticks(x)
    ax1.tick_params(axis="y", labelsize=8)
    # Shared x: only the lower panel carries the model names.
    ax2.tick_params(axis="x", bottom=False, labelbottom=False)
    ax2.tick_params(axis="y", labelsize=8)
    ax1.set_title(f"Dataset: {ds}", fontweight="bold", fontsize=10, pad=4)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
    ax1.set_axisbelow(True)
    # Open frame, as in Figures 4 and 5. This plot is dual-axis, so each y-axis
    # keeps its own side -- exact match on the left, energy on the right -- and
    # only the top rule goes.
    for side in ("top", "right"):
        ax1.spines[side].set_visible(False)
    for side in ("top", "left"):
        ax2.spines[side].set_visible(False)

axes[-1].set_xticklabels(MODELS, rotation=25, ha="right", fontsize=8.5)

from matplotlib.patches import Patch
handles = [Patch(facecolor=COLOR["text"], edgecolor="black", label="Text-only"),
           Patch(facecolor=COLOR["vis"], edgecolor="black", label="Vision-language"),
           Patch(facecolor=COLOR["spec"], edgecolor="black", label="Specialized"),
           Patch(facecolor="white", edgecolor="black", label="Exact Match (%)"),
           Patch(facecolor="white", edgecolor="black", hatch="//", label="Energy (mWh/page)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=5,
           fontsize=8.5, frameon=True)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")
for ds in data:
    print(f"\n{ds}: " + "  ".join(f"{m}={data[ds][m][0]:.1f}" for m in TEXT + VIS + SPEC if m in data[ds]))
