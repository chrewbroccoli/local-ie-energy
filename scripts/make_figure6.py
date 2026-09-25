"""Regenerate the RQ3 quantization figure
(figures/acl_rq3_qwen_quant_batch_acc_labeled_all_energy_vrdu.pdf).

Fixes relative to visualization.ipynb:
  * One metric for every series. The previous figure read `avg_calculated_field_em`
    where that column existed (Qwen3-4B FP16 -> 58.8%) and fell back to
    `avg_field_em` where it did not (Qwen3-4B FP8 -> 56.5%), so the two bars of the
    quantization comparison were computed with different metrics. We use
    `avg_field_em` throughout, which is present in every run report and is the
    metric defined in the paper.
  * Compact legend inside the axes instead of a detached box below the plot.

Run from the repository root:  python scripts/make_figure6.py
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
       / "acl_rq3_qwen_quant_batch_acc_labeled_all_energy_vrdu.pdf")

BATCHES = [1, 5, 10, 20, 40, 60]
METRIC = "avg_field_em"
# The figure is built from the dedicated FP16-vs-FP8 repeat campaign only: three
# runs per (model, precision, batch size), all run together. These are kept out
# of results/run_reports/ so Figures 3-5 and the tables are unaffected.
RUNS_DIR = RESULTS / "fp8_repeats"
N_RUNS = 3
# The campaign used the vision arm's sampling (0.7 / 0.8) for both models, so the
# Qwen3-4B runs here differ in decoding from the text runs elsewhere in the paper
# (0.6 / 0.9). FP16 and FP8 are still decoded identically, which is what matters
# for this comparison; the pin makes any stray config fail the count check below.
TEMPERATURE = "0.7"

# (display label, model_name, task)
SERIES = [
    ("Qwen3-VL-4B\n(FP16)", "Qwen3-VL-4B-Instruct", "vrdu_vision", False),
    ("Qwen3-VL-4B\n(FP8)", "Qwen3-VL-4B-Instruct", "vrdu_vision", True),
    ("Qwen3-4B\n(FP16)", "Qwen3-4B", "vrdu_tesseract", False),
    ("Qwen3-4B\n(FP8)", "Qwen3-4B", "vrdu_tesseract", True),
]



def _check_precision_dir(path, backend):
    """A run belongs in a directory matching its precision.

    Misfiled files silently compete with the runs the paper reports -- an FP16
    Qwen3-VL-8B run filed under vrdu_vision_quant/ differed from its correctly
    filed twin by 59% in energy. Such files now live in superseded_runs/; this
    guard makes a recurrence fail loudly instead of changing a figure quietly.
    """
    top = Path(path).parent.name  # e.g. vrdu_quant / vrdu_noquant
    if ("quant" in backend) != top.endswith("_quant"):
        sys.exit(f"ERROR: {path} has backend={backend} but sits in {top}. "
                 f"Move it to the directory matching its precision, or to superseded_runs/.")

def load():
    """energy[series][bs] = [mWh/page, one per run] ; em[series] = mean exact match
    over all runs and batch sizes.

    Repeated runs of a cell are kept, not overwritten, so the figure shows their
    mean and -- once a cell has two or more runs -- their standard deviation.
    """
    energy = {s[0]: {} for s in SERIES}
    ems = {s[0]: [] for s in SERIES}
    for path in glob.glob(str(RUNS_DIR / "*" / "*.csv")):
        for row in csv.DictReader(open(path)):
            _check_precision_dir(path, row["backend"])
            quant = "quant" in row["backend"]
            for label, model, task, want_quant in SERIES:
                if row["model_name"] != model or row["task"] != task:
                    continue
                if quant != want_quant:
                    continue
                if row["temperature"] != TEMPERATURE:
                    continue
                bs = int(row["batch_size"]) if row["scenario"] == "batch" else 1
                energy[label].setdefault(bs, []).append(
                    float(row["total_energy_wh"]) / int(row["num_queries"])
                    / R.pages_per_doc("VRDU") * 1000)
                ems[label].append(float(row[METRIC]) * 100)
    return energy, {k: sum(v) / len(v) for k, v in ems.items() if v}


def mean_std(runs):
    """Mean and sample standard deviation; std is NaN for a single run."""
    a = np.asarray(runs, dtype=float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else np.nan)


runs, em = load()
for label, *_ in SERIES:
    for bs in BATCHES:
        n = len(runs[label].get(bs, []))
        if n != N_RUNS:
            sys.exit(f"ERROR: {label!r} BS={bs} has {n} runs in {RUNS_DIR}, expected {N_RUNS}.")
energy ={k: {bs: mean_std(v)[0] for bs, v in d.items()} for k, d in runs.items()}
spread = {k: {bs: mean_std(v)[1] for bs, v in d.items()} for k, d in runs.items()}
REPEATED = any(len(v) > 1 for d in runs.values() for v in d.values())

BLUES = plt.get_cmap("Blues")(np.linspace(0.30, 0.92, len(BATCHES)))

fig, ax = plt.subplots(figsize=(3.6, 2.4))
ax2 = ax.twinx()

group_w = 0.82
bar_w = group_w / len(BATCHES)
centers = np.arange(len(SERIES))

for j, bs in enumerate(BATCHES):
    xs = centers - group_w / 2 + bar_w * (j + 0.5)
    ys = [energy[s[0]].get(bs, np.nan) for s in SERIES]
    ax.bar(xs, ys, bar_w * 0.92, color=BLUES[j], edgecolor="black", linewidth=0.35,
           label=f"BS={bs}", zorder=3)
    # Error bars only where a cell has repeats; single runs have no spread to show.
    errs = [spread[s[0]].get(bs, np.nan) for s in SERIES]
    if REPEATED:
        ax.errorbar(xs, ys, yerr=np.nan_to_num(errs), fmt="none", ecolor="black",
                    elinewidth=0.5, capsize=1.2, capthick=0.5, zorder=4)

# Exact match as a marker per group, on the right axis.
acc = [em[s[0]] for s in SERIES]
ax2.plot(centers, acc, linestyle="none", marker="D", markersize=5.5,
         color="#b3202c", markeredgecolor="black", markeredgewidth=0.4,
         label="Exact match (%)", zorder=6)
for x, a in zip(centers, acc):
    ax2.annotate(f"{a:.1f}%", xy=(x, a), xytext=(0, 7), textcoords="offset points",
                 ha="center", fontsize=6.4, color="#b3202c", fontweight="bold", zorder=7)

ax.set_ylabel("Energy per page (mWh)", fontsize=7.5)
ax2.set_ylabel("Avg. field exact match (%)", fontsize=7.5, color="#b3202c")
ax.set_xticks(centers)
ax.set_xticklabels([s[0] for s in SERIES])
# Note: tick_params must come after set_xticklabels, otherwise it overrides the size.
ax.tick_params(axis="y", labelsize=6.8)
ax.tick_params(axis="x", labelsize=6.0)
ax2.tick_params(labelsize=6.8, colors="#b3202c")
ax.set_ylim(0, max(energy[k][b] + np.nan_to_num(spread[k][b])
                   for k in energy for b in energy[k]) * 1.34)
ax2.set_ylim(0, 100)
ax.grid(True, axis="y", linestyle=":", alpha=0.45, zorder=0)
ax.set_axisbelow(True)
# Open frame, as in Figures 3, 4 and 5. Dual-axis, so each y-axis keeps its own
# side -- energy on the left, exact match on the right -- and only the top goes.
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("top", "left"):
    ax2.spines[side].set_visible(False)
# The right spine now belongs solely to the exact-match axis, so it takes that
# axis's colour along with its ticks and label.
ax2.spines["right"].set_color("#b3202c")

# Compact legend inside the plot: the upper-right region is empty because the
# tall bars sit on the left. Saves the detached legend box below the figure.
handles, labels = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
leg = ax.legend(handles + h2, labels + l2, loc="upper right", ncol=2,
                fontsize=6.2, frameon=True, framealpha=0.92, borderpad=0.35,
                labelspacing=0.28, columnspacing=0.9, handlelength=1.2,
                handletextpad=0.45)
leg.get_frame().set_linewidth(0.4)
leg.set_zorder(8)

fig.tight_layout(pad=0.25)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

def cell(label, b):
    m, s = energy[label][b], spread[label][b]
    n = len(runs[label][b])
    return f"{m:.1f}" if n == 1 else f"{m:.1f}±{s:.1f}(n={n})"


print("  energy per page (mWh), BS " + " ".join(map(str, BATCHES)))
for label, *_ in SERIES:
    row = " ".join(cell(label, b) for b in BATCHES if b in energy[label])
    print(f"  {label.replace(chr(10),' '):22s} EM={em[label]:.1f}%  {row}")

# The numbers Section 5.3 quotes: FP16 vs FP8 at batch size 1 and at each
# precision's best (lowest-mean) batch size, with the combined standard error
# of the difference when both cells have repeats.
print("\n  FP16 vs FP8 (for Section 5.3 / Limitations)")
for fp16, fp8 in [(SERIES[0][0], SERIES[1][0]), (SERIES[2][0], SERIES[3][0])]:
    name = fp16.split("\n")[0]
    for tag, b16, b8 in [("BS 1", 1, 1),
                         ("best BS", min(energy[fp16], key=energy[fp16].get),
                          min(energy[fp8], key=energy[fp8].get))]:
        d = energy[fp8][b8] - energy[fp16][b16]
        se = np.sqrt(sum(spread[l][b] ** 2 / len(runs[l][b])
                         for l, b in [(fp16, b16), (fp8, b8)]))
        se_txt = "" if np.isnan(se) else f" ± {se:.2f} (SE)"
        print(f"  {name:12s} {tag:8s} FP16 {cell(fp16, b16)} @BS{b16}  "
              f"FP8 {cell(fp8, b8)} @BS{b8}  diff {d:+.2f}{se_txt} "
              f"({d / energy[fp16][b16] * 100:+.0f}%)")
