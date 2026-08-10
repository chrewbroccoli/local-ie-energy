"""Regenerate the end-to-end Pareto figure for the paper (figures/pareto_e2e_stacked.pdf).

Fixes relative to visualization_pareto.ipynb:
  * Kleister parsing energy is amortized over the 500 documents actually profiled,
    not over the 337 documents used for extraction (the notebook divided the
    500-document parsing total by 337, inflating Kleister OCR energy by 1.48x).
  * Arctic-TILT end-to-end energy comes from inference_energy_report_pareto.csv
    (full-length runs at bs=10, including the VRDU + DeepSeek-OCR 2 configuration
    that was previously unmeasured).
  * VRDU "Tesseract" points are the real `vrdu_tesseract` runs. The notebook's
    category function had a catch-all that labelled every task other than docling
    and deepseek as "Tesseract", so `vrdu_default` -- the OCR shipped with the
    VRDU benchmark -- was plotted as Tesseract while being charged Tesseract's
    energy. The dataset-provided OCR is excluded here: its production energy is
    not attributable to our pipeline.
  * Kleister vision rows average the two runs that exist per configuration.
  * Side-by-side panels sized for a two-column figure*, with model names matching
    the paper. All configurations are drawn at full strength -- the dominated
    points carry evidence of their own -- and the frontier is marked by enlarging
    those points, ringing them in black, and joining them with a step line.

Run from the repository root:  python visualizations/make_pareto_acl.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "figures" / "pareto_e2e_stacked.pdf"


import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import results as R

GROUP = {"tesseract": "tess", "docling": "doc", "deepseek": "ds"}


def build(dataset):
    """(label, group, end-to-end mWh/page, exact match) for every configuration.

    End-to-end = parsing energy per page + model inference per page.
    Configurations without a measured inference run are dropped rather than
    reported with parsing energy alone.
    """
    runs = R.load_runs()
    arctic, nuextract = R.load_arctic(), R.load_nuextract()
    pts = []

    for parser, suffix in GROUP.items():
        a = arctic.get((dataset, parser))
        if not a or a["inference_per_page"] is None:
            continue
        pts.append(("Arctic-TILT", f"arctic-{suffix}",
                    R.parser_energy_per_page(dataset, parser) + a["inference_per_page"], a["em"]))

    for model in R.TEXT_MODELS:
        for parser, suffix in GROUP.items():
            rec = R.select(runs, dataset, model, parser)
            if rec:
                pts.append((model, f"text-{suffix}",
                            R.parser_energy_per_page(dataset, parser) + rec["energy_per_page"],
                            rec["em"]))

    for model in R.VISION_MODELS:
        rec = R.select(runs, dataset, model, "vision")
        if rec:
            pts.append((model, "vision", rec["energy_per_page"], rec["em"]))

    n = nuextract.get(dataset)
    if n:
        pts.append(("NuExtract-2.0-4B", "nuextract", n["energy_per_page"], n["em"]))
    return pts


VRDU = build("VRDU")
KLEISTER = build("Kleister-NDA")

TEXT = "#3f8f6b"
VIS = "#e07a3f"
SPEC = "#7b52c4"

STYLE = {
    "text-tess": (TEXT, "o", "Text-only + Tesseract"),
    "text-doc": (TEXT, "D", "Text-only + Docling"),
    "text-ds": (TEXT, "s", "Text-only + DeepSeek-OCR 2"),
    "vision": (VIS, "^", "Vision-language (no parser)"),
    "arctic-tess": (SPEC, "o", "Arctic-TILT + Tesseract"),
    "arctic-doc": (SPEC, "D", "Arctic-TILT + Docling"),
    "arctic-ds": (SPEC, "s", "Arctic-TILT + DeepSeek-OCR 2"),
    "nuextract": (SPEC, "^", "NuExtract-2.0-4B"),
}


def pareto(points, exclude=()):
    """Points on the upper-left frontier: cheapest first, accuracy strictly increasing.

    `exclude` names groups that are ineligible for the frontier. Arctic-TILT is
    fine-tuned on Kleister-NDA by its authors, so on that dataset it is evaluated
    in-domain and must not be compared against the zero-shot models.
    """
    best, front = -1.0, []
    for p in sorted(points, key=lambda r: r[2]):
        if p[1] in exclude:
            continue
        if p[3] > best:
            front.append(p)
            best = p[3]
    return front


def panel(ax, points, title, label_offsets, xticks, exclude=(), label_coords="offset"):
    front = pareto(points, exclude)
    front_keys = {(p[2], p[3]) for p in front}

    # Every configuration is drawn at full strength -- the dominated points are
    # evidence too (e.g. the stranded DeepSeek-OCR cluster). The frontier is marked
    # by adding a ring and the step line, not by suppressing everything else.
    seen = set()
    for name, group, energy, acc in points:
        color, marker, legend = STYLE[group]
        on_front = (energy, acc) in front_keys
        ax.scatter(
            energy, acc,
            color=color, marker=marker,
            s=68 if on_front else 38,
            alpha=1.0,
            # Thin white outline keeps overlapping points in the dense clusters apart.
            edgecolors="black" if on_front else "white",
            linewidths=1.0 if on_front else 0.5,
            zorder=5 if on_front else 3,
            label=legend if legend not in seen else None,
        )
        seen.add(legend)

    ax.step([p[2] for p in front], [p[3] for p in front],
            where="post", color="#3b3b3b", linestyle="--", linewidth=1.2, zorder=4)

    # Disambiguate: the same model can sit on the frontier with two different parsers.
    repeated = {n for n in (p[0] for p in front)
                if sum(1 for q in front if q[0] == n) > 1}
    suffix = {"text-tess": "\n+ Tesseract", "text-doc": "\n+ Docling", "text-ds": "\n+ DeepSeek",
              "arctic-tess": "\n+ Tesseract", "arctic-doc": "\n+ Docling", "arctic-ds": "\n+ DeepSeek"}

    for name, group, energy, acc in front:
        text = name + (suffix.get(group, "") if name in repeated else "")
        dx, dy, ha = label_offsets.get((name, group), (6, 5, "left"))
        # Labels sit well clear of the markers; a hairline keeps the association
        # readable at that distance, which crowding the marker did not.
        ax.annotate(text, xy=(energy, acc), xytext=(dx, dy), ha=ha,
                    textcoords=("data" if label_coords == "data" else "offset points"),
                    va=("top" if label_coords == "data" else "baseline"),
                    fontsize=7.0, zorder=6, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", color="#8a8a85", lw=0.45,
                                    shrinkA=1.5, shrinkB=5))

    ax.set_xscale("log")
    # Explicit decimal ticks: matplotlib's default log minor labels collide at this size.
    ax.xaxis.set_major_locator(FixedLocator(xticks))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_minor_formatter(NullFormatter())
    # Extra head-room on both sides so annotations near the extremes are not clipped.
    lo = min(p[2] for p in points)
    hi = max(p[2] for p in points)
    ax.set_xlim(lo * 0.70, hi * 1.45)
    ax.set_xlabel("End-to-end energy per page (mWh, log scale)", fontsize=9)
    ax.set_ylabel("Avg. field exact match (%)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.grid(True, which="major", linestyle=":", alpha=0.45)
    ax.tick_params(labelsize=8)


fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

panel(axes[0], VRDU, "VRDU (scanned, layout-rich)", {
    ("Qwen3-0.6B", "text-tess"): (-4, -26, "center"),
    ("Qwen3-VL-2B", "vision"): (-16, 20, "center"),
    ("Qwen3-VL-4B", "vision"): (16, -74, "left"),
    ("NuExtract-2.0-4B", "nuextract"): (2, 20, "center"),
}, xticks=[5, 10, 20, 50, 100])
# Six frontier points sit in a narrow cluster here, so the labels go into the
# empty band below it and are placed in data coordinates (mWh/page), staggered so the
# leader lines stay short and do not cross.
panel(axes[1], KLEISTER, "Kleister-NDA (born-digital, text-heavy)", {
    # Label x-order matches point x-order and alternates between two rows, so
    # the leader lines fan out without crossing.
    ("Qwen3-0.6B", "text-doc"): (1.70, 46, "left"),
    ("Qwen3-1.7B", "text-doc"): (2.76, 30, "left"),
    ("Ministral-3-3B", "text-doc"): (4.48, 46, "left"),
    ("Qwen3-4B", "text-doc"): (7.28, 30, "left"),
    ("Ministral-3-3B", "text-tess"): (13.5, 46, "left"),
    ("Qwen3-4B", "text-tess"): (23.5, 30, "left"),
}, xticks=[2, 5, 10, 20, 50],
   exclude=("arctic-tess", "arctic-doc", "arctic-ds"), label_coords="data")

# Arctic-TILT is shown on the Kleister panel but excluded from the frontier:
# it is fine-tuned on Kleister-NDA by its authors, so it is not a zero-shot peer.
axes[1].annotate("Arctic-TILT: in-domain,\nexcluded from frontier",
                 xy=(8.28, 91.9), xytext=(-104, -2), textcoords="offset points",
                 fontsize=7.0, style="italic", color=SPEC, ha="left", va="center",
                 arrowprops=dict(arrowstyle="-", color=SPEC, lw=0.6,
                                 shrinkA=2, shrinkB=4))

handles, labels = axes[0].get_legend_handles_labels()
h2, l2 = axes[1].get_legend_handles_labels()
for h, l in zip(h2, l2):
    if l not in labels:
        handles.append(h)
        labels.append(l)
order = [
    "Text-only + Tesseract", "Text-only + Docling", "Text-only + DeepSeek-OCR 2",
    "Vision-language (no parser)", "Arctic-TILT + Tesseract", "Arctic-TILT + Docling",
    "Arctic-TILT + DeepSeek-OCR 2", "NuExtract-2.0-4B",
]
pairs = {l: h for h, l in zip(handles, labels)}
fig.legend([pairs[l] for l in order if l in pairs],
           [l for l in order if l in pairs],
           loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=4,
           fontsize=8, frameon=True)

fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight")
print(f"wrote {OUT}")

for name, pts, excl in (("VRDU", VRDU, ()),
                        ("Kleister-NDA", KLEISTER,
                         ("arctic-tess", "arctic-doc", "arctic-ds"))):
    print(f"\n{name} frontier (zero-shot models only):")
    for label, group, e, a in pareto(pts, excl):
        print(f"  {label:20s} {STYLE[group][2]:30s} {e:.1f} mWh/pg  {a:.1f} EM")
