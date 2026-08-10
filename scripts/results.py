"""Shared result loader.

Every selector the paper depends on lives here and nowhere else: which batch
size, which decoding configuration, which metric column, how energy is
normalized. Each figure and table script imports from this module, so a figure
and a table cannot disagree about what they are showing.

Energy unit. Everything is reported in **mWh per page**, normalized by
``pages_per_doc``. Per-document energy would conflate cost per page with
document length, and Kleister-NDA documents are about three times longer than
VRDU ones, so the two datasets would not be comparable on a per-document axis.
Table 2 is the one deliberate exception -- see ``pages_per_doc``.

Two guards are built in, both from defects that actually occurred:

* ``_check_precision_dir`` -- a run whose backend disagrees with the directory it
  sits in aborts the load. An unquantized Qwen3-VL-8B run once sat in a quantized
  directory and reported 0.0412 Wh/doc against 0.0657 for its correctly filed
  twin, changing a figure depending on glob order.
* duplicate configurations are **averaged**, never resolved by whichever file the
  filesystem happened to yield last.

Metric note. ``avg_calculated_field_em`` is the post-hoc recomputation applied
uniformly across every system, and is the only field-level exact match the
Arctic-TILT and NuExtract pipelines emit; it is what Tables 3-4 and Figures 3, 4,
5 and 7 report. Figure 6 is the one exception and uses Bench360's own
``avg_field_em``, because the FP8 runs do not carry the recomputed column.
"""

from __future__ import annotations

import csv
import glob
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

# --- pinned selectors -------------------------------------------------------

METRIC = "avg_calculated_field_em"          # see the metric note above
METRIC_FUZZY = "avg_calculated_fuzzy_score"
METRIC_F1 = "avg_calculated_document_f1"

TABLE_BATCH = 10                            # batch size reported in Tables 3-4
TEMP_TEXT = "0.6"                           # text-only decoding (top_p 0.9)
TEMP_VISION = "0.7"                         # vision-language decoding (top_p 0.8)
BATCHES = [1, 5, 10, 20, 40, 60]

# Documents the extraction runs cover. VRDU uses 500; Kleister-NDA uses the 337
# documents the run reports record. Parsing energy is profiled over 500 documents
# for both datasets and is amortized over that 500, not over 337.
N_EVAL = {"VRDU": 500, "Kleister-NDA": 337}
N_PARSED = 500

# Total parsing energy in Wh over the 500 profiled documents, from
# results/ocr_profiling/*.csv (kWh in the files).
PARSER_TOTAL_WH = {
    "VRDU": {"tesseract": 4.867, "docling": 12.947, "deepseek": 76.409},
    "Kleister-NDA": {"tesseract": 12.909, "docling": 4.385, "deepseek": 220.488},
}


def pages_per_doc(dataset: str) -> float:
    """Canonical pages per document -- the denominator for every energy figure.

    Kleister-NDA documents are about three times longer than VRDU ones (5.87
    against 1.83 pages), so per-document energy conflates cost per page with
    document length and the two datasets cannot be compared on it. Everything
    the paper reports is therefore per page.

    Docling and DeepSeek-OCR 2 follow the PDF page structure; Tesseract
    rasterizes and reports ~3% more (3016/1025 against 2934/915). We take the
    PDF page count as canonical so that one constant per dataset serves every
    figure and table. Table 2 is the deliberate exception: it reports each
    parser against its own page count, because the parsers disagree about
    segmentation and that disagreement is the point of the table.

    Note that this is a dataset-level average, not a per-document page count --
    no per-document page vector exists in the measurement data. Dividing by it
    is a constant rescale within a dataset, so every within-dataset comparison
    (batching, quantization, Pareto membership, model ranking) is unchanged.
    """
    return parser_page_counts()[(dataset, "docling")]["pages"] / N_PARSED

DISPLAY = {
    "Llama-3.2-1B-Instruct": "Llama-3.2-1B", "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
    "Ministral-3-3B-Instruct-2512": "Ministral-3-3B", "Mistral-7B-Instruct-v0.3": "Mistral-7B",
    "Qwen3-0.6B": "Qwen3-0.6B", "Qwen3-1.7B": "Qwen3-1.7B",
    "Qwen3-4B": "Qwen3-4B", "Qwen3-8B": "Qwen3-8B",
    "Qwen3-VL-2B-Instruct": "Qwen3-VL-2B", "Qwen3-VL-4B-Instruct": "Qwen3-VL-4B",
    "Qwen3-VL-8B-Instruct": "Qwen3-VL-8B",
}
TEXT_MODELS = ["Llama-3.2-1B", "Llama-3.2-3B", "Ministral-3-3B", "Mistral-7B",
               "Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B"]
VISION_MODELS = ["Qwen3-VL-2B", "Qwen3-VL-4B", "Qwen3-VL-8B"]

PARSERS = ["default", "deepseek", "docling", "tesseract"]
PARSER_LABEL = {"default": "Baseline (Dataset)", "deepseek": "DeepSeek",
                "docling": "Docling", "tesseract": "Tesseract"}


def _check_precision_dir(path: Path, backend: str) -> None:
    top = path.parent.name
    if ("quant" in backend) != top.endswith("_quant"):
        sys.exit(
            f"ERROR: {path} has backend={backend} but sits in {top}.\n"
            f"A run must live in a directory matching its precision. Move it, or "
            f"put it in results/superseded/ if it is not part of the reported set."
        )


def _dataset(task: str) -> str:
    return "VRDU" if task.startswith("vrdu") else "Kleister-NDA"


def load_runs(quantized: bool = False, metric: str = METRIC):
    """All run reports as records, duplicates averaged.

    Returns a dict keyed by (dataset, display_model, task, batch_size) with
    ``em``, ``fuzzy``, ``f1``, ``energy_per_page``, ``temperature``, ``n_runs``.
    Energy is in mWh per page; see ``pages_per_doc``. Rows missing the requested
    metric are skipped.
    """
    buckets = defaultdict(list)
    for p in sorted(RESULTS.glob("run_reports/*/*.csv")):
        for row in csv.DictReader(open(p)):
            _check_precision_dir(p, row["backend"])
            if ("quant" in row["backend"]) != quantized:
                continue
            model = DISPLAY.get(row["model_name"])
            if not model or not row.get(metric):
                continue
            bs = int(row["batch_size"]) if row["scenario"] == "batch" else 1
            ds = _dataset(row["task"])
            buckets[(ds, model, row["task"], bs)].append({
                "em": float(row[metric]) * 100,
                "fuzzy": float(row[METRIC_FUZZY]) * 100 if row.get(METRIC_FUZZY) else None,
                "f1": float(row[METRIC_F1]) * 100 if row.get(METRIC_F1) else None,
                "energy_per_page": (float(row["total_energy_wh"]) / int(row["num_queries"])
                                    / pages_per_doc(ds) * 1000),
                "temperature": row["temperature"],
            })

    out = {}
    for key, runs in buckets.items():
        def avg(field):
            vals = [r[field] for r in runs if r[field] is not None]
            return sum(vals) / len(vals) if vals else None
        out[key] = {"em": avg("em"), "fuzzy": avg("fuzzy"), "f1": avg("f1"),
                    "energy_per_page": avg("energy_per_page"),
                    "temperature": runs[0]["temperature"], "n_runs": len(runs)}
    return out


def select(runs, dataset, model, source, batch=TABLE_BATCH, temperature=None):
    """One record, with the decoding configuration pinned by arm."""
    task = ("vrdu_" if dataset == "VRDU" else "kleister_nda_") + source
    if temperature is None:
        temperature = TEMP_VISION if source == "vision" else TEMP_TEXT
    rec = runs.get((dataset, model, task, batch))
    if rec is None or rec["temperature"] != temperature:
        return None
    return rec


def parser_energy_per_page(dataset: str, source: str) -> float:
    """Parsing energy in mWh per page, over the 500 profiled documents."""
    if source not in PARSER_TOTAL_WH[dataset]:
        return 0.0
    return PARSER_TOTAL_WH[dataset][source] / N_PARSED / pages_per_doc(dataset) * 1000


def _read_macro(path: Path, name: str):
    """Pull one Macro row out of an Arctic-TILT evaluation file."""
    for row in csv.reader(open(path)):
        if len(row) > 1 and row[1].strip() == name:
            for cell in reversed(row):
                cell = cell.strip()
                if cell and cell.replace(".", "", 1).isdigit():
                    return float(cell) * 100
    return None


def load_arctic():
    """Arctic-TILT accuracy per parser, plus inference energy in mWh per page."""
    acc = {}
    for p in sorted(RESULTS.glob("arctic_tilt/evaluation_*.csv")):
        _, ds_key, parser = p.stem.split("_", 2)
        ds = "VRDU" if ds_key == "vrdu" else "Kleister-NDA"
        parser = "default" if parser == "baseline" else parser
        acc[(ds, parser)] = {
            "em": _read_macro(p, "avg_calculated_field_em"),
            "fuzzy": _read_macro(p, "avg_calculated_fuzzy_score"),
            "f1": _read_macro(p, "avg_calculated_document_f1"),
        }

    energy = {}
    pareto = RESULTS / "arctic_tilt" / "inference_energy_report_pareto.csv"
    if pareto.exists():
        for row in csv.DictReader(open(pareto)):
            name = row["dataset_name"].strip("./").lower()
            ds = "VRDU" if "vrdu" in name else "Kleister-NDA"
            parser = ("deepseek" if "deepseek" in name else
                      "docling" if "docling" in name else "tesseract")
            energy[(ds, parser)] = (float(row["energy_consumed"]) * 1000 / N_EVAL[ds]
                                    / pages_per_doc(ds) * 1000)

    for key, rec in acc.items():
        rec["inference_per_page"] = energy.get(key)
    return acc


def load_nuextract():
    """NuExtract runs its own pipeline at batch size 1."""
    out = {}
    files = {"VRDU": RESULTS / "nuextract" / "nuextract_vrdu_metrics.csv",
             "Kleister-NDA": RESULTS / "nuextract" / "kleister_standalone_metrics_output.csv"}
    for ds, p in files.items():
        if not p.exists():
            continue
        row = next(csv.DictReader(open(p)))
        g = lambda k: float(row[k]) * 100 if row.get(k) else None
        out[ds] = {"em": g("avg_calculated_field_em"),
                   "fuzzy": g("avg_calculated_fuzzy_score"),
                   "f1": g("avg_calculated_document_f1"),
                   "energy_per_page": (float(row["total_energy_wh"]) / int(row["num_queries"])
                                       / pages_per_doc(ds) * 1000)}
    return out


@lru_cache(maxsize=1)
def parser_page_counts():
    """Pages each parser reported, for Table 2."""
    out = {}
    for p in sorted(RESULTS.glob("ocr_profiling/*.csv")):
        tool = ("tesseract" if "tesseract" in p.stem else
                "docling" if "docling" in p.stem else "deepseek")
        ds = "VRDU" if "vrdu" in p.stem else "Kleister-NDA"
        rows = list(csv.reader(open(p)))
        if rows and rows[0][0] == "Metric":
            d = {r[0]: r[1] for r in rows[1:] if len(r) > 1}
            pages = int(d["Total Pages Processed"])
            kwh = float(d["Energy Consumed (kWh)"])
        else:  # single wide header row
            d = dict(zip(rows[0], rows[1]))
            pages = int(d["Total Pages Processed"])
            kwh = float(d["Total Energy Consumed (kWh)"])
        out[(ds, tool)] = {"pages": pages, "total_wh": kwh * 1000}
    return out


def report_duplicates(quantized: bool = False) -> None:
    """Print configurations backed by more than one run (these are averaged)."""
    buckets = defaultdict(int)
    for p in sorted(RESULTS.glob("run_reports/*/*.csv")):
        for row in csv.DictReader(open(p)):
            if ("quant" in row["backend"]) != quantized:
                continue
            bs = row["batch_size"] if row["scenario"] == "batch" else "1"
            buckets[(row["model_name"], row["task"], bs, row["temperature"])] += 1
    dups = {k: v for k, v in buckets.items() if v > 1}
    if dups:
        print(f"  note: {len(dups)} configuration(s) have repeated runs; values are averaged")
        for k, v in sorted(dups.items()):
            print(f"        {v}x  {k[0]} / {k[1]} / bs={k[2]}")
