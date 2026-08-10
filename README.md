# Reproducibility package

Configurations, measured results, and the scripts that regenerate every figure
and table in the paper.

This package is **self-contained**: it needs no GPU, no model weights, and no
dataset. It replays the measurements we recorded and rebuilds the paper's
artifacts from them.

```bash
pip install -r requirements.txt
python scripts/make_all.py
```

Figures land in `figures/`, LaTeX tables in `tables/`.

## What produces what

| Paper artifact | Script |
|---|---|
| Table 2 — parsing energy | `scripts/make_tables.py` |
| Table 3 — VRDU detailed results | `scripts/make_tables.py` |
| Table 4 — Kleister-NDA detailed results | `scripts/make_tables.py` |
| Figure 3 — RQ1 baselines | `scripts/make_figure3.py` |
| Figure 4 — RQ2 input representation | `scripts/make_figure4.py` |
| Figure 5 — RQ3 batch size | `scripts/make_figure5.py` |
| Figure 6 — RQ3 quantization | `scripts/make_figure6.py` |
| Figure 7 — Pareto frontier | `scripts/make_figure7.py` |

Figures 1 and 2 are a schematic and two document screenshots; neither is derived
from measurements, so neither is generated here.

## Layout

```
configs/      experiment configurations, plus the serving commands (serving.md)
results/
  run_reports/    per-run benchmark reports, one CSV per (model, task, batch size)
  arctic_tilt/    Arctic-TILT accuracy and energy
  nuextract/      NuExtract accuracy and energy
  ocr_profiling/  parser energy and page counts
  superseded/     runs excluded from the reported set, with the reason
scripts/      results.py (shared loader) and one script per artifact
```

## How results are selected

Every selector lives in `scripts/results.py` and nowhere else, so a figure and a
table cannot disagree about what they show:

- **Batch size 10** for the detailed tables and the Pareto frontier; Figure 3
  uses single-request inference and Figure 5 sweeps 1–60.
- **Decoding** is pinned per arm — 0.6/0.9 for text-only models, 0.7/0.8 for
  vision-language models — matching each family's recommended defaults.
- **Metric** is `avg_calculated_field_em`, the post-hoc recomputation applied
  uniformly across every system and the only field-level exact match the
  Arctic-TILT and NuExtract pipelines emit.
- **Energy per document** divides inference energy by the number of documents
  evaluated (500 for VRDU, 337 for Kleister-NDA) and adds parsing energy
  amortized over the 500 documents of the parsing benchmark.

Two guards are built into the loader, both from mistakes that actually occurred
during this work:

1. A run whose backend disagrees with the directory it sits in **aborts** the
   build, naming the file. An unquantized run once sat in a quantized directory
   and reported 0.0412 Wh/doc against 0.0657 for its correctly filed twin.
2. Configurations backed by more than one run are **averaged**, never resolved by
   whichever file the filesystem yielded last. `make_all.py` prints which
   configurations these are.

## Caveats worth knowing

**Figure 6 uses a different metric.** The FP8 run reports do not carry
`avg_calculated_field_em`, and they do not store predictions, so it cannot be
recomputed. That figure therefore uses Bench360's own `avg_field_em`. Absolute
values sit about two points below the tables; the FP16-versus-FP8 difference,
which is what the figure is about, is unaffected.

**Two configurations have repeated runs.** They agree within 0.9 and 2.2
exact-match points and within 0.7% and 8.8% energy. This is the only noise floor
we can quote — the study does not otherwise repeat measurements.

**`results/superseded/` is not part of the reported set.** It holds eight run
files that were misfiled — unquantized runs under a quantized directory, a
backend variant that no reported number uses, and one orphan outside the run
report tree. They are kept rather than deleted so the exclusion is auditable;
`results/superseded/README.md` explains each.

**Energy accounting.** GPU power is sampled through NVML at 10 Hz by both
instruments (CodeCarbon for parsing and the specialized models, Bench360 for the
vLLM-served models); CPU and RAM contributions are modeled rather than measured.
Model loading is excluded throughout, so the figures describe steady-state
serving rather than cold start. Geographic fields have been removed from the
CodeCarbon outputs for anonymity.

## License

MIT. The copyright line is anonymized for double-blind review and will name the
authors in the camera-ready release.
