# The Right Information Extraction Pipeline Depends on the Document: Accuracy--Energy Trade-offs for Small, Local Models

Configurations, measured results, and the scripts that regenerate every figure
and table in the paper.

```bash
pip install -r requirements.txt
python scripts/make_all.py
```

## Layout

```
configs/      experiment configurations, plus the serving commands (serving.md)
results/
  run_reports/    per-run benchmark reports, one CSV per (model, task, batch size)
  fp8_repeats/    three repeated FP16/FP8 runs per cell for Figure 6
  arctic_tilt/    Arctic-TILT accuracy and energy
  nuextract/      NuExtract accuracy and energy
  ocr_profiling/  parser energy and page counts
  superseded/     runs excluded from the reported set, with the reason
scripts/      results.py (shared loader) and one script per artifact
```

## Energy accounting

GPU power is sampled through NVML at 10 Hz by both
instruments (CodeCarbon for parsing and the specialized models, Bench360 for the
vLLM-served models); CPU and RAM contributions are modeled rather than measured.
Model loading is excluded throughout, so the figures describe steady-state
serving rather than cold start. Geographic fields have been removed from the
CodeCarbon outputs.

## Experiment code

This branch holds what the paper reports: configurations, measured results, and
the scripts that turn them into figures and tables. The code that produced the
runs -- the Bench360 information-extraction tasks, the OCR profiling scripts,
and the NuExtract and Arctic-TILT inference pipelines -- is kept with its full
development history on the
[`bench360-dev`](https://github.com/chrewbroccoli/local-ie-energy/tree/bench360-dev)
branch.

## Citation

Christoph Walser, Mauricio Fadel Argerich, and Jonathan Fürst. 2026.
*The Right Information Extraction Pipeline Depends on the Document:
Accuracy--Energy Trade-offs for Small, Local Models.*

## License

MIT.
