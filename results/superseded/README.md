# Superseded / misfiled run reports

These files were moved out of the active result directories because they were
being picked up by `*/run_report/*.csv` globs and silently competing with the
runs the paper reports. Nothing here is deleted — move a file back if it turns
out to be needed.

## `vrdu_vision_quant/` — unquantized runs filed under a quantized directory
Three runs with `backend: vllm-vl` (i.e. FP16) were sitting in
`vrdu_vision_quant/run_report/`. Each duplicates a configuration that already
exists in `vrdu_vision_noquant/run_report/`:

| config | here | in the noquant directory |
|---|---|---|
| Qwen3-VL-2B, BS 20 | 0.0131 Wh/doc | 0.0128 |
| Qwen3-VL-4B, BS 20 | 0.0196 Wh/doc | 0.0196 |
| Qwen3-VL-8B, BS 20 | **0.0412 Wh/doc** | **0.0657** |

The Qwen3-VL-8B pair mattered: 0.0412 is an outlier against its own batch-size
series (BS 10 = 0.0635, BS 40 = 0.0646, BS 60 = 0.0666), whereas 0.0657 fits it.
Both runs are complete and internally consistent (power x time reproduces the
reported energy in each), so this is a throughput difference, not a measurement
error — but the run in the correct directory is the one the paper uses.

## `kleister_nda_vl_quant/` — a different backend variant
Four runs with `backend: vllm-vl-test`, filed under a quantized directory while
being neither the quantized nor the standard `vllm-vl` configuration. One is
InternVL3_5-4B, a model that does not appear in the paper. None of them feed any
reported number.

## `vrdu_vision_noquant/` — orphan outside `run_report/`
One CSV sat at the top level of `vrdu_vision_noquant/` rather than in its
`run_report/` subdirectory, so no glob ever read it. It is an older-schema run
(43 columns, with the `avg_calculated_*` fields) of Qwen3-VL-8B at BS 20, and it
disagrees with both files above (35.06 Wh total against 32.84).
