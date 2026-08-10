# Serving configuration

The commands used to serve each model. These reproduce Appendix A of the paper.

## Text-only models (vLLM)

```bash
vllm/vllm-openai:latest \
    --model "$MODEL" \
    --trust-remote-code \
    --max-model-len 31872 \
    --gpu-memory-utilization 0.95 \
    --port "$PORT"
```

Add `--quantization fp8` for the FP8 runs.

## Vision-language models (vLLM)

```bash
vllm/vllm-openai:latest \
    --model "$MODEL" \
    --trust-remote-code \
    --max-model-len 28000 \
    --gpu-memory-utilization 0.95 \
    --port "$PORT" \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image": 15}'
```

Add `--quantization fp8` for the FP8 runs.

## NuExtract 2.0 4B

```bash
vllm/vllm-openai:latest \
  --model numind/NuExtract-2.0-4B \
  --trust-remote-code \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --chat-template-content-format openai
```

## Arctic-TILT

`$BS` is the batch size.

```bash
python examples/tilt_example.py \
    --model Snowflake/snowflake-arctic-tilt-v1.3 \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --gpu-memory-utilization 0.8 \
    --subset "" \
    --limit-documents 500 \
    --async \
    --max-num-seqs $BS \
    --enforce-eager
```

## Decoding

Each model family uses its own recommended sampling defaults. Reasoning traces
are disabled for the Qwen3 models via
`chat_template_kwargs={"enable_thinking": false}`; the Mistral and Ministral
models do not expose such a mode.

| Arm | Dataset | temperature | top_p | max tokens |
|---|---|---|---|---|
| Text-only | Kleister-NDA | 0.6 | 0.9 | 128 |
| Text-only | VRDU | 0.6 | 0.9 | 256 |
| Vision | Kleister-NDA | 0.7 | 0.8 | 128 |
| Vision | VRDU | 0.7 | 0.8 | 128 |

Observed generation lengths averaged 7.4 to 25.9 tokens across every run, so no
configuration was limited by its token budget.

A separate set of text-only runs at the vision arm's settings (0.7 / 0.8) is
included in the results. Those are not used for any reported number; they exist
to bound the effect of the decoding difference between the two arms, which the
paper reports as -1.3 to +2.1 exact-match points.
