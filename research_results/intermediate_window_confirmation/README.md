# 100-entity intermediate-window experiment

This package contains the code, raw outputs, validation records, full-cohort analysis, figures, and PDF report for the Qwen3-8B-Base intermediate-token entity-access experiment completed on 2026-09-24. The analysis includes all 100 entities from this run and does not exclude entities based on any earlier experiment.

## Terminology

Layer intervals are zero-indexed and half-open. For example, `[5, 7)` contains layers 5 and 6.

| Term | Definition |
|---|---|
| Intermediate-access window | A layer interval in which intermediate prompt tokens retain their native entity-value contribution. |
| `W_early = [5, 7)` | Prespecified early candidate, layers 5-6. |
| `W_middle = [12, 14)` | Prespecified middle candidate, layers 12-13. |
| `W_late-mid = [19, 21)` | Prespecified later-middle candidate, layers 19-20. |
| `W_union` | Sparse union `{5, 6, 12, 13, 19, 20}`. |
| `W_control = [30, 32)` | Matched late negative control, layers 30-31. |
| Full intermediate bottleneck | Entity-value contributions are removed for every intermediate prompt-token query in all 36 layers. |
| Rescue | Incorrect under the full bottleneck and correct under a candidate condition. |

The intervention preserves native attention scores and post-softmax weights. It zeroes the entity-span value contribution for targeted queries without renormalizing the remaining attention coefficients. The final prompt query stays native. In every non-clean condition from this run, generated-token queries are also prevented from rereading the entity during cached decoding.

## Analysis population

The run contains 100 entities, five templates, eight conditions, and 4,000 successful rows. All 100 entities are analyzed:

- 50 one-token names per template form the primary prompt-stage window analysis.
- 50 multi-token names per template expose a separate cached-decoding requirement and are reported as a complementary stratum.

The token-count strata are kept separate because generated-token blocking can truncate multi-token names after a correct first token. This separation is a property of the present experiment, not an exclusion based on previous data.

## Main results on all 50 one-token entities

- `friend`: `W_early` rescued 36 of 37 full-bottleneck failures, harmed 0 cases, and reached 0.980 semantic accuracy (`p = 2.91e-11`). `W_union` rescued all 37 failures.
- `person_in_list`: `W_late-mid` rescued 6 of 11 failures with no harm (`p = 0.0313`). `W_union` rescued 10 of 11 failures (`p = 0.0020`).
- `dialogue`: `W_union` rescued 24 of 31 failures with no harm (`p = 1.19e-07`).
- `direct_fact` and `visitor_register`: the full bottleneck did not reduce complete-identity accuracy, so they are insensitive semantic controls rather than rescue tests.
- `W_control` remained near the full bottleneck and recovered essentially none of the late entity-readout gap.

Successful conditions also restored the final token's late entity-directed attention and projected value-contribution norm. This supports a mechanism in which selected early and middle intermediate-token computations configure a later direct readout from the original entity.

## Multi-token boundary

Under generated-token blocking, the first answer token was correct for all 50 multi-token entities in every template, while complete-identity accuracy ranged from 0.16 to 0.62. The common failure was generating only the first name. These cases remain part of the 100-entity experiment but should be reported separately from prompt-stage window rescue because later name tokens often require entity rereading during autoregressive decoding.

## Reproduce the experiment

From the repository root on a GPU machine:

```bash
CUDA_VISIBLE_DEVICES=0 PHYSICAL_GPU_ID=0 \
uv run --extra gpu python scripts/inspect/qwen_intermediate_window_confirmation.py \
  --out-dir results/relay-qwen3-8b/intermediate-window-confirmation
```

The portable runner defaults to `scripts/inspect/data/entities100.json` and `scripts/inspect/qwen_entity_attention_ablation_span.py`. The exact source files used for the completed run are preserved in `source/` with hashes recorded in `raw/source_hashes.json`.

Run the lightweight checks without loading the model:

```bash
uv run --extra gpu python scripts/inspect/qwen_intermediate_window_confirmation.py --self-test
```

Rebuild the original aggregate tables and plots from an existing result directory:

```bash
uv run --extra gpu python scripts/inspect/qwen_intermediate_window_confirmation.py \
  --out-dir results/relay-qwen3-8b/intermediate-window-confirmation \
  --analysis-only
```

Rebuild the corrected report from this package with Python packages `pandas`, `matplotlib`, `reportlab`, and `pypdf` installed:

```bash
python research_results/intermediate_window_confirmation/build_summary.py \
  --input research_results/intermediate_window_confirmation/raw \
  --output-dir research_results/intermediate_window_confirmation/report
```

## Package layout

- `source/`: immutable copies of the three experiment inputs used on Drexel.
- `raw/results.jsonl`: all 4,000 trial rows.
- `raw/`: configuration, validations, run metadata, original aggregate tables, and original run summary.
- `derived/`: full-cohort tables for all 50 one-token and all 50 multi-token entities.
- `report/`: the reviewed PDF and its five source figures.
- `build_summary.py`: reproducible analysis and PDF builder.

The original aggregate summary is retained as `raw/original_summary.md` for provenance. The PDF and `derived/` tables provide the paper-facing interpretation of all 100 entities while separating prompt-stage one-token effects from multi-token cached decoding.
