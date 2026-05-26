# MHC Training Spec Analysis Design

## Scope

Add HyperConnection (MHC) analysis to the spec-based training estimation path only.

In scope:

- `--estimate-config` / spec training estimator reports.
- Excel report output.
- Lightweight fields for training search summaries and best-config Excel exports.
- DeepSeek V4-style MHC ops: `hc_expand`, `mhc_pre`, `mhc_post`, `mhc_head`.

Out of scope:

- HTML report changes.
- Graph-native training report changes.
- Inference throughput or serving analysis.
- Model-accuracy or quality impact.

## Goal

Make it easy to answer, from the final training report:

- How much training time does MHC consume?
- Which MHC op kind dominates?
- How does the configuration compare with an otherwise identical `hc_mult=1` baseline?
- Does MHC materially affect step time, compute time, memory, MFU, or HFU?

## Architecture

Add a small reusable analysis module:

`python/zrt/training/analysis/mhc.py`

The module should expose one public function:

`analyze_mhc(report, graph, model, system, strategy, op_costs=None, include_counterfactual=True)`

It returns a structured dataclass or dict with:

- `enabled`
- `hc_mult`
- `hc_sinkhorn_iters`
- `op_counts`
- `by_kind`
- `total`
- `shares`
- `counterfactual`

The analysis module owns MHC classification and aggregation. Exporters and search code should consume its result rather than reimplementing op-kind checks.

## Metrics

Classify these op kinds as MHC:

- `hc_expand`
- `mhc_pre`
- `mhc_post`
- `mhc_head`

For each kind, aggregate:

- op count
- forward time
- backward dx time
- backward dw time
- total time
- forward FLOPs
- backward FLOPs
- total FLOPs
- forward bytes
- backward bytes
- total bytes

Use the same timing path as the Excel Ops sheet so numbers match the existing report:

- `op_cost(op, model, system)`
- `_resolve_compute_dtype(op, model)`
- `_cost_phase_time(...)`

Shares should include:

- MHC total time / report step time.
- MHC total time / report compute time.
- MHC recompute time / report recompute time, when recompute includes the `hc` category.

## Counterfactual

When `include_counterfactual=True` and `model.hc_mult > 1`, clone the model, set `hc_mult=1`, rebuild and re-estimate the graph with the same system and strategy.

Report the following on/off comparison:

- `step_time_ms`
- `compute_time_ms`
- `fwd_compute_ms`
- `bwd_compute_ms`
- `recompute_time_ms`
- `memory_gb`, when available
- `mfu`
- `mfu_native`
- `hfu`
- deltas for step time and compute time

The counterfactual must not mutate the original `ModelSpec`.

If the counterfactual estimate fails, return an error string in the analysis result and still emit the single-run MHC breakdown.

## Excel Report

Add one new sheet to `export_estimate_excel`:

`MHC Analysis`

Suggested sections:

1. Summary
   - enabled
   - hc_mult
   - sinkhorn iters
   - total MHC time
   - MHC share of step time
   - MHC share of compute time

2. By Kind
   - kind
   - count
   - fwd ms
   - bwd dx ms
   - bwd dw ms
   - total ms
   - total FLOPs
   - total bytes

3. Counterfactual
   - metric
   - HC on
   - HC off
   - delta
   - delta %

Do not change the HTML exporter in this iteration.

## Training Search

Add lightweight MHC columns to search summaries after each successful report:

- `mhc_enabled`
- `mhc_total_ms`
- `mhc_step_share`
- `mhc_compute_share`
- `mhc_delta_step_ms`
- `mhc_delta_step_pct`

Search should not emit the full per-kind breakdown in the CSV. The best-config Excel export will include the full `MHC Analysis` sheet through `export_estimate_excel`.

## Error Handling

- If no MHC ops exist, emit an enabled=false style result with zero totals.
- If `report.step_time_ms` or `report.compute_time_ms` is zero, shares should be zero rather than raising.
- If `op_costs` is incomplete, compute missing costs locally.
- Counterfactual failures should not fail the main report export.

## Tests

Add focused tests for:

- `hc_mult=1` returns disabled/zero MHC totals.
- `hc_mult=4` finds MHC op kinds and nonzero totals.
- counterfactual analysis does not mutate the original model.
- Excel workbook contains `MHC Analysis`.
- search summary rows include the lightweight MHC columns.

## Non-Goals

This change does not decide whether MHC should be enabled for a production training run. It only makes its cost visible and comparable in the current analytical estimator.
