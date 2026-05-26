from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from zrt.training.compose.stage import _cost_phase_time, _resolve_compute_dtype
from zrt.training.models.flops import op_cost as _op_cost

if TYPE_CHECKING:
    from zrt.training.ir.training_graph import Graph, Op
    from zrt.training.models.flops import OpCost
    from zrt.training.spec.model import ModelSpec
    from zrt.training.spec.report import TrainingReport
    from zrt.training.spec.strategy import Strategy
    from zrt.training.spec.system import SystemSpec


MHC_OP_KINDS = frozenset({"hc_expand", "mhc_pre", "mhc_post", "mhc_head"})


@dataclass
class MHCMetric:
    count: int = 0
    fwd_ms: float = 0.0
    bwd_dx_ms: float = 0.0
    bwd_dw_ms: float = 0.0
    total_ms: float = 0.0
    fwd_flops: float = 0.0
    bwd_flops: float = 0.0
    total_flops: float = 0.0
    fwd_bytes: float = 0.0
    bwd_bytes: float = 0.0
    total_bytes: float = 0.0


@dataclass
class MHCShare:
    step_time: float = 0.0
    compute_time: float = 0.0
    recompute_time: float = 0.0


@dataclass
class MHCCounterfactual:
    hc_on_step_time_ms: float = 0.0
    hc_off_step_time_ms: float = 0.0
    delta_step_time_ms: float = 0.0
    delta_step_time_pct: float = 0.0
    hc_on_compute_time_ms: float = 0.0
    hc_off_compute_time_ms: float = 0.0
    delta_compute_time_ms: float = 0.0
    delta_compute_time_pct: float = 0.0
    hc_on_fwd_compute_ms: float = 0.0
    hc_off_fwd_compute_ms: float = 0.0
    hc_on_bwd_compute_ms: float = 0.0
    hc_off_bwd_compute_ms: float = 0.0
    hc_on_recompute_time_ms: float = 0.0
    hc_off_recompute_time_ms: float = 0.0
    hc_on_memory_gb: float | None = None
    hc_off_memory_gb: float | None = None
    hc_on_mfu: float = 0.0
    hc_off_mfu: float = 0.0
    hc_on_mfu_native: float = 0.0
    hc_off_mfu_native: float = 0.0
    hc_on_hfu: float = 0.0
    hc_off_hfu: float = 0.0
    error: str = ""


@dataclass
class MHCAnalysis:
    enabled: bool = False
    hc_mult: int = 1
    hc_sinkhorn_iters: int = 0
    op_counts: dict[str, int] = field(default_factory=dict)
    by_kind: dict[str, MHCMetric] = field(default_factory=dict)
    total: MHCMetric = field(default_factory=MHCMetric)
    shares: MHCShare = field(default_factory=MHCShare)
    counterfactual: MHCCounterfactual | None = None


def analyze_mhc(
    report: "TrainingReport",
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    op_costs: dict[str, "OpCost"] | None = None,
    include_counterfactual: bool = True,
) -> MHCAnalysis:
    """Aggregate DeepSeek-V4 HyperConnection costs for spec training reports."""
    by_kind: dict[str, MHCMetric] = {}
    total = MHCMetric()

    for op in graph.ops:
        if op.kind not in MHC_OP_KINDS:
            continue
        metric = by_kind.setdefault(op.kind, MHCMetric())
        _add_op(metric, op, model, system, op_costs)

    for metric in by_kind.values():
        _merge_metric(total, metric)

    op_counts = {kind: metric.count for kind, metric in by_kind.items()}
    shares = MHCShare(
        step_time=_safe_ratio(total.total_ms, getattr(report, "step_time_ms", 0.0)),
        compute_time=_safe_ratio(total.total_ms, getattr(report, "compute_time_ms", 0.0)),
        recompute_time=_safe_ratio(
            _mhc_recompute_ms(graph, model, system, strategy, op_costs),
            getattr(report, "recompute_time_ms", 0.0),
        ),
    )

    enabled = bool(getattr(model, "hc_mult", 1) > 1 and total.count > 0)
    counterfactual = None
    if include_counterfactual and getattr(model, "hc_mult", 1) > 1:
        counterfactual = _estimate_counterfactual(report, model, system, strategy)

    return MHCAnalysis(
        enabled=enabled,
        hc_mult=int(getattr(model, "hc_mult", 1)),
        hc_sinkhorn_iters=int(getattr(model, "hc_sinkhorn_iters", 0)),
        op_counts=op_counts,
        by_kind=by_kind,
        total=total,
        shares=shares,
        counterfactual=counterfactual,
    )


def _add_op(
    metric: MHCMetric,
    op: "Op",
    model: "ModelSpec",
    system: "SystemSpec",
    op_costs: dict[str, "OpCost"] | None,
) -> None:
    cost = op_costs.get(op.name) if op_costs else None
    if cost is None:
        cost = _op_cost(op, model, system)

    dtype = _resolve_compute_dtype(op, model)
    fwd_s = _cost_phase_time(cost, "fwd", system, system.gpu.name, dtype=dtype)
    dx_s = _cost_phase_time(cost, "dx", system, system.gpu.name, dtype=dtype)
    dw_s = _cost_phase_time(cost, "dw", system, system.gpu.name, dtype=dtype)

    fwd_flops = cost.fwd_cube_flops + cost.fwd_vector_flops
    bwd_flops = (
        cost.dx_cube_flops + cost.dx_vector_flops
        + cost.dw_cube_flops + cost.dw_vector_flops
    )
    fwd_bytes = cost.fwd_bytes
    bwd_bytes = cost.dx_bytes + cost.dw_bytes

    metric.count += 1
    metric.fwd_ms += fwd_s * 1000.0
    metric.bwd_dx_ms += dx_s * 1000.0
    metric.bwd_dw_ms += dw_s * 1000.0
    metric.total_ms += (fwd_s + dx_s + dw_s) * 1000.0
    metric.fwd_flops += fwd_flops
    metric.bwd_flops += bwd_flops
    metric.total_flops += fwd_flops + bwd_flops
    metric.fwd_bytes += fwd_bytes
    metric.bwd_bytes += bwd_bytes
    metric.total_bytes += fwd_bytes + bwd_bytes


def _merge_metric(total: MHCMetric, metric: MHCMetric) -> None:
    total.count += metric.count
    total.fwd_ms += metric.fwd_ms
    total.bwd_dx_ms += metric.bwd_dx_ms
    total.bwd_dw_ms += metric.bwd_dw_ms
    total.total_ms += metric.total_ms
    total.fwd_flops += metric.fwd_flops
    total.bwd_flops += metric.bwd_flops
    total.total_flops += metric.total_flops
    total.fwd_bytes += metric.fwd_bytes
    total.bwd_bytes += metric.bwd_bytes
    total.total_bytes += metric.total_bytes


def _mhc_recompute_ms(
    graph: "Graph",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
    op_costs: dict[str, "OpCost"] | None,
) -> float:
    cats_by_layer = getattr(getattr(strategy, "recompute", None), "per_layer", {}) or {}
    if not cats_by_layer:
        return 0.0

    total = 0.0
    for op in graph.ops:
        if op.kind not in MHC_OP_KINDS or op.layer_id < 0 or op.layer_id >= len(model.layers):
            continue
        layer_kind = model.layers[op.layer_id].value
        cats = cats_by_layer.get(layer_kind, set())
        if "full" not in cats and "hc" not in cats:
            continue
        metric = MHCMetric()
        _add_op(metric, op, model, system, op_costs)
        total += metric.fwd_ms
    return total


def _estimate_counterfactual(
    report: "TrainingReport",
    model: "ModelSpec",
    system: "SystemSpec",
    strategy: "Strategy",
) -> MHCCounterfactual:
    try:
        from zrt.training.ir.builders import build_graph
        from zrt.training.search.estimator import estimate

        off_model = deepcopy(model)
        off_model.hc_mult = 1
        off_graph = build_graph(off_model, strategy)
        off_report = estimate(off_model, system, strategy, graph=off_graph)
    except Exception as exc:  # pragma: no cover - defensive report path
        return MHCCounterfactual(error=str(exc))

    on_step = float(getattr(report, "step_time_ms", 0.0))
    off_step = float(getattr(off_report, "step_time_ms", 0.0))
    on_compute = float(getattr(report, "compute_time_ms", 0.0))
    off_compute = float(getattr(off_report, "compute_time_ms", 0.0))

    return MHCCounterfactual(
        hc_on_step_time_ms=on_step,
        hc_off_step_time_ms=off_step,
        delta_step_time_ms=on_step - off_step,
        delta_step_time_pct=_safe_ratio_pct(on_step - off_step, off_step),
        hc_on_compute_time_ms=on_compute,
        hc_off_compute_time_ms=off_compute,
        delta_compute_time_ms=on_compute - off_compute,
        delta_compute_time_pct=_safe_ratio_pct(on_compute - off_compute, off_compute),
        hc_on_fwd_compute_ms=float(getattr(report, "fwd_compute_ms", 0.0)),
        hc_off_fwd_compute_ms=float(getattr(off_report, "fwd_compute_ms", 0.0)),
        hc_on_bwd_compute_ms=float(getattr(report, "bwd_compute_ms", 0.0)),
        hc_off_bwd_compute_ms=float(getattr(off_report, "bwd_compute_ms", 0.0)),
        hc_on_recompute_time_ms=float(getattr(report, "recompute_time_ms", 0.0)),
        hc_off_recompute_time_ms=float(getattr(off_report, "recompute_time_ms", 0.0)),
        hc_on_memory_gb=_memory_gb(report),
        hc_off_memory_gb=_memory_gb(off_report),
        hc_on_mfu=float(getattr(report, "mfu", 0.0)),
        hc_off_mfu=float(getattr(off_report, "mfu", 0.0)),
        hc_on_mfu_native=float(getattr(report, "mfu_native", 0.0)),
        hc_off_mfu_native=float(getattr(off_report, "mfu_native", 0.0)),
        hc_on_hfu=float(getattr(report, "hfu", 0.0)),
        hc_off_hfu=float(getattr(off_report, "hfu", 0.0)),
    )


def _memory_gb(report: "TrainingReport") -> float | None:
    memory = getattr(report, "memory", None)
    if memory is None:
        return None
    total = getattr(memory, "total", None)
    return None if total is None else float(total) / 1e9


def _safe_ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _safe_ratio_pct(num: float, den: float) -> float:
    return _safe_ratio(num, den) * 100.0
