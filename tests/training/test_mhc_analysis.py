from __future__ import annotations

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.analysis.mhc import MHC_OP_KINDS, analyze_mhc
from zrt.training.ir.builders import build_graph
from zrt.training.search.estimator import estimate
from zrt.training.spec.dtype import Dtype
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.strategy import Strategy
from zrt.training.spec.system import GPU, SystemSpec


def _system() -> SystemSpec:
    link = LinkSpec(type="test", bandwidth_gbps=1000, latency_us=1)
    return SystemSpec(
        gpu=GPU(
            name="test-gpu",
            flops_bf16=100,
            flops_fp8=200,
            flops_fp4=400,
            hbm_gb=80,
            hbm_bw_gbps=3000,
        ),
        host_mem_gb=1024,
        interconnect=InterconnectSpec(intra_node=link, inter_node=link),
        nodes=1,
        gpus_per_node=1,
    )


def _model(hc_mult: int) -> ModelSpec:
    return ModelSpec(
        hidden=128,
        ffn=256,
        num_heads=4,
        num_kv_heads=1,
        head_dim=32,
        vocab=1024,
        seq_len=64,
        layers=[LayerKind.MOE],
        q_lora_rank=32,
        qk_rope_head_dim=8,
        o_lora_rank=32,
        o_groups=4,
        compress_ratios=[4],
        swa_window=16,
        index_n_heads=4,
        index_head_dim=16,
        index_topk=8,
        num_experts=8,
        moe_ffn=64,
        top_k=2,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=3,
        act_dtype=Dtype.BF16,
    )


def test_analyze_mhc_reports_zero_when_hc_disabled():
    model = _model(hc_mult=1)
    system = _system()
    strategy = Strategy()
    graph = build_graph(model, strategy)
    report = estimate(model, system, strategy, graph=graph)

    analysis = analyze_mhc(
        report, graph, model, system, strategy, include_counterfactual=False
    )

    assert analysis.enabled is False
    assert analysis.total.total_ms == 0
    assert analysis.op_counts == {}
    assert analysis.counterfactual is None


def test_analyze_mhc_aggregates_enabled_hc_ops_and_counterfactual():
    model = _model(hc_mult=4)
    original_hc_mult = model.hc_mult
    system = _system()
    strategy = Strategy()
    graph = build_graph(model, strategy)
    report = estimate(model, system, strategy, graph=graph)

    analysis = analyze_mhc(report, graph, model, system, strategy)

    assert model.hc_mult == original_hc_mult
    assert analysis.enabled is True
    assert set(analysis.op_counts).issubset(MHC_OP_KINDS)
    assert analysis.op_counts["mhc_pre"] > 0
    assert analysis.op_counts["mhc_post"] > 0
    assert analysis.op_counts["mhc_head"] == 1
    assert analysis.total.total_ms > 0
    assert analysis.total.total_flops > 0
    assert analysis.total.total_bytes > 0
    assert analysis.shares.step_time > 0
    assert analysis.counterfactual is not None
    assert analysis.counterfactual.hc_off_step_time_ms > 0
