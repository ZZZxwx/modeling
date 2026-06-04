# Graph Capture Megatron Parallel Domains Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Megatron-style `ETP/EP/EDP` domain metadata and rank samples to the graph-capture transform path without changing the spec training path.

**Architecture:** Add a small transform-only parallel-domain helper that derives dense and expert rank groups from `ParallelConfig`. `CommInserterPass` will attach canonical communication metadata and CP-local EP A2A bytes. `CommLatencyPass` will prefer `rank_sample` for intra-node vs cross-node placement and retain the old group-size heuristic as fallback.

**Tech Stack:** Python dataclasses, existing `OpGraph`/`OpNode` transform passes, pytest.

---

## File Structure

- Create `python/zrt/transform/parallel/domains.py`
  - Owns graph-capture-only derivation of `TP/CP/DP/PP` and `ETP/EP/EDP`.
  - Provides explicit group sizes and representative rank samples.
- Modify `python/zrt/transform/context.py`
  - Add `tp_extend_ep: bool = False` to `ParallelConfig`.
  - Keep `describe()` and existing fields stable.
- Modify `python/zrt/transform/parallel/comm_inserter.py`
  - Attach `comm_group`, `comm_domain`, `comm_bytes`, and `rank_sample`.
  - Compute EP A2A bytes with CP-local sequence length.
- Modify `python/zrt/transform/analysis/comm_latency.py`
  - Prefer `rank_sample` for intra-node vs cross-node detection.
  - Continue falling back to `group_size > intra_node_devices`.
- Create `tests/transform/test_parallel_domains.py`
  - Unit tests for domain derivation and rank samples.
- Create `tests/transform/test_comm_inserter_domains.py`
  - Transform-path tests for EP A2A metadata and bytes.
- Create `tests/transform/analysis/test_comm_latency_rank_sample.py`
  - Tests for rank-sample-aware communication placement.

## Task 1: Transform Parallel Domain Helper

**Files:**
- Create: `python/zrt/transform/parallel/domains.py`
- Modify: `python/zrt/transform/context.py`
- Test: `tests/transform/test_parallel_domains.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/transform/test_parallel_domains.py`:

```python
import pytest

from python.zrt.transform.context import ParallelConfig
from python.zrt.transform.parallel.domains import build_parallel_domains


def test_dense_and_expert_domains_without_tp_extend_ep():
    parallel = ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4)

    domains = build_parallel_domains(parallel, world_size=128)

    assert domains.stage_world == 64
    assert domains.tp == 4
    assert domains.cp == 4
    assert domains.dp == 4
    assert domains.pp == 2
    assert domains.etp == 4
    assert domains.ep == 4
    assert domains.edp == 4
    assert domains.rank_sample("TP") == [0, 1, 2, 3]
    assert domains.rank_sample("CP") == [0, 4, 8, 12]
    assert domains.rank_sample("DP") == [0, 16, 32, 48]
    assert domains.rank_sample("PP") == [0, 64]
    assert domains.rank_sample("ETP") == [0, 1, 2, 3]
    assert domains.rank_sample("EP") == [0, 4, 8, 12]
    assert domains.rank_sample("EDP") == [0, 16, 32, 48]


def test_dense_and_expert_domains_with_tp_extend_ep():
    parallel = ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4, tp_extend_ep=True)

    domains = build_parallel_domains(parallel, world_size=128)

    assert domains.stage_world == 64
    assert domains.etp == 1
    assert domains.ep == 4
    assert domains.edp == 16
    assert domains.rank_sample("ETP") == [0]
    assert domains.rank_sample("EP") == [0, 1, 2, 3]
    assert domains.rank_sample("EDP") == [
        0, 4, 8, 12, 16, 20, 24, 28,
        32, 36, 40, 44, 48, 52, 56, 60,
    ]


def test_parallel_domains_reject_non_integral_edp():
    parallel = ParallelConfig(tp=4, pp=2, dp=3, cp=4, ep=8)

    with pytest.raises(ValueError, match="EDP"):
        build_parallel_domains(parallel, world_size=96)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_parallel_domains.py -v
```

Expected: FAIL because `python.zrt.transform.parallel.domains` does not exist and `ParallelConfig` does not accept `tp_extend_ep`.

- [ ] **Step 3: Implement `tp_extend_ep` and domain helper**

Modify `python/zrt/transform/context.py`:

```python
@dataclass
class ParallelConfig:
    tp: int = 1
    pp: int = 1
    ep: int = 1
    dp: int = 1
    cp: int = 1
    sp: bool = False
    tp_extend_ep: bool = False
```

Create `python/zrt/transform/parallel/domains.py`:

```python
"""Transform-path parallel domain derivation.

This helper is intentionally scoped to graph capture. The spec training path
has its own process-group and communication-domain modelling.
"""
from __future__ import annotations

from dataclasses import dataclass

from python.zrt.transform.context import ParallelConfig


@dataclass(frozen=True)
class ParallelDomains:
    world_size: int
    stage_world: int
    tp: int
    cp: int
    dp: int
    pp: int
    etp: int
    ep: int
    edp: int

    def group_size(self, name: str) -> int:
        key = name.upper()
        values = {
            "TP": self.tp,
            "CP": self.cp,
            "DP": self.dp,
            "PP": self.pp,
            "ETP": self.etp,
            "EP": self.ep,
            "EDP": self.edp,
            "MOE_EP": self.ep,
        }
        if key not in values:
            raise KeyError(f"unknown parallel domain {name!r}")
        return values[key]

    def rank_sample(self, name: str) -> list[int]:
        key = name.upper()
        if key == "MOE_EP":
            key = "EP"
        if key == "TP":
            return [tp for tp in range(self.tp)]
        if key == "CP":
            return [cp * self.tp for cp in range(self.cp)]
        if key == "DP":
            stride = self.cp * self.tp
            return [dp * stride for dp in range(self.dp)]
        if key == "PP":
            return [pp * self.stage_world for pp in range(self.pp)]
        if key == "ETP":
            return [etp for etp in range(self.etp)]
        if key == "EP":
            return [ep * self.etp for ep in range(self.ep)]
        if key == "EDP":
            stride = self.ep * self.etp
            return [edp * stride for edp in range(self.edp)]
        raise KeyError(f"unknown parallel domain {name!r}")


def build_parallel_domains(parallel: ParallelConfig, world_size: int | None = None) -> ParallelDomains:
    pp = max(1, parallel.pp)
    tp = max(1, parallel.tp)
    cp = max(1, parallel.cp)
    dp = max(1, parallel.dp)
    ep = max(1, parallel.ep)
    if world_size is None:
        world_size = tp * cp * dp * pp
    if world_size % pp != 0:
        raise ValueError(f"world_size={world_size} must be divisible by PP={pp}")
    stage_world = world_size // pp
    dense_world = tp * cp * dp
    if dense_world != stage_world:
        raise ValueError(
            f"TP*CP*DP={dense_world} must equal world_size/PP={stage_world}"
        )
    etp = 1 if parallel.tp_extend_ep else tp
    denom = etp * ep
    if dense_world % denom != 0:
        raise ValueError(
            f"EDP must be integral: TP*CP*DP={dense_world}, ETP*EP={denom}"
        )
    edp = dense_world // denom
    return ParallelDomains(
        world_size=world_size,
        stage_world=stage_world,
        tp=tp,
        cp=cp,
        dp=dp,
        pp=pp,
        etp=etp,
        ep=ep,
        edp=edp,
    )
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_parallel_domains.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add python\zrt\transform\context.py python\zrt\transform\parallel\domains.py tests\transform\test_parallel_domains.py
git commit -m "feat: derive transform parallel domains"
```

## Task 2: EP A2A Metadata and CP-Local Bytes

**Files:**
- Modify: `python/zrt/transform/parallel/comm_inserter.py`
- Test: `tests/transform/test_comm_inserter_domains.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/transform/test_comm_inserter_domains.py`:

```python
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.context import ParallelConfig, TrainingConfig, TransformContext
from python.zrt.transform.parallel.comm_inserter import CommInserterPass


class _Profile:
    moe_active = 2


def _tensor(name: str, shape=(1, 16, 64)):
    return TensorMeta.from_shape_dtype(name, shape, DType.BF16)


def _expert_graph() -> OpGraph:
    gate = OpNode(
        id="gate_up",
        op_type="GroupedMatMul",
        inputs=[_tensor("gate_in")],
        outputs=[_tensor("gate_out")],
        scope="model.layers.0.mlp.experts.gate_up",
        layer="0",
        category="compute",
    )
    down = OpNode(
        id="down",
        op_type="GroupedMatMul",
        inputs=[_tensor("down_in")],
        outputs=[_tensor("down_out")],
        scope="model.layers.0.mlp.experts.down",
        layer="0",
        category="compute",
    )
    gate.annotations["ep_needs_a2a"] = True
    gate.annotations["ep_block_down_id"] = "down"
    down.annotations["ep_needs_a2a"] = True
    edge = Edge("gate_up", 0, "down", 0, _tensor("gate_to_down"))
    graph = OpGraph(
        name="ep_test",
        phase="train_forward",
        nodes={"gate_up": gate, "down": down},
        edges=[edge],
    )
    graph.metadata["seq_len"] = 16
    graph.metadata["hidden"] = 64
    return graph


def test_ep_a2a_nodes_carry_domain_metadata_and_rank_samples():
    ctx = TransformContext(
        hw_spec=None,
        parallel=ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4, tp_extend_ep=True),
        training=TrainingConfig(micro_batch=2, seq_len=16, hidden=64),
        profile=_Profile(),
    )

    result = CommInserterPass().run(_expert_graph(), ctx)

    dispatch = result.nodes["comm_a2a_dispatch_gate_up"]
    combine = result.nodes["comm_a2a_combine_down"]
    for node in (dispatch, combine):
        assert node.attrs["comm_group"] == "EP"
        assert node.attrs["comm_domain"] == "MOE_EP"
        assert node.attrs["group_size"] == 4
        assert node.attrs["rank_sample"] == [0, 1, 2, 3]
        assert node.attrs["comm_bytes"] == node.attrs["msg_bytes"]
        assert node.attrs["msg_bytes_semantics"] == "per_a2a_direction"


def test_ep_a2a_bytes_use_cp_local_sequence_length():
    ctx = TransformContext(
        hw_spec=None,
        parallel=ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4),
        training=TrainingConfig(micro_batch=2, seq_len=16, hidden=64),
        profile=_Profile(),
    )

    result = CommInserterPass().run(_expert_graph(), ctx)

    dispatch = result.nodes["comm_a2a_dispatch_gate_up"]
    expected = 2 * (16 // 4) * 2 * 64 * 2
    assert dispatch.attrs["msg_bytes"] == expected
    assert dispatch.attrs["comm_bytes"] == expected
    assert dispatch.inputs[0].shape == (2, 4, 64)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_comm_inserter_domains.py -v
```

Expected: FAIL because EP A2A nodes do not include canonical domain attrs, use full `seq_len`, and dispatch tensor shape remains full sequence length.

- [ ] **Step 3: Add metadata and CP-local bytes**

Modify `python/zrt/transform/parallel/comm_inserter.py`:

```python
from python.zrt.transform.parallel.domains import build_parallel_domains
```

Inside `_insert_ep_comm`, replace the EP sizing block with:

```python
        seq_len = g.metadata.get("seq_len", ctx.training.seq_len if ctx.training else 2048)
        hidden = g.metadata.get("hidden", ctx.training.hidden if ctx.training else 4096)
        dtype_bytes = 2
        micro_batch = ctx.training.micro_batch if ctx.training else 1
        topk = ctx.profile.moe_active if ctx.profile else 8
        domains = build_parallel_domains(ctx.parallel)
        seq_local = seq_len // max(1, domains.cp)

        routed_tokens = micro_batch * seq_local * topk
        ep_msg_bytes = routed_tokens * hidden * dtype_bytes
        ep_rank_sample = domains.rank_sample("EP")
```

Change dispatch and combine tensors to use `seq_local`:

```python
            shape=(micro_batch, seq_local, hidden),
```

Add these attrs to both EP A2A nodes while preserving existing attrs:

```python
                           "comm_group": "EP",
                           "comm_domain": "MOE_EP",
                           "comm_bytes": ep_msg_bytes,
                           "rank_sample": ep_rank_sample,
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_comm_inserter_domains.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add python\zrt\transform\parallel\comm_inserter.py tests\transform\test_comm_inserter_domains.py
git commit -m "feat: annotate graph capture ep communication domains"
```

## Task 3: Canonical Metadata for TP and CP Communication Nodes

**Files:**
- Modify: `python/zrt/transform/parallel/comm_inserter.py`
- Test: `tests/transform/test_comm_inserter_domains.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/transform/test_comm_inserter_domains.py`:

```python
def test_tp_comm_node_carries_domain_metadata():
    node = OpNode(
        id="row_linear",
        op_type="aten.mm.default",
        inputs=[_tensor("row_in", (8, 64))],
        outputs=[_tensor("row_out", (8, 64))],
        scope="model.layers.0.self_attn.o_proj",
        layer="0",
        category="compute",
    )
    node.annotations["tp_split"] = {"comm_after": "all_reduce"}
    graph = OpGraph(name="tp_test", phase="train_forward", nodes={"row_linear": node}, edges=[])
    ctx = TransformContext(
        hw_spec=None,
        parallel=ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4),
        training=TrainingConfig(micro_batch=1, seq_len=16, hidden=64),
    )

    result = CommInserterPass().run(graph, ctx)

    comm = result.nodes["comm_allreduce_row_linear"]
    assert comm.attrs["comm_group"] == "TP"
    assert comm.attrs["comm_domain"] == "DENSE_TP"
    assert comm.attrs["group_size"] == 4
    assert comm.attrs["rank_sample"] == [0, 1, 2, 3]


def test_cp_comm_nodes_carry_domain_metadata():
    node = OpNode(
        id="attn",
        op_type="aten.scaled_dot_product_attention.default",
        inputs=[_tensor("attn_in", (1, 4, 64))],
        outputs=[_tensor("attn_out", (1, 4, 64))],
        scope="model.layers.0.self_attn",
        layer="0",
        category="compute",
    )
    node.annotations["cp_split"] = {"kind": "ulysses"}
    graph = OpGraph(name="cp_test", phase="train_forward", nodes={"attn": node}, edges=[])
    ctx = TransformContext(
        hw_spec=None,
        parallel=ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4),
        training=TrainingConfig(micro_batch=1, seq_len=16, hidden=64),
    )

    result = CommInserterPass().run(graph, ctx)

    cp_nodes = [n for n in result.nodes.values() if n.category == "communication"]
    assert cp_nodes
    for comm in cp_nodes:
        assert comm.attrs["comm_group"] == "CP"
        assert comm.attrs["comm_domain"] == "DENSE_CP"
        assert comm.attrs["group_size"] == 4
        assert comm.attrs["rank_sample"] == [0, 4, 8, 12]
        assert comm.attrs["comm_bytes"] == comm.attrs["bytes"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_comm_inserter_domains.py -v
```

Expected: FAIL because TP and CP comm nodes do not yet include canonical metadata.

- [ ] **Step 3: Add shared metadata helper inside `comm_inserter.py`**

Add helper:

```python
def _attach_comm_domain_attrs(
    node: OpNode,
    *,
    comm_group: str,
    comm_domain: str,
    group_size: int,
    rank_sample: list[int],
    comm_bytes: int | None = None,
) -> None:
    node.attrs["comm_group"] = comm_group
    node.attrs["comm_domain"] = comm_domain
    node.attrs["group_size"] = group_size
    node.attrs["rank_sample"] = rank_sample
    if comm_bytes is not None:
        node.attrs["comm_bytes"] = comm_bytes
```

Update `_make_comm_node` signature to accept optional domain fields:

```python
def _make_comm_node(
    node_id: str,
    collective: str,
    src_node: OpNode,
    group_size: int,
    *,
    comm_group: str | None = None,
    comm_domain: str | None = None,
    rank_sample: list[int] | None = None,
) -> OpNode:
```

After node creation:

```python
    if comm_group and comm_domain and rank_sample is not None:
        _attach_comm_domain_attrs(
            node,
            comm_group=comm_group,
            comm_domain=comm_domain,
            group_size=group_size,
            rank_sample=rank_sample,
        )
```

In `_insert_tp_comm`, build domains and pass TP metadata:

```python
        domains = build_parallel_domains(ctx.parallel)
```

```python
            comm_node = _make_comm_node(
                comm_id,
                "all_reduce",
                node,
                domains.group_size("TP"),
                comm_group="TP",
                comm_domain="DENSE_TP",
                rank_sample=domains.rank_sample("TP"),
            )
```

For CP creation helpers, build domains once in `_insert_cp_comm` and pass
`rank_sample=domains.rank_sample("CP")` into CP helper methods. Each CP helper
should call `_attach_comm_domain_attrs` with `comm_group="CP"`,
`comm_domain="DENSE_CP"`, `group_size=cp`, `rank_sample=rank_sample`, and the
same byte value already stored in that node's `bytes` attr.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_comm_inserter_domains.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git add python\zrt\transform\parallel\comm_inserter.py tests\transform\test_comm_inserter_domains.py
git commit -m "feat: annotate dense graph capture communication domains"
```

## Task 4: Rank-Sample-Aware Communication Latency

**Files:**
- Modify: `python/zrt/transform/analysis/comm_latency.py`
- Test: `tests/transform/analysis/test_comm_latency_rank_sample.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/transform/analysis/test_comm_latency_rank_sample.py`:

```python
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import DType, TensorMeta
from python.zrt.transform.analysis.comm_latency import CommLatencyPass
from python.zrt.transform.context import ParallelConfig, TransformContext


def _comm_node(rank_sample):
    return OpNode(
        id="ep_a2a",
        op_type="comm.all_to_all",
        inputs=[TensorMeta.from_shape_dtype("in", (1024, 1024), DType.BF16)],
        outputs=[TensorMeta.from_shape_dtype("out", (1024, 1024), DType.BF16)],
        attrs={
            "collective": "all_to_all",
            "group_size": 4,
            "msg_bytes": 1024 * 1024 * 2,
            "comm_group": "EP",
            "comm_domain": "MOE_EP",
            "rank_sample": rank_sample,
        },
        category="communication",
    )


def _run(node):
    import python.zrt.hardware.registry as hw_registry

    hw = hw_registry.load("nvidia_h100_sxm")
    graph = OpGraph(name="latency", phase="train_forward", nodes={node.id: node}, edges=[])
    ctx = TransformContext(hw_spec=hw, parallel=ParallelConfig(tp=4, pp=2, dp=4, cp=4, ep=4))
    return CommLatencyPass().run(graph, ctx).nodes[node.id]


def test_rank_sample_can_force_cross_node_even_when_group_size_fits_one_node():
    result = _run(_comm_node([0, 8, 16, 24]))

    assert result.annotations["cross_node"] is True
    assert result.annotations["placement_source"] == "rank_sample"


def test_rank_sample_can_keep_comm_intra_node_when_group_size_fits_one_node():
    result = _run(_comm_node([0, 1, 2, 3]))

    assert result.annotations["cross_node"] is False
    assert result.annotations["placement_source"] == "rank_sample"


def test_latency_falls_back_to_group_size_without_rank_sample():
    node = _comm_node(None)
    del node.attrs["rank_sample"]

    result = _run(node)

    assert result.annotations["cross_node"] is False
    assert result.annotations["placement_source"] == "group_size"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\analysis\test_comm_latency_rank_sample.py -v
```

Expected: FAIL because `CommLatencyPass` ignores `rank_sample` and does not set `placement_source`.

- [ ] **Step 3: Implement rank-sample placement helper**

Modify `python/zrt/transform/analysis/comm_latency.py` with:

```python
def _rank_to_node(rank: int, devices_per_node: int) -> int:
    return rank // max(1, devices_per_node)


def _cross_node_from_rank_sample(rank_sample: list[int] | tuple[int, ...], devices_per_node: int) -> bool:
    if not rank_sample:
        return False
    nodes = {_rank_to_node(int(rank), devices_per_node) for rank in rank_sample}
    return len(nodes) > 1
```

Replace cross-node detection with:

```python
            intra_node_devices = hw_spec.interconnect.intra_node.num_devices
            rank_sample = node.attrs.get("rank_sample")
            if rank_sample is not None:
                cross_node = _cross_node_from_rank_sample(rank_sample, intra_node_devices)
                placement_source = "rank_sample"
            else:
                cross_node = group_size > intra_node_devices
                placement_source = "group_size"
```

After annotations:

```python
            node.annotations["placement_source"] = placement_source
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\analysis\test_comm_latency_rank_sample.py -v
```

Expected: PASS.

- [ ] **Step 5: Run existing comm latency tests**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\test_executor.py::test_comm_latency_pass_intra_node tests\test_executor.py::test_comm_latency_pass_cross_node tests\test_executor.py::test_comm_latency_pass_alltoall -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git add python\zrt\transform\analysis\comm_latency.py tests\transform\analysis\test_comm_latency_rank_sample.py
git commit -m "feat: price graph capture comm by rank sample"
```

## Task 5: Integration Verification

**Files:**
- Modify only if failures reveal a graph-capture integration issue.

- [ ] **Step 1: Run focused transform tests**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\transform\test_parallel_domains.py tests\transform\test_comm_inserter_domains.py tests\transform\analysis\test_comm_latency_rank_sample.py -v
```

Expected: PASS.

- [ ] **Step 2: Run related existing tests**

Run:

```powershell
$env:PYTHONPATH='python'; py -m pytest tests\test_executor.py tests\training\test_transform_integration.py tests\IT\test_ep_e2e.py -v
```

Expected: PASS. If `tests\IT\test_ep_e2e.py` is slow or environment-gated, record the reason and run the fastest matching subset first.

- [ ] **Step 3: Inspect changed files**

Run:

```powershell
git diff --stat HEAD~4..HEAD
git diff --check
git status --short --untracked-files=no
```

Expected: no whitespace errors; only intended tracked files changed or committed.

- [ ] **Step 4: Final commit if integration fixes were needed**

If Step 2 required integration fixes, commit them:

```powershell
git status --short --untracked-files=no
```

Then add only the files reported by the command that belong to this feature and
commit them:

```powershell
git commit -m "fix: align graph capture parallel domain integration"
```

If no fixes were needed, do not create an empty commit.

## Self-Review

- Spec coverage: The plan covers `tp_extend_ep`, `ETP/EP/EDP`, explicit `group_size`, rank samples, EP A2A CP-local bytes, transform comm node metadata, and rank-sample-aware latency. It explicitly avoids the spec training path.
- Placeholder scan: No task contains an unfinished marker or unspecified implementation step.
- Type consistency: `ParallelConfig.tp_extend_ep`, `build_parallel_domains`, `ParallelDomains.group_size`, and `ParallelDomains.rank_sample` are used consistently across tests and implementation steps.
