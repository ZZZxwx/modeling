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
