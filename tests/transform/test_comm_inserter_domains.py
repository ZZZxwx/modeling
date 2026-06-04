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
