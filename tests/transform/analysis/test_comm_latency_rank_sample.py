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
