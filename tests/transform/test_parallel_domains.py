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
