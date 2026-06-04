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


def build_parallel_domains(
    parallel: ParallelConfig,
    world_size: int | None = None,
) -> ParallelDomains:
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
