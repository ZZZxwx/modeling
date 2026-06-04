# Graph Capture Megatron Parallel Domains Design

## Scope

This change is limited to the graph-capture / transform OpGraph path.

It does not change the spec training path, including `training/ir/shard.py`,
`training/models/comm.py`, or the existing spec-level communication summary.

## Goal

Represent Megatron-style dense and expert parallel domains inside the graph
capture pipeline while keeping the user-facing configuration and final reports
as stable as possible.

The dense domain is:

```text
TP * CP * DP
```

The expert domain is:

```text
ETP * EP * EDP
```

For each pipeline stage:

```text
stage_world = world_size / PP
stage_world = TP * CP * DP = ETP * EP * EDP
```

## Configuration

The only new user-facing switch is:

```text
tp_extend_ep
```

The `EDP` value is internal only. It should be derived by the transform path
and may appear in debug metadata or rank-domain annotations, but it should not
be introduced as a primary config field.

When `tp_extend_ep` is false:

```text
ETP = TP
EDP = CP * DP / EP
```

When `tp_extend_ep` is true:

```text
ETP = 1
EDP = TP * CP * DP / EP
```

The implementation must validate that `ETP * EP * EDP == TP * CP * DP` and
that derived divisions are integral.

## Rank Samples

The graph-capture path should derive representative rank samples for each
parallel domain. A rank sample explains which ranks one example group contains.
It also gives communication analysis enough information to distinguish
same-node and cross-node collectives more accurately than a group-size-only
heuristic.

`group_size` must be the explicitly derived degree for the communication
domain. It must not be inferred from `len(rank_sample)`.

Example metadata:

```python
{
    "comm_group": "EP",
    "comm_domain": "MOE_EP",
    "group_size": ep,
    "rank_sample": ep_rank_sample,
    "comm_bytes": a2a_bytes,
}
```

For `world_size=128`, `PP=2`, `DP=4`, `CP=4`, and therefore `TP=4`, using a
layout where TP is the fastest dense dimension:

```text
dense TP sample = [0, 1, 2, 3]
dense CP sample = [0, 4, 8, 12]
dense DP sample = [0, 16, 32, 48]
PP sample       = [0, 64]
```

With `EP=4` and `tp_extend_ep=false`:

```text
ETP sample = [0, 1, 2, 3]
EP sample  = [0, 4, 8, 12]
EDP sample = [0, 16, 32, 48]
```

With `EP=4` and `tp_extend_ep=true`:

```text
ETP sample = [0]
EP sample  = [0, 1, 2, 3]
EDP sample = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60]
```

These samples demonstrate why rank samples are needed: both modes can have the
same `EP` group size, but their EP communication can land on different physical
rank placements.

## Communication Node Metadata

Transform-inserted communication nodes should continue to preserve existing
fields such as:

```text
collective
group_size
msg_bytes
bytes
role
```

The capture path should also add canonical domain fields where applicable:

```text
comm_group
comm_domain
comm_bytes
rank_sample
```

For MoE dispatch/combine all-to-all:

```text
comm_group  = "EP"
comm_domain = "MOE_EP"
group_size  = EP
rank_sample = EP sample
comm_bytes  = message bytes for the A2A direction
```

For dense collectives, the same pattern applies with domains such as TP, CP, DP,
or PP as appropriate.

## EP A2A Bytes

In the graph-capture path, EP A2A token volume should use the CP-local sequence
length:

```text
seq_local = seq_len / CP
a2a_bytes = micro_batch * seq_local * topk * hidden * dtype_bytes
```

The first implementation should avoid a broad rewrite of GroupedMM shape
semantics. The priority is to correct the EP communication domain, rank sample,
and CP-local token volume.

## Communication Latency

The transform communication latency path should prefer rank-sample-aware
placement when it is available:

```text
if rank_sample is present:
    determine the highest topology tier crossed by the sample
else:
    use the existing group_size-based fallback
```

On a two-tier system this reduces to:

```text
all sampled ranks on one node -> intra-node
otherwise -> inter-node
```

If the system exposes more tiers, the communication analyzer should use the
highest tier crossed by the representative sample.

This keeps the old behavior as a fallback while allowing EP, ETP, EDP, TP, CP,
DP, and PP groups with the same group size to price differently when their rank
placement differs.

## Tests

Implementation should follow TDD. Add failing transform-path tests first for:

1. `tp_extend_ep=false` derives `ETP=TP` and `EDP=CP*DP/EP`.
2. `tp_extend_ep=true` derives `ETP=1` and `EDP=TP*CP*DP/EP`.
3. Rank samples match the `world_size=128`, `PP=2`, `DP=4`, `CP=4`, `TP=4`,
   `EP=4` examples above.
4. EP A2A nodes carry `comm_group`, `comm_domain`, explicit `group_size`,
   `rank_sample`, and `comm_bytes`.
5. EP A2A bytes use `seq_len / CP`.
6. Communication latency prefers `rank_sample` over the group-size-only
   heuristic when both are available.
