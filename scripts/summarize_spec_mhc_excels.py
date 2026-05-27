from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        mhc_time_from_ops,
        normalize_batch_token,
        percent_as_ratio,
        sheet_key_values,
        step_time_ms,
        strategy_values,
        value_as_float,
        write_summary_xlsx,
    )
except ModuleNotFoundError:
    from spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        mhc_time_from_ops,
        normalize_batch_token,
        percent_as_ratio,
        sheet_key_values,
        step_time_ms,
        strategy_values,
        value_as_float,
        write_summary_xlsx,
    )


MHC_KINDS = ("hc_expand", "mhc_head", "mhc_post", "mhc_pre")

COLUMNS = [
    "name",
    "file",
    "hw",
    "seq_len",
    "world_size",
    "tp",
    "cp",
    "pp",
    "ep",
    "dp",
    "pp_schedule",
    "vpp_chunks",
    "cp_kind",
    "micro_batch",
    "global_batch",
    "zero_stage",
    "optimizer",
    "compute_time_ms",
    "fwd_compute_ms",
    "bwd_compute_ms",
    "exposed_comm_ms",
    "tp_total_ms",
    "tp_exposed_ms",
    "cp_total_ms",
    "cp_exposed_ms",
    "ep_total_ms",
    "ep_exposed_ms",
    "pp_total_ms",
    "pp_exposed_ms",
    "pp_hidden_ms",
    "dp_total_ms",
    "dp_exposed_ms",
    "optimizer_compute_ms",
    "optimizer_comm_ms",
    "optimizer_exposed_ms",
    "recompute_time_ms",
    "recompute_time_raw_ms",
    "step_time_ms",
    "pipeline_time_ms",
    "mfu",
    "mfu_native",
    "hfu",
    "bubble_fraction",
    "bubble_time_ms",
    "tokens_per_sec",
    "mhc_total_ms",
    "mhc_step_share",
    "mhc_compute_share",
    "mhc_recompute_share",
    "mhc_delta_step_ms",
    "mhc_delta_step_pct",
    "mhc_hc_on_step_time_ms",
    "mhc_hc_off_step_time_ms",
        "mhc_delta_compute_ms",
        "mhc_delta_compute_pct",
        "mhc_time_source",
        "mhc_fwd_ms",
    "mhc_bwd_dx_ms",
    "mhc_bwd_dw_ms",
    "mhc_total_flops",
    "mhc_total_bytes",
    "mhc_enabled",
    "mhc_hc_mult",
    "mhc_hc_sinkhorn_iters",
    *[
        f"{kind}_{field}"
        for kind in MHC_KINDS
        for field in (
            "count",
            "fwd_ms",
            "bwd_dx_ms",
            "bwd_dw_ms",
            "total_ms",
            "total_flops",
            "total_bytes",
        )
    ],
]


def summarize_mhc_reports(input_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in find_excel_files(input_dir):
        wb = load_workbook(path)
        summary = sheet_key_values(wb, "Summary")
        if "Step Time" not in summary:
            continue
        step_ms = step_time_ms(wb)
        mhc = sheet_key_values(wb, "MHC Analysis")
        if mhc:
            mhc_ms = value_as_float(mhc.get("Total MHC Time"))
            share = percent_as_ratio(mhc.get("MHC Share of Step"))
            if share == 0.0 and step_ms:
                share = mhc_ms / step_ms
            row = {
                **_mhc_analysis_values(wb),
                "mhc_total_ms": mhc_ms,
                "mhc_step_share": share,
                "mhc_time_source": "MHC Analysis",
            }
        else:
            ops_values = _mhc_ops_values(wb)
            mhc_ms = float(ops_values.get("mhc_total_ms", 0.0))
            share = mhc_ms / step_ms if step_ms else 0.0
            row = {
                **ops_values,
                "mhc_total_ms": mhc_ms,
                "mhc_step_share": share,
                "mhc_time_source": "Ops",
            }
        rows.append({
            **_report_summary_values(wb, clean_report_name(path)),
            "file": str(path),
            **_empty_mhc_values(),
            **row,
        })
    return rows


def _empty_mhc_values() -> dict[str, object]:
    return {col: 0.0 for col in COLUMNS if col.startswith("mhc_") or col.startswith("hc_")}


def _mhc_analysis_values(wb) -> dict[str, object]:
    mhc = sheet_key_values(wb, "MHC Analysis")
    values: dict[str, object] = {
        "mhc_total_ms": value_as_float(mhc.get("Total MHC Time")),
        "mhc_step_share": percent_as_ratio(mhc.get("MHC Share of Step")),
        "mhc_compute_share": percent_as_ratio(mhc.get("MHC Share of Compute")),
        "mhc_recompute_share": percent_as_ratio(mhc.get("MHC Share of Recompute")),
        "mhc_fwd_ms": value_as_float(mhc.get("FWD")),
        "mhc_bwd_dx_ms": value_as_float(mhc.get("BWD dX")),
        "mhc_bwd_dw_ms": value_as_float(mhc.get("BWD dW")),
        "mhc_total_flops": value_as_float(mhc.get("Total MHC FLOPs")),
        "mhc_total_bytes": value_as_float(mhc.get("Total MHC Bytes")),
        "mhc_enabled": mhc.get("Enabled"),
        "mhc_hc_mult": mhc.get("HC Mult"),
        "mhc_hc_sinkhorn_iters": mhc.get("HC Sinkhorn Iters"),
    }
    values.update(_mhc_counterfactual_values(wb))
    values.update(_mhc_by_kind_values(wb))
    return values


def _mhc_counterfactual_values(wb) -> dict[str, float]:
    out: dict[str, float] = {}
    if "MHC Analysis" not in wb.sheetnames:
        return out
    ws = wb["MHC Analysis"]
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        key = str(row[0]).strip()
        if key == "Step Time (ms)":
            out.update({
                "mhc_hc_on_step_time_ms": value_as_float(row[1] if len(row) > 1 else None),
                "mhc_hc_off_step_time_ms": value_as_float(row[2] if len(row) > 2 else None),
                "mhc_delta_step_ms": value_as_float(row[3] if len(row) > 3 else None),
                "mhc_delta_step_pct": percent_as_ratio(row[4] if len(row) > 4 else None) * 100.0,
            })
        elif key == "Compute Time (ms)":
            out.update({
                "mhc_delta_compute_ms": value_as_float(row[3] if len(row) > 3 else None),
                "mhc_delta_compute_pct": percent_as_ratio(row[4] if len(row) > 4 else None) * 100.0,
            })
    return out


def _mhc_ops_values(wb) -> dict[str, object]:
    out: dict[str, object] = {"mhc_total_ms": mhc_time_from_ops(wb)}
    if "Ops" not in wb.sheetnames:
        return out
    ws = wb["Ops"]
    rows = ws.iter_rows(values_only=True)
    try:
        header = [str(v).strip() if v is not None else "" for v in next(rows)]
    except StopIteration:
        return out

    kind_idx = _find_header(header, "Kind")
    latency_idx = _find_header(header, "Latency")
    flops_idx = _find_header(header, "Total FLOPs")
    fwd_flops_idx = _find_header(header, "Fwd FLOPs")
    bwd_flops_idx = _find_header(header, "Bwd FLOPs")
    fwd_bytes_idx = _find_header(header, "Fwd Bytes")
    bwd_bytes_idx = _find_header(header, "Bwd Bytes")
    if kind_idx is None or latency_idx is None:
        return out

    total_flops = 0.0
    total_bytes = 0.0
    for row in rows:
        if len(row) <= max(kind_idx, latency_idx):
            continue
        kind = str(row[kind_idx]).strip() if row[kind_idx] is not None else ""
        if kind not in MHC_KINDS:
            continue
        prefix = kind
        latency_ms = value_as_float(row[latency_idx]) / 1000.0
        fwd_flops = _row_float(row, fwd_flops_idx)
        bwd_flops = _row_float(row, bwd_flops_idx)
        flops = _row_float(row, flops_idx, fwd_flops + bwd_flops)
        bytes_ = _row_float(row, fwd_bytes_idx) + _row_float(row, bwd_bytes_idx)

        out[f"{prefix}_count"] = value_as_float(out.get(f"{prefix}_count")) + 1
        out[f"{prefix}_total_ms"] = value_as_float(out.get(f"{prefix}_total_ms")) + latency_ms
        out[f"{prefix}_total_flops"] = value_as_float(out.get(f"{prefix}_total_flops")) + flops
        out[f"{prefix}_total_bytes"] = value_as_float(out.get(f"{prefix}_total_bytes")) + bytes_
        total_flops += flops
        total_bytes += bytes_

    out["mhc_total_flops"] = total_flops
    out["mhc_total_bytes"] = total_bytes
    return out


def _row_float(row: tuple[object, ...], idx: int | None, default: float = 0.0) -> float:
    if idx is None or len(row) <= idx:
        return default
    return value_as_float(row[idx], default)


def _find_header(header: list[str], target: str) -> int | None:
    target_l = target.lower()
    for idx, name in enumerate(header):
        if target_l in name.lower():
            return idx
    return None


def _mhc_by_kind_values(wb) -> dict[str, float]:
    out: dict[str, float] = {}
    if "MHC Analysis" not in wb.sheetnames:
        return out
    ws = wb["MHC Analysis"]
    for row in ws.iter_rows(values_only=True):
        if not row or row[0] is None:
            continue
        kind = str(row[0]).strip()
        if kind not in MHC_KINDS:
            continue
        out.update({
            f"{kind}_count": value_as_float(row[1] if len(row) > 1 else None),
            f"{kind}_fwd_ms": value_as_float(row[2] if len(row) > 2 else None),
            f"{kind}_bwd_dx_ms": value_as_float(row[3] if len(row) > 3 else None),
            f"{kind}_bwd_dw_ms": value_as_float(row[4] if len(row) > 4 else None),
            f"{kind}_total_ms": value_as_float(row[5] if len(row) > 5 else None),
            f"{kind}_total_flops": value_as_float(row[6] if len(row) > 6 else None),
            f"{kind}_total_bytes": value_as_float(row[7] if len(row) > 7 else None),
        })
    return out


def _report_summary_values(wb, name: str) -> dict[str, object]:
    summary = sheet_key_values(wb, "Summary")
    model = sheet_key_values(wb, "Model")
    hardware = sheet_key_values(wb, "Hardware")
    hw, seq_len = _split_hw_seq_name(name)
    return {
        "name": name,
        "hw": hw,
        "seq_len": value_as_float(model.get("Seq Len"), value_as_float(seq_len)),
        "world_size": value_as_float(hardware.get("World Size")),
        **strategy_values(wb),
        "compute_time_ms": (
            value_as_float(summary.get("Forward Compute"))
            + value_as_float(summary.get("Backward Compute"))
        ),
        "fwd_compute_ms": value_as_float(summary.get("Forward Compute")),
        "bwd_compute_ms": value_as_float(summary.get("Backward Compute")),
        "exposed_comm_ms": value_as_float(summary.get("Communication (exposed)")),
        "tp_total_ms": value_as_float(summary.get("TP (RS/AG)")),
        "tp_exposed_ms": value_as_float(summary.get("TP (RS/AG)")),
        "cp_total_ms": value_as_float(summary.get("CP (A2A)")),
        "cp_exposed_ms": value_as_float(summary.get("CP (A2A)")),
        "ep_total_ms": value_as_float(summary.get("EP (A2A)")),
        "ep_exposed_ms": value_as_float(summary.get("EP (A2A)")),
        "pp_total_ms": value_as_float(summary.get("PP (P2P)")),
        "pp_exposed_ms": value_as_float(summary.get("PP (P2P)")),
        "pp_hidden_ms": value_as_float(summary.get("PP hidden")),
        "dp_total_ms": value_as_float(summary.get("DP (AR/RS)")),
        "dp_exposed_ms": value_as_float(summary.get("DP (AR/RS)")),
        "optimizer_compute_ms": value_as_float(summary.get("Optimizer (compute)")),
        "optimizer_comm_ms": value_as_float(summary.get("Optimizer (comm)")),
        "optimizer_exposed_ms": value_as_float(summary.get("Optimizer (comm)")),
        "recompute_time_ms": value_as_float(summary.get("Recompute (critical path)")),
        "recompute_time_raw_ms": value_as_float(summary.get("Recompute (raw, NOT in step)")),
        "step_time_ms": step_time_ms(wb),
        "pipeline_time_ms": value_as_float(summary.get("Per-Stage Time")),
        "mfu": percent_as_ratio(summary.get("MFU")),
        "mfu_native": percent_as_ratio(summary.get("MFU (native)")),
        "hfu": percent_as_ratio(summary.get("HFU")),
        "bubble_fraction": 0.0,
        "bubble_time_ms": value_as_float(summary.get("Pipeline Bubble")),
        "tokens_per_sec": value_as_float(summary.get("Tokens/Second")),
    }


def _split_hw_seq_name(name: str) -> tuple[str, str]:
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        seq = normalize_batch_token(parts[1])
        if seq.isdigit():
            return parts[0], seq
    return name, ""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize spec-path Excel reports with MHC operators."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Excel reports.")
    parser.add_argument("--output", required=True, help="Output .xlsx summary path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = summarize_mhc_reports(args.input_dir)
    out = write_summary_xlsx(rows, COLUMNS, args.output)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
