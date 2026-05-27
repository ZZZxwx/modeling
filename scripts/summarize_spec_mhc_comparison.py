from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        normalize_batch_token,
        write_summary_xlsx,
    )
    from scripts.summarize_spec_mhc_excels import (
        COLUMNS as MHC_COLUMNS,
        _report_summary_values,
        summarize_mhc_reports,
    )
except ModuleNotFoundError:
    from spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        normalize_batch_token,
        write_summary_xlsx,
    )
    from summarize_spec_mhc_excels import (
        COLUMNS as MHC_COLUMNS,
        _report_summary_values,
        summarize_mhc_reports,
    )


NO_MHC_BASE_COLUMNS = [
    "source_name",
    "source_file",
    "step_time_ms",
    "tokens_per_sec",
    "mfu",
    "mfu_native",
    "hfu",
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
]

NO_MHC_COLUMNS = [
    "no_mhc_match_status",
    *[f"no_mhc_{col}" for col in NO_MHC_BASE_COLUMNS],
    "mhc_vs_no_mhc_step_delta_ms",
    "mhc_vs_no_mhc_step_delta_pct",
    "mhc_vs_no_mhc_tokens_per_sec_delta",
    "mhc_vs_no_mhc_tokens_per_sec_delta_pct",
]

COLUMNS = [
    "match_key",
    *MHC_COLUMNS,
    *NO_MHC_COLUMNS,
]


def summarize_comparison(
    mhc_input_dir: str | Path,
    no_mhc_input_dir: str | Path,
    output: str | Path,
) -> list[dict[str, object]]:
    mhc_rows = summarize_mhc_reports(mhc_input_dir)
    no_mhc_rows = summarize_no_mhc_reports(no_mhc_input_dir)
    no_mhc_by_key = _index_no_mhc_rows(no_mhc_rows)

    rows: list[dict[str, object]] = []
    for mhc_row in mhc_rows:
        key = match_key(mhc_row)
        no_mhc_row = no_mhc_by_key.get(key)
        row = {"match_key": key, **mhc_row}
        if no_mhc_row is None:
            row.update({"no_mhc_match_status": "missing"})
        elif no_mhc_row.get("__duplicate__"):
            row.update({"no_mhc_match_status": "duplicate"})
        else:
            row.update(_prefixed_no_mhc_values(no_mhc_row))
            row.update(_comparison_values(mhc_row, no_mhc_row))
        rows.append(row)

    write_summary_xlsx(rows, COLUMNS, output)
    return rows


def summarize_no_mhc_reports(input_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in find_excel_files(input_dir):
        wb = load_workbook(path)
        values = _report_summary_values(wb, clean_report_name(path))
        if not values.get("step_time_ms"):
            continue
        rows.append({**values, "file": str(path)})
    return rows


def match_key(row: dict[str, object]) -> str:
    hw = str(row.get("hw") or "").strip()
    seq_len = row.get("seq_len")
    seq_text = _seq_text(seq_len)
    if hw and seq_text:
        return f"{hw}_{seq_text}".lower()
    return _key_from_name(str(row.get("name") or ""))


def _key_from_name(name: str) -> str:
    stem = clean_report_name(name)
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return stem.lower()
    seq = normalize_batch_token(parts[1])
    return f"{parts[0]}_{seq}".lower()


def _seq_text(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, (int, float)):
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip()
    if not text:
        return ""
    return normalize_batch_token(text)


def _index_no_mhc_rows(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    index: dict[str, dict[str, object]] = {}
    for row in rows:
        key = match_key(row)
        if key in index:
            index[key] = {"__duplicate__": True}
        else:
            index[key] = row
    return index


def _prefixed_no_mhc_values(row: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {"no_mhc_match_status": "ok"}
    row = {**row, "source_name": row.get("name"), "source_file": row.get("file")}
    for col in NO_MHC_BASE_COLUMNS:
        out[f"no_mhc_{col}"] = row.get(col)
    return out


def _comparison_values(mhc_row: dict[str, object], no_mhc_row: dict[str, object]) -> dict[str, object]:
    mhc_step = _float_value(mhc_row.get("step_time_ms"))
    no_mhc_step = _float_value(no_mhc_row.get("step_time_ms"))
    mhc_tps = _float_value(mhc_row.get("tokens_per_sec"))
    no_mhc_tps = _float_value(no_mhc_row.get("tokens_per_sec"))
    return {
        "mhc_vs_no_mhc_step_delta_ms": mhc_step - no_mhc_step,
        "mhc_vs_no_mhc_step_delta_pct": _safe_pct(mhc_step - no_mhc_step, no_mhc_step),
        "mhc_vs_no_mhc_tokens_per_sec_delta": mhc_tps - no_mhc_tps,
        "mhc_vs_no_mhc_tokens_per_sec_delta_pct": _safe_pct(mhc_tps - no_mhc_tps, no_mhc_tps),
    }


def _float_value(value: object) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return 0.0


def _safe_pct(delta: float, base: float) -> float:
    return delta / base * 100.0 if base else 0.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one MHC vs no-MHC .xlsx summary from two report folders."
    )
    parser.add_argument("--mhc-input-dir", required=True, help="Directory containing MHC Excel reports.")
    parser.add_argument("--no-mhc-input-dir", required=True, help="Directory containing no-MHC Excel reports.")
    parser.add_argument("--output", required=True, help="Output .xlsx summary path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = summarize_comparison(args.mhc_input_dir, args.no_mhc_input_dir, args.output)
    print(f"Wrote {len(rows)} rows to {Path(args.output)}")


if __name__ == "__main__":
    main()
