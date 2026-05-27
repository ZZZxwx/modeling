from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        sheet_key_values,
        step_time_ms,
        strategy_values,
        write_summary_xlsx,
    )
except ModuleNotFoundError:
    from spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        sheet_key_values,
        step_time_ms,
        strategy_values,
        write_summary_xlsx,
    )


COLUMNS = [
    "name",
    "file",
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
    "step_time_ms",
]


def summarize_no_mhc_reports(input_dir: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in find_excel_files(input_dir):
        wb = load_workbook(path)
        if "Step Time" not in sheet_key_values(wb, "Summary") or "Strategy" not in wb.sheetnames:
            continue
        row = {
            "name": clean_report_name(path),
            "file": str(path),
            **strategy_values(wb),
            "step_time_ms": step_time_ms(wb),
        }
        rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize spec-path Excel reports without MHC operators."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing Excel reports.")
    parser.add_argument("--output", required=True, help="Output .xlsx summary path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = summarize_no_mhc_reports(args.input_dir)
    out = write_summary_xlsx(rows, COLUMNS, args.output)
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
