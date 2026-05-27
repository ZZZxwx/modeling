from __future__ import annotations

import argparse
from pathlib import Path

try:
    from scripts.spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        mhc_time_from_ops,
        percent_as_ratio,
        sheet_key_values,
        step_time_ms,
        value_as_float,
        write_summary_xlsx,
    )
except ModuleNotFoundError:
    from spec_excel_summary_common import (
        clean_report_name,
        find_excel_files,
        load_workbook,
        mhc_time_from_ops,
        percent_as_ratio,
        sheet_key_values,
        step_time_ms,
        value_as_float,
        write_summary_xlsx,
    )


COLUMNS = ["name", "file", "step_time_ms", "mhc_time_ms", "mhc_step_share"]


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
        else:
            mhc_ms = mhc_time_from_ops(wb)
            share = mhc_ms / step_ms if step_ms else 0.0
        rows.append({
            "name": clean_report_name(path),
            "file": str(path),
            "step_time_ms": step_ms,
            "mhc_time_ms": mhc_ms,
            "mhc_step_share": share,
        })
    return rows


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
