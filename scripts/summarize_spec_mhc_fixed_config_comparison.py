from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.spec_excel_summary_common import value_as_float
    from scripts.summarize_spec_mhc_comparison import (
        COLUMNS,
        _comparison_values,
        _prefixed_no_mhc_values,
        match_key,
        summarize_no_mhc_reports,
    )
    from scripts.summarize_spec_mhc_excels import summarize_mhc_reports
except ModuleNotFoundError:
    from spec_excel_summary_common import value_as_float
    from summarize_spec_mhc_comparison import (
        COLUMNS,
        _comparison_values,
        _prefixed_no_mhc_values,
        match_key,
        summarize_no_mhc_reports,
    )
    from summarize_spec_mhc_excels import summarize_mhc_reports

try:
    from scripts.spec_excel_summary_common import write_summary_xlsx
except ModuleNotFoundError:
    from spec_excel_summary_common import write_summary_xlsx


Runner = Callable[[dict[str, object], Path], Path]


def summarize_fixed_config_comparison(
    *,
    mhc_input_dir: str | Path,
    generated_no_mhc_dir: str | Path,
    output: str | Path,
    default_model: str = "deepseek_v4_pro",
    runner: Runner | None = None,
) -> list[dict[str, object]]:
    mhc_rows = summarize_mhc_reports(mhc_input_dir)
    generated_no_mhc_dir = Path(generated_no_mhc_dir)
    runner = runner or run_no_mhc_spec_report_for_config

    for mhc_row in mhc_rows:
        key = match_key(mhc_row)
        report_path = generated_no_mhc_dir / f"{key}.xlsx"
        if not report_path.exists():
            config = no_mhc_config_from_mhc_row(mhc_row, default_model)
            runner(config, report_path)

    no_mhc_rows = summarize_no_mhc_reports(generated_no_mhc_dir)
    no_mhc_by_key = {match_key(row): row for row in no_mhc_rows}

    rows: list[dict[str, object]] = []
    for mhc_row in mhc_rows:
        key = match_key(mhc_row)
        no_mhc_row = no_mhc_by_key.get(key)
        row = {"match_key": key, **mhc_row}
        if no_mhc_row is None:
            row.update({"no_mhc_match_status": "missing"})
        else:
            row.update(_prefixed_no_mhc_values(no_mhc_row))
            row.update(_comparison_values(mhc_row, no_mhc_row))
        rows.append(row)

    write_summary_xlsx(rows, COLUMNS, output)
    return rows


def no_mhc_config_from_mhc_row(
    mhc_row: dict[str, object],
    default_model: str = "deepseek_v4_pro",
) -> dict[str, object]:
    return {
        "model": default_model,
        "hw": mhc_row.get("hw"),
        "seq_len": _int_value(mhc_row.get("seq_len")),
        "world_size": _int_value(mhc_row.get("world_size"), 1),
        "tp": _int_value(mhc_row.get("tp"), 1),
        "cp": _int_value(mhc_row.get("cp"), 1),
        "pp": _int_value(mhc_row.get("pp"), 1),
        "ep": _int_value(mhc_row.get("ep"), 1),
        "dp": _int_value(mhc_row.get("dp"), 1),
        "pp_schedule": mhc_row.get("pp_schedule") or "1f1b",
        "vpp_chunks": _int_value(mhc_row.get("vpp_chunks"), 1),
        "cp_kind": mhc_row.get("cp_kind") or "none",
        "micro_batch": _int_value(mhc_row.get("micro_batch"), 1),
        "global_batch": _int_value(mhc_row.get("global_batch"), 0),
        "zero_stage": _int_value(mhc_row.get("zero_stage"), 0),
        "optimizer": mhc_row.get("optimizer") or "adam",
        "hc_mult": 1,
    }


def run_no_mhc_spec_report_for_config(config: dict[str, object], report_path: Path) -> Path:
    from zrt.hardware.registry import load as load_hw
    from zrt.training.io.excel_exporter import export_estimate_excel
    from zrt.training.ir.builders import build_graph
    from zrt.training.models.flops import op_cost
    from zrt.training.search.estimator import estimate
    from zrt.training.search.training_search_util import (
        _ceil_nodes_for_world_size,
        _inferred_gpus_per_node,
        _load_model_spec,
        _make_strategy_from_config,
        _system_from_hw,
    )

    model = _load_model_spec(str(config["model"]), quant_preset=config.get("quant_preset") or None)
    model.seq_len = int(config.get("seq_len", model.seq_len))
    model.hc_mult = 1
    model.hc_sinkhorn_iters = 0

    hw = load_hw(str(config.get("hw", "nvidia_h100_sxm")))
    world_size = int(config.get("world_size", 1))
    gpus_per_node = _inferred_gpus_per_node(hw)
    system = _system_from_hw(
        hw,
        nodes=_ceil_nodes_for_world_size(world_size, gpus_per_node),
        gpus_per_node=gpus_per_node,
        world_size_override=world_size,
        host_mem_gb=float(config.get("host_mem_gb", 256.0) or 256.0),
    )

    strategy = _make_strategy_from_config(config)
    graph = build_graph(model, strategy)
    op_costs = {op.name: op_cost(op, model, system) for op in graph.ops}
    report = estimate(model, system, strategy, graph=graph)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    export_estimate_excel(
        report=report,
        graph=graph,
        model=model,
        system=system,
        strategy=strategy,
        op_costs=op_costs,
        output_path=report_path,
    )
    return report_path


def _int_value(value: object, default: int = 0) -> int:
    val = value_as_float(value, float(default))
    return int(val)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a fixed-config MHC vs no-MHC summary by regenerating no-MHC reports."
    )
    parser.add_argument("--mhc-input-dir", required=True, help="Directory containing MHC Excel reports.")
    parser.add_argument(
        "--generated-no-mhc-dir",
        required=True,
        help="Directory used to cache generated fixed-config no-MHC Excel reports.",
    )
    parser.add_argument("--output", required=True, help="Output .xlsx summary path.")
    parser.add_argument("--default-model", default="deepseek_v4_pro")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = summarize_fixed_config_comparison(
        mhc_input_dir=args.mhc_input_dir,
        generated_no_mhc_dir=args.generated_no_mhc_dir,
        output=args.output,
        default_model=args.default_model,
    )
    print(f"Wrote {len(rows)} rows to {Path(args.output)}")


if __name__ == "__main__":
    main()
