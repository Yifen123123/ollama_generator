from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_PATH = BASE_DIR / "outputs" / "fields_by_type_per_file.json"
DEFAULT_FORM_OUTPUT = BASE_DIR / "outputs" / "unit_form_fields_summary.json"
DEFAULT_CALL_OUTPUT = BASE_DIR / "outputs" / "unit_call_fields_summary.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_unit_summary(
    per_file_records: list[dict[str, Any]],
    field_key: str,
) -> dict[str, Any]:
    """
    field_key:
      - non_empty_form_fields
      - non_empty_call_fields
    """
    grouped: dict[str, Any] = defaultdict(
        lambda: {
            "files": [],
            "categories": set(),
            "fields_union": set(),
            "field_counts": defaultdict(int),
        }
    )

    for record in per_file_records:
        unit = str(record.get("unit", "UNKNOWN_UNIT")).strip() or "UNKNOWN_UNIT"
        category = str(record.get("category", "UNKNOWN_CATEGORY")).strip() or "UNKNOWN_CATEGORY"
        file_id = str(record.get("file_id", "")).strip()

        fields = record.get(field_key, [])
        if not isinstance(fields, list):
            fields = []

        grouped[unit]["files"].append(file_id)
        grouped[unit]["categories"].add(category)

        for field in fields:
            grouped[unit]["fields_union"].add(field)
            grouped[unit]["field_counts"][field] += 1

    summary: dict[str, Any] = {}

    for unit, info in grouped.items():
        summary[unit] = {
            "file_count": len(info["files"]),
            "files": sorted(info["files"]),
            "categories": sorted(info["categories"]),
            "fields_union": sorted(info["fields_union"]),
            "field_counts": dict(sorted(info["field_counts"].items())),
        }

    return dict(sorted(summary.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize fields by unit and separate form/call outputs."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to fields_by_type_per_file.json",
    )
    parser.add_argument(
        "--form_output",
        type=str,
        default=str(DEFAULT_FORM_OUTPUT),
        help="Output path for unit form fields summary",
    )
    parser.add_argument(
        "--call_output",
        type=str,
        default=str(DEFAULT_CALL_OUTPUT),
        help="Output path for unit call fields summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    form_output = Path(args.form_output)
    call_output = Path(args.call_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    per_file_records = read_json(input_path)

    if not isinstance(per_file_records, list):
        raise ValueError("fields_by_type_per_file.json 必須是 list")

    form_summary = build_unit_summary(
        per_file_records=per_file_records,
        field_key="non_empty_form_fields",
    )

    call_summary = build_unit_summary(
        per_file_records=per_file_records,
        field_key="non_empty_call_fields",
    )

    write_json(form_output, form_summary)
    write_json(call_output, call_summary)

    print(f"Saved form summary to: {form_output}")
    print(f"Saved call summary to: {call_output}")
    print(f"Total units (form): {len(form_summary)}")
    print(f"Total units (call): {len(call_summary)}")


if __name__ == "__main__":
    main()
