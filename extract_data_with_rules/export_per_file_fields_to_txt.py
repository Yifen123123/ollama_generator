from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_PATH = BASE_DIR / "outputs" / "fields_by_type_per_file.json"
DEFAULT_FORM_TXT_PATH = BASE_DIR / "outputs" / "per_file_form_fields.txt"
DEFAULT_CALL_TXT_PATH = BASE_DIR / "outputs" / "per_file_call_fields.txt"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def build_txt(records: list[dict[str, Any]], field_key: str) -> str:
    """
    field_key:
      - non_empty_form_fields
      - non_empty_call_fields
    """
    lines: list[str] = []

    for idx, record in enumerate(records, start=1):
        unit = str(record.get("unit", "UNKNOWN_UNIT")).strip() or "UNKNOWN_UNIT"
        category = str(record.get("category", "UNKNOWN_CATEGORY")).strip() or "UNKNOWN_CATEGORY"
        fields = record.get(field_key, [])

        if not isinstance(fields, list):
            fields = []

        lines.append(f"【第 {idx} 筆】")
        lines.append(f"會辦單位: {unit}")
        lines.append(f"會辦單類別: {category}")
        lines.append("欄位:")

        if fields:
            for field in fields:
                lines.append(f"- {field}")
        else:
            lines.append("- 無")

        lines.append("")
        lines.append("-" * 50)
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export per-file form/call fields from fields_by_type_per_file.json to txt files."
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
        default=str(DEFAULT_FORM_TXT_PATH),
        help="Output path for per-file form fields txt",
    )
    parser.add_argument(
        "--call_output",
        type=str,
        default=str(DEFAULT_CALL_TXT_PATH),
        help="Output path for per-file call fields txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    form_output_path = Path(args.form_output)
    call_output_path = Path(args.call_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records = read_json(input_path)

    if not isinstance(records, list):
        raise ValueError("fields_by_type_per_file.json 必須是 list")

    form_text = build_txt(records, "non_empty_form_fields")
    call_text = build_txt(records, "non_empty_call_fields")

    write_text(form_output_path, form_text)
    write_text(call_output_path, call_text)

    print(f"Saved form txt to: {form_output_path}")
    print(f"Saved call txt to: {call_output_path}")
    print(f"Total records processed: {len(records)}")


if __name__ == "__main__":
    main()
