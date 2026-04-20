from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_COMPARE_PATH = BASE_DIR / "outputs" / "compare_results.json"
DEFAULT_TYPE_MAP_PATH = BASE_DIR / "data" / "file_type_mapping.json"
DEFAULT_PER_FILE_OUTPUT = BASE_DIR / "outputs" / "fields_by_type_per_file.json"
DEFAULT_SUMMARY_OUTPUT = BASE_DIR / "outputs" / "fields_by_type_summary.json"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_value(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if value == "":
        return None
    if value.lower() in {"none", "null", "nan"}:
        return None
    return value


def extract_non_empty_fields(field_compare: dict[str, Any], value_key: str) -> list[str]:
    result = []
    for field_name, info in field_compare.items():
        value = normalize_value(info.get(value_key))
        if value is not None:
            result.append(field_name)
    return sorted(result)


def build_type_key(unit: str, category: str) -> str:
    return f"{unit}｜{category}"


def get_case_info(file_id: str, file_type_mapping: dict[str, Any]) -> dict[str, str]:
    """
    支援兩種 mapping 格式：

    1. 新格式
    {
      "001": {"unit": "...", "category": "..."}
    }

    2. 舊格式
    {
      "001": "保單借款"
    }
    舊格式會被轉成 unit=UNKNOWN_UNIT, category=<value>
    """
    raw = file_type_mapping.get(file_id)

    if raw is None:
        return {
            "unit": "UNKNOWN_UNIT",
            "category": "UNKNOWN_CATEGORY",
            "type_key": build_type_key("UNKNOWN_UNIT", "UNKNOWN_CATEGORY"),
        }

    if isinstance(raw, dict):
        unit = str(raw.get("unit", "UNKNOWN_UNIT")).strip() or "UNKNOWN_UNIT"
        category = str(raw.get("category", "UNKNOWN_CATEGORY")).strip() or "UNKNOWN_CATEGORY"
        return {
            "unit": unit,
            "category": category,
            "type_key": build_type_key(unit, category),
        }

    # backward compatibility
    category = str(raw).strip() or "UNKNOWN_CATEGORY"
    unit = "UNKNOWN_UNIT"
    return {
        "unit": unit,
        "category": category,
        "type_key": build_type_key(unit, category),
    }


def build_per_file_records(
    compare_results: list[dict[str, Any]],
    file_type_mapping: dict[str, Any],
) -> list[dict[str, Any]]:
    records = []

    for item in compare_results:
        file_id = str(item.get("file_id", "")).strip()
        case_info = get_case_info(file_id, file_type_mapping)

        field_compare = item.get("field_compare", {}) or {}

        non_empty_form_fields = extract_non_empty_fields(field_compare, "form_value")
        non_empty_call_fields = extract_non_empty_fields(field_compare, "call_value")

        records.append(
            {
                "file_id": file_id,
                "unit": case_info["unit"],
                "category": case_info["category"],
                "type_key": case_info["type_key"],
                "non_empty_form_fields": non_empty_form_fields,
                "non_empty_call_fields": non_empty_call_fields,
            }
        )

    return records


def build_type_summary(per_file_records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Any] = defaultdict(
        lambda: {
            "unit": "",
            "category": "",
            "files": [],
            "form_fields_union": set(),
            "call_fields_union": set(),
            "form_field_counts": defaultdict(int),
            "call_field_counts": defaultdict(int),
        }
    )

    for record in per_file_records:
        type_key = record["type_key"]
        file_id = record["file_id"]
        unit = record["unit"]
        category = record["category"]

        grouped[type_key]["unit"] = unit
        grouped[type_key]["category"] = category
        grouped[type_key]["files"].append(file_id)

        for field in record["non_empty_form_fields"]:
            grouped[type_key]["form_fields_union"].add(field)
            grouped[type_key]["form_field_counts"][field] += 1

        for field in record["non_empty_call_fields"]:
            grouped[type_key]["call_fields_union"].add(field)
            grouped[type_key]["call_field_counts"][field] += 1

    summary: dict[str, Any] = {}

    for type_key, info in grouped.items():
        summary[type_key] = {
            "unit": info["unit"],
            "category": info["category"],
            "file_count": len(info["files"]),
            "files": sorted(info["files"]),
            "form_fields_union": sorted(info["form_fields_union"]),
            "call_fields_union": sorted(info["call_fields_union"]),
            "form_field_counts": dict(sorted(info["form_field_counts"].items())),
            "call_field_counts": dict(sorted(info["call_field_counts"].items())),
        }

    return dict(sorted(summary.items()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize non-empty form/call fields by (unit + category) from compare_results.json"
    )
    parser.add_argument(
        "--compare_input",
        type=str,
        default=str(DEFAULT_COMPARE_PATH),
        help="Path to compare_results.json",
    )
    parser.add_argument(
        "--type_mapping",
        type=str,
        default=str(DEFAULT_TYPE_MAP_PATH),
        help="Path to file_type_mapping.json",
    )
    parser.add_argument(
        "--per_file_output",
        type=str,
        default=str(DEFAULT_PER_FILE_OUTPUT),
        help="Output path for per-file field summary",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=str(DEFAULT_SUMMARY_OUTPUT),
        help="Output path for type-level field summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compare_path = Path(args.compare_input)
    type_map_path = Path(args.type_mapping)
    per_file_output = Path(args.per_file_output)
    summary_output = Path(args.summary_output)

    if not compare_path.exists():
        raise FileNotFoundError(f"compare_results.json not found: {compare_path}")

    if not type_map_path.exists():
        raise FileNotFoundError(f"file_type_mapping.json not found: {type_map_path}")

    compare_results = read_json(compare_path)
    file_type_mapping = read_json(type_map_path)

    if not isinstance(compare_results, list):
        raise ValueError("compare_results.json 必須是 list")

    if not isinstance(file_type_mapping, dict):
        raise ValueError("file_type_mapping.json 必須是 object")

    per_file_records = build_per_file_records(compare_results, file_type_mapping)
    type_summary = build_type_summary(per_file_records)

    write_json(per_file_output, per_file_records)
    write_json(summary_output, type_summary)

    print(f"Saved per-file output to: {per_file_output}")
    print(f"Saved type summary to: {summary_output}")
    print(f"Total files processed: {len(per_file_records)}")
    print(f"Total type keys found: {len(type_summary)}")


if __name__ == "__main__":
    main()
