from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_PATH = BASE_DIR / "outputs" / "normalized_forms_without_calls.json"
DEFAULT_TYPE_MAP_PATH = BASE_DIR / "data" / "file_type_mapping_2.json"
DEFAULT_JSON_OUTPUT = BASE_DIR / "outputs" / "forms_without_calls_fields_by_type.json"
DEFAULT_TXT_OUTPUT = BASE_DIR / "outputs" / "forms_without_calls_fields_by_type.txt"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def build_type_key(unit: str, category: str) -> str:
    return f"{unit}｜{category}"


def get_case_info(file_id: str, file_type_mapping: dict[str, Any]) -> dict[str, str]:
    raw = file_type_mapping.get(file_id)

    if raw is None:
        unit = "UNKNOWN_UNIT"
        category = "UNKNOWN_CATEGORY"
        return {
            "unit": unit,
            "category": category,
            "type_key": build_type_key(unit, category),
        }

    if isinstance(raw, dict):
        unit = str(raw.get("unit", "UNKNOWN_UNIT")).strip() or "UNKNOWN_UNIT"
        category = str(raw.get("category", "UNKNOWN_CATEGORY")).strip() or "UNKNOWN_CATEGORY"
        return {
            "unit": unit,
            "category": category,
            "type_key": build_type_key(unit, category),
        }

    # 舊格式相容
    unit = "UNKNOWN_UNIT"
    category = str(raw).strip() or "UNKNOWN_CATEGORY"
    return {
        "unit": unit,
        "category": category,
        "type_key": build_type_key(unit, category),
    }


def extract_non_empty_fields(structured_fields: dict[str, Any]) -> list[str]:
    result = []
    for field_name, value in structured_fields.items():
        if normalize_value(value) is not None:
            result.append(field_name)
    return sorted(result)


def build_summary(
    normalized_forms: list[dict[str, Any]],
    file_type_mapping: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, Any] = defaultdict(
        lambda: {
            "unit": "",
            "category": "",
            "files": [],
            "fields_union": set(),
            "field_counts": defaultdict(int),
        }
    )

    for item in normalized_forms:
        file_id = str(item.get("file_id", "")).strip()
        case_info = get_case_info(file_id, file_type_mapping)

        structured_fields = item.get("structured_fields", {}) or {}
        non_empty_fields = extract_non_empty_fields(structured_fields)

        type_key = case_info["type_key"]
        grouped[type_key]["unit"] = case_info["unit"]
        grouped[type_key]["category"] = case_info["category"]
        grouped[type_key]["files"].append(file_id)

        for field in non_empty_fields:
            grouped[type_key]["fields_union"].add(field)
            grouped[type_key]["field_counts"][field] += 1

    summary: dict[str, Any] = {}

    for type_key, info in grouped.items():
        summary[type_key] = {
            "unit": info["unit"],
            "category": info["category"],
            "file_count": len(info["files"]),
            "files": sorted(info["files"]),
            "fields_union": sorted(info["fields_union"]),
            "field_counts": dict(sorted(info["field_counts"].items())),
        }

    return dict(sorted(summary.items()))


def summary_to_text(summary: dict[str, Any]) -> str:
    lines: list[str] = []

    for type_key, info in summary.items():
        unit = info.get("unit", "UNKNOWN_UNIT")
        category = info.get("category", "UNKNOWN_CATEGORY")
        file_count = info.get("file_count", 0)
        files = info.get("files", [])
        fields_union = info.get("fields_union", [])
        field_counts = info.get("field_counts", {})

        lines.append(f"【{type_key}】")
        lines.append(f"會辦單位: {unit}")
        lines.append(f"會辦單類別: {category}")
        lines.append(f"檔案數: {file_count}")
        lines.append(f"檔案: {', '.join(files) if files else '無'}")
        lines.append("")
        lines.append("欄位:")
        if fields_union:
            for field in fields_union:
                lines.append(f"- {field}")
        else:
            lines.append("- 無")

        lines.append("")
        lines.append("欄位出現次數:")
        if field_counts:
            for field, count in field_counts.items():
                lines.append(f"- {field}: {count}")
        else:
            lines.append("- 無")

        lines.append("")
        lines.append("-" * 50)
        lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize non-empty fields from normalized forms without calls by (unit + category)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="Path to normalized_forms_without_calls.json",
    )
    parser.add_argument(
        "--type_mapping",
        type=str,
        default=str(DEFAULT_TYPE_MAP_PATH),
        help="Path to file_type_mapping_2.json",
    )
    parser.add_argument(
        "--json_output",
        type=str,
        default=str(DEFAULT_JSON_OUTPUT),
        help="Output path for JSON summary",
    )
    parser.add_argument(
        "--txt_output",
        type=str,
        default=str(DEFAULT_TXT_OUTPUT),
        help="Output path for TXT summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    type_map_path = Path(args.type_mapping)
    json_output = Path(args.json_output)
    txt_output = Path(args.txt_output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not type_map_path.exists():
        raise FileNotFoundError(f"Type mapping file not found: {type_map_path}")

    normalized_forms = read_json(input_path)
    file_type_mapping = read_json(type_map_path)

    if not isinstance(normalized_forms, list):
        raise ValueError("normalized_forms_without_calls.json 必須是 list")

    if not isinstance(file_type_mapping, dict):
        raise ValueError("file_type_mapping_2.json 必須是 object")

    summary = build_summary(normalized_forms, file_type_mapping)
    summary_text = summary_to_text(summary)

    write_json(json_output, summary)
    write_text(txt_output, summary_text)

    print(f"Saved JSON summary to: {json_output}")
    print(f"Saved TXT summary to: {txt_output}")
    print(f"Total forms processed: {len(normalized_forms)}")
    print(f"Total type keys found: {len(summary)}")


if __name__ == "__main__":
    main()
