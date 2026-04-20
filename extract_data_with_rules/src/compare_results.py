from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# =========================
# 預設路徑
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "src" else Path(__file__).resolve().parent

DEFAULT_FORM_PATH = BASE_DIR / "outputs" / "normalized_forms.json"
DEFAULT_CALL_PATH = BASE_DIR / "outputs" / "extracted_calls.json"
DEFAULT_FIELD_ROLES_PATH = BASE_DIR / "data" / "field_roles.json"

DEFAULT_COMPARE_OUTPUT = BASE_DIR / "outputs" / "compare_results.json"
DEFAULT_SUMMARY_OUTPUT = BASE_DIR / "outputs" / "compare_summary.json"


# =========================
# 基本 I/O
# =========================
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# 工具函式
# =========================
def normalize_value(value: Any) -> str | None:
    if value is None:
        return None

    if not isinstance(value, str):
        value = str(value)

    value = value.strip()

    if value == "":
        return None

    lowered = value.lower()
    if lowered in {"none", "null", "nan"}:
        return None

    return value


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def index_by_file_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in items:
        file_id = str(item.get("file_id", "")).strip()
        if file_id:
            result[file_id] = item
    return result


def is_same_entity(form_value: str, call_value: str) -> bool:
    """
    entity 型欄位的寬鬆比對：
    - 完全相等
    - 一方包含另一方
    """
    if form_value == call_value:
        return True

    if form_value in call_value or call_value in form_value:
        return True

    return False


def compare_entity_field(form_value: str | None, call_value: str | None, role_info: dict[str, Any]) -> str:
    primary_role = role_info.get("primary_role", "unclassified")
    may_appear_in_call = bool(role_info.get("may_appear_in_call", False))
    may_come_from_system = bool(role_info.get("may_come_from_system", False))

    if form_value is None and call_value is None:
        return "both_empty"

    if form_value is not None and call_value is not None:
        if is_same_entity(form_value, call_value):
            return "match"
        return "mismatch"

    if form_value is None and call_value is not None:
        if may_appear_in_call:
            return "call_extra_possible_detail"
        return "call_extra_needs_review"

    # form_value is not None and call_value is None
    if primary_role == "system_enriched" or may_come_from_system:
        return "form_only_system_ok"

    if primary_role == "call_observable":
        return "form_only_possible_missed_extraction"

    return "form_only"


def compare_summary_field(form_value: str | None, call_value: str | None, role_info: dict[str, Any]) -> str:
    primary_role = role_info.get("primary_role", "unclassified")
    may_come_from_system = bool(role_info.get("may_come_from_system", False))

    if form_value is None and call_value is None:
        return "both_empty"

    if form_value is not None and call_value is not None:
        if form_value == call_value:
            return "match"
        return "both_present_needs_semantic_review"

    if form_value is None and call_value is not None:
        return "call_extra_summary_variant"

    # form_value is not None and call_value is None
    if primary_role == "summary_or_rewritten":
        return "form_only_summary_ok"

    if may_come_from_system:
        return "form_only_system_ok"

    return "form_only"


def compare_field(form_value: str | None, call_value: str | None, role_info: dict[str, Any]) -> str:
    comparison_mode = role_info.get("comparison_mode", "entity")

    if comparison_mode == "summary":
        return compare_summary_field(form_value, call_value, role_info)

    return compare_entity_field(form_value, call_value, role_info)


def compare_one_file(
    file_id: str,
    form_item: dict[str, Any] | None,
    call_item: dict[str, Any] | None,
    field_roles: dict[str, dict[str, Any]],
    all_fields: list[str],
) -> dict[str, Any]:
    form_fields = {}
    call_fields = {}

    if form_item:
        form_fields = form_item.get("structured_fields", {}) or {}

    if call_item:
        call_fields = call_item.get("structured_fields", {}) or {}

    field_compare: dict[str, Any] = {}

    for field in all_fields:
        role_info = field_roles.get(
            field,
            {
                "primary_role": "unclassified",
                "may_appear_in_call": False,
                "may_come_from_system": False,
                "comparison_mode": "entity",
            },
        )

        form_value = normalize_value(form_fields.get(field))
        call_value = normalize_value(call_fields.get(field))

        status = compare_field(
            form_value=form_value,
            call_value=call_value,
            role_info=role_info,
        )

        field_compare[field] = {
            "role_info": role_info,
            "form_value": form_value,
            "call_value": call_value,
            "status": status,
        }

    return {
        "file_id": file_id,
        "form_exists": form_item is not None,
        "call_exists": call_item is not None,
        "field_compare": field_compare,
    }


def build_summary(compare_results: list[dict[str, Any]], all_fields: list[str], field_roles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    field_stats: dict[str, Any] = {}

    for field in all_fields:
        role_info = field_roles.get(
            field,
            {
                "primary_role": "unclassified",
                "may_appear_in_call": False,
                "may_come_from_system": False,
                "comparison_mode": "entity",
            },
        )

        stats = {
            "role_info": role_info,
            "total_files": 0,
            "form_has_value_count": 0,
            "call_has_value_count": 0,
            "status_counts": {},
        }

        for item in compare_results:
            fc = item["field_compare"][field]
            form_value = fc["form_value"]
            call_value = fc["call_value"]
            status = fc["status"]

            stats["total_files"] += 1

            if form_value is not None:
                stats["form_has_value_count"] += 1

            if call_value is not None:
                stats["call_has_value_count"] += 1

            stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1

        total_files = stats["total_files"]
        stats["form_has_value_ratio"] = safe_ratio(stats["form_has_value_count"], total_files)
        stats["call_has_value_ratio"] = safe_ratio(stats["call_has_value_count"], total_files)

        field_stats[field] = stats

    overall_status_counts: dict[str, int] = {}

    for item in compare_results:
        for field in all_fields:
            status = item["field_compare"][field]["status"]
            overall_status_counts[status] = overall_status_counts.get(status, 0) + 1

    total_field_comparisons = sum(overall_status_counts.values())

    summary = {
        "total_files": len(compare_results),
        "total_fields": len(all_fields),
        "total_field_comparisons": total_field_comparisons,
        "overall_status_counts": overall_status_counts,
        "overall_status_ratios": {
            k: safe_ratio(v, total_field_comparisons)
            for k, v in overall_status_counts.items()
        },
        "field_stats": field_stats,
    }
    return summary


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare normalized forms and extracted calls with field-role-aware rules.")
    parser.add_argument(
        "--form_input",
        type=str,
        default=str(DEFAULT_FORM_PATH),
        help="Path to normalized_forms.json",
    )
    parser.add_argument(
        "--call_input",
        type=str,
        default=str(DEFAULT_CALL_PATH),
        help="Path to extracted_calls.json",
    )
    parser.add_argument(
        "--field_roles",
        type=str,
        default=str(DEFAULT_FIELD_ROLES_PATH),
        help="Path to field_roles.json",
    )
    parser.add_argument(
        "--compare_output",
        type=str,
        default=str(DEFAULT_COMPARE_OUTPUT),
        help="Output path for compare_results.json",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=str(DEFAULT_SUMMARY_OUTPUT),
        help="Output path for compare_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    form_input_path = Path(args.form_input)
    call_input_path = Path(args.call_input)
    field_roles_path = Path(args.field_roles)
    compare_output_path = Path(args.compare_output)
    summary_output_path = Path(args.summary_output)

    if not form_input_path.exists():
        raise FileNotFoundError(f"Form input not found: {form_input_path}")

    if not call_input_path.exists():
        raise FileNotFoundError(f"Call input not found: {call_input_path}")

    if not field_roles_path.exists():
        raise FileNotFoundError(f"Field roles file not found: {field_roles_path}")

    forms = read_json(form_input_path)
    calls = read_json(call_input_path)
    field_roles = read_json(field_roles_path)

    if not isinstance(forms, list):
        raise ValueError("normalized_forms.json 必須是 list")

    if not isinstance(calls, list):
        raise ValueError("extracted_calls.json 必須是 list")

    if not isinstance(field_roles, dict):
        raise ValueError("field_roles.json 必須是 object")

    form_map = index_by_file_id(forms)
    call_map = index_by_file_id(calls)

    all_fields_set: set[str] = set(field_roles.keys())

    for item in forms:
        structured_fields = item.get("structured_fields", {}) or {}
        all_fields_set.update(structured_fields.keys())

    for item in calls:
        structured_fields = item.get("structured_fields", {}) or {}
        all_fields_set.update(structured_fields.keys())

    all_fields = sorted(all_fields_set)
    all_file_ids = sorted(set(form_map.keys()) | set(call_map.keys()))

    compare_results: list[dict[str, Any]] = []

    for file_id in all_file_ids:
        result = compare_one_file(
            file_id=file_id,
            form_item=form_map.get(file_id),
            call_item=call_map.get(file_id),
            field_roles=field_roles,
            all_fields=all_fields,
        )
        compare_results.append(result)

    summary = build_summary(
        compare_results=compare_results,
        all_fields=all_fields,
        field_roles=field_roles,
    )

    write_json(compare_output_path, compare_results)
    write_json(summary_output_path, summary)

    print(f"Compare results saved to: {compare_output_path}")
    print(f"Summary saved to: {summary_output_path}")
    print(f"Total files: {summary['total_files']}")
    print(f"Total fields: {summary['total_fields']}")
    print(f"Total field comparisons: {summary['total_field_comparisons']}")


if __name__ == "__main__":
    main()
