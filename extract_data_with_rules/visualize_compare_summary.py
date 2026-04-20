from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# =========================
# 預設路徑
# =========================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_PATH = BASE_DIR / "outputs" / "compare_summary.json"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "charts"


# =========================
# 基本 I/O
# =========================
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# =========================
# 資料整理
# =========================
def extract_field_stats(summary: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    field_stats = summary.get("field_stats", {})
    if not isinstance(field_stats, dict) or not field_stats:
        raise ValueError("compare_summary.json 中找不到有效的 field_stats")

    # 保持 JSON 原本順序；若想排序可在這裡改
    fields = list(field_stats.keys())
    stats_list = [field_stats[field] for field in fields]
    return fields, stats_list


def collect_all_statuses(stats_list: list[dict[str, Any]]) -> list[str]:
    statuses: set[str] = set()
    for item in stats_list:
        status_counts = item.get("status_counts", {})
        if isinstance(status_counts, dict):
            statuses.update(status_counts.keys())

    preferred_order = [
        "match",
        "mismatch",
        "call_extra_possible_detail",
        "call_extra_needs_review",
        "call_extra_summary_variant",
        "form_only_system_ok",
        "form_only_possible_missed_extraction",
        "form_only_summary_ok",
        "form_only",
        "both_present_needs_semantic_review",
        "both_empty",
        "unknown",
    ]

    ordered = [s for s in preferred_order if s in statuses]
    remaining = sorted(statuses - set(ordered))
    return ordered + remaining


# =========================
# 繪圖
# =========================
def plot_value_ratio_chart(
    fields: list[str],
    stats_list: list[dict[str, Any]],
    output_path: Path,
) -> None:
    form_ratios = [item.get("form_has_value_ratio", 0.0) for item in stats_list]
    call_ratios = [item.get("call_has_value_ratio", 0.0) for item in stats_list]

    x = np.arange(len(fields))
    width = 0.38

    plt.figure(figsize=(max(12, len(fields) * 0.6), 7))
    plt.bar(x - width / 2, form_ratios, width=width, label="Form has value ratio")
    plt.bar(x + width / 2, call_ratios, width=width, label="Call has value ratio")

    plt.xticks(x, fields, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel("Ratio")
    plt.title("Field Value Presence Ratio: Form vs Call")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_status_stacked_chart(
    fields: list[str],
    stats_list: list[dict[str, Any]],
    output_path: Path,
    normalize: bool = False,
) -> None:
    statuses = collect_all_statuses(stats_list)
    x = np.arange(len(fields))
    bottom = np.zeros(len(fields), dtype=float)

    plt.figure(figsize=(max(14, len(fields) * 0.7), 8))

    for status in statuses:
        values = []
        for item in stats_list:
            status_counts = item.get("status_counts", {})
            total_files = item.get("total_files", 0)
            raw_value = status_counts.get(status, 0)

            if normalize:
                value = raw_value / total_files if total_files else 0.0
            else:
                value = raw_value

            values.append(value)

        values_arr = np.array(values, dtype=float)
        plt.bar(x, values_arr, bottom=bottom, label=status)
        bottom += values_arr

    plt.xticks(x, fields, rotation=45, ha="right")
    plt.ylabel("Ratio" if normalize else "Count")
    plt.title(
        "Field Status Distribution (Normalized)"
        if normalize
        else "Field Status Distribution (Counts)"
    )
    plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_overall_status_chart(
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    overall_status_counts = summary.get("overall_status_counts", {})
    if not isinstance(overall_status_counts, dict) or not overall_status_counts:
        raise ValueError("compare_summary.json 中找不到有效的 overall_status_counts")

    statuses = list(overall_status_counts.keys())
    counts = [overall_status_counts[s] for s in statuses]

    x = np.arange(len(statuses))

    plt.figure(figsize=(max(10, len(statuses) * 0.7), 6))
    plt.bar(x, counts)
    plt.xticks(x, statuses, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title("Overall Status Counts")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize compare_summary.json into charts."
    )
    parser.add_argument(
        "--summary_input",
        type=str,
        default=str(DEFAULT_SUMMARY_PATH),
        help="Path to compare_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save chart images",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary_path = Path(args.summary_input)
    output_dir = Path(args.output_dir)

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    ensure_dir(output_dir)

    summary = read_json(summary_path)
    fields, stats_list = extract_field_stats(summary)

    ratio_chart_path = output_dir / "field_value_ratio_chart.png"
    status_count_chart_path = output_dir / "field_status_stacked_counts.png"
    status_ratio_chart_path = output_dir / "field_status_stacked_ratios.png"
    overall_status_chart_path = output_dir / "overall_status_counts.png"

    plot_value_ratio_chart(
        fields=fields,
        stats_list=stats_list,
        output_path=ratio_chart_path,
    )

    plot_status_stacked_chart(
        fields=fields,
        stats_list=stats_list,
        output_path=status_count_chart_path,
        normalize=False,
    )

    plot_status_stacked_chart(
        fields=fields,
        stats_list=stats_list,
        output_path=status_ratio_chart_path,
        normalize=True,
    )

    plot_overall_status_chart(
        summary=summary,
        output_path=overall_status_chart_path,
    )

    print(f"Saved: {ratio_chart_path}")
    print(f"Saved: {status_count_chart_path}")
    print(f"Saved: {status_ratio_chart_path}")
    print(f"Saved: {overall_status_chart_path}")


if __name__ == "__main__":
    main()
