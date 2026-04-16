from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# =========================
# 設定區
# =========================

INPUT_JSON = Path("eval_reports_v2/summary.json")
OUTPUT_DIR = Path("eval_reports_v2/plots")

# 你最在意的指標
METRICS = [
    "avg_faithfulness_to_call",
    "avg_hallucination_control",
    "avg_main_issue_understanding",
    "avg_personal_info_extraction",
    "avg_bullet_form_usability",
    "avg_similarity_to_original_form",
]

# 顯示名稱
METRIC_LABELS = {
    "avg_faithfulness_to_call": "Faithfulness to Call",
    "avg_hallucination_control": "Hallucination Control",
    "avg_main_issue_understanding": "Main Issue Understanding",
    "avg_personal_info_extraction": "Personal Info Extraction",
    "avg_bullet_form_usability": "Bullet Form Usability",
    "avg_similarity_to_original_form": "Similarity to Original Form",
    "avg_overall_score": "Overall Score",
    "avg_elapsed_seconds": "Avg Elapsed Seconds",
}


def load_summary(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    for model_name, metrics in data.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_grouped_bar_chart(df: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    plot_df = df[["model"] + metrics].copy()
    plot_df = plot_df.set_index("model")

    ax = plot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title("Model Comparison on Core Evaluation Metrics")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend([METRIC_LABELS.get(m, m) for m in metrics], bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_single_metric_bar_chart(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    plot_df = df[["model", metric]].copy().sort_values(by=metric, ascending=False)

    ax = plot_df.plot(kind="bar", x="model", y=metric, figsize=(10, 5), legend=False)
    ax.set_title(METRIC_LABELS.get(metric, metric))
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    for i, value in enumerate(plot_df[metric]):
        ax.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_overall_score_chart(df: pd.DataFrame, output_path: Path) -> None:
    if "avg_overall_score" not in df.columns:
        return

    plot_df = df[["model", "avg_overall_score"]].copy().sort_values(by="avg_overall_score", ascending=False)

    ax = plot_df.plot(kind="bar", x="model", y="avg_overall_score", figsize=(10, 5), legend=False)
    ax.set_title("Overall Score")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)

    for i, value in enumerate(plot_df["avg_overall_score"]):
        if pd.notna(value):
            ax.text(i, value + 0.02, f"{value:.3f}", ha="center")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_elapsed_time_chart(df: pd.DataFrame, output_path: Path) -> None:
    if "avg_elapsed_seconds" not in df.columns:
        return

    plot_df = df[["model", "avg_elapsed_seconds"]].copy().sort_values(by="avg_elapsed_seconds", ascending=False)

    ax = plot_df.plot(kind="bar", x="model", y="avg_elapsed_seconds", figsize=(10, 5), legend=False)
    ax.set_title("Average Elapsed Seconds")
    ax.set_xlabel("Model")
    ax.set_ylabel("Seconds")

    for i, value in enumerate(plot_df["avg_elapsed_seconds"]):
        if pd.notna(value):
            ax.text(i, value, f"{value:.2f}", ha="center", va="bottom")

    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_radar_chart(df: pd.DataFrame, metrics: list[str], output_path: Path) -> None:
    """
    額外提供一張雷達圖。模型少時好看，但不一定最好讀。
    """
    import math

    plot_df = df[["model"] + metrics].copy()

    labels = [METRIC_LABELS.get(m, m) for m in metrics]
    num_vars = len(metrics)

    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in plot_df.iterrows():
        values = [float(row[m]) if pd.notna(row[m]) else 0.0 for m in metrics]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=row["model"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1)
    ax.set_title("Radar Chart of Core Metrics")
    ax.legend(bbox_to_anchor=(1.15, 1.1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def export_table_csv(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False, encoding="utf-8-sig")


def main() -> None:
    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"找不到 summary.json: {INPUT_JSON}")

    ensure_output_dir(OUTPUT_DIR)

    df = load_summary(INPUT_JSON)

    # 只保留存在的 metrics
    available_metrics = [m for m in METRICS if m in df.columns]

    if not available_metrics:
        raise ValueError("summary.json 中找不到指定的評估指標。")

    # 輸出表格
    export_table_csv(df, OUTPUT_DIR / "summary_table.csv")

    # 1. 核心總覽圖
    save_grouped_bar_chart(
        df=df,
        metrics=available_metrics,
        output_path=OUTPUT_DIR / "core_metrics_grouped_bar.png",
    )

    # 2. 各單指標圖
    for metric in available_metrics:
        save_single_metric_bar_chart(
            df=df,
            metric=metric,
            output_path=OUTPUT_DIR / f"{metric}.png",
        )

    # 3. overall score
    save_overall_score_chart(
        df=df,
        output_path=OUTPUT_DIR / "avg_overall_score.png",
    )

    # 4. elapsed time
    save_elapsed_time_chart(
        df=df,
        output_path=OUTPUT_DIR / "avg_elapsed_seconds.png",
    )

    # 5. radar chart
    save_radar_chart(
        df=df,
        metrics=available_metrics,
        output_path=OUTPUT_DIR / "core_metrics_radar.png",
    )

    print("✅ 圖表輸出完成")
    print(f"輸出資料夾：{OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
