from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 路徑設定
# =========================
SUMMARY_PATH = Path("reports/summary.json")
OUT_DIR = Path("reports/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 讀取 summary.json
# =========================
with SUMMARY_PATH.open("r", encoding="utf-8") as f:
    summary = json.load(f)

# =========================
# 整理成 DataFrame
# =========================
rows = []
for model_name, info in summary.items():
    avg_scores = info.get("avg_scores", {})
    row = {
        "model": model_name,
        "num_samples": info.get("num_samples", 0),
        "format_compliance": avg_scores.get("format_compliance"),
        "task_alignment": avg_scores.get("task_alignment"),
        "naturalness": avg_scores.get("naturalness"),
        "insurance_realism": avg_scores.get("insurance_realism"),
        "hallucination_risk": avg_scores.get("hallucination_risk"),
    }
    rows.append(row)

df = pd.DataFrame(rows)

if df.empty:
    raise ValueError("summary.json 中沒有可用的 avg_scores")

# 額外加一欄：越高越好
# 原本 hallucination_risk 是 1~5，越低越好
# 轉成 hallucination_control：越高越好
df["hallucination_control"] = 6 - df["hallucination_risk"]

# 存成 csv 方便檢查
df.to_csv(OUT_DIR / "summary_scores.csv", index=False, encoding="utf-8-sig")
print("已輸出整理表格：", OUT_DIR / "summary_scores.csv")
print(df)

# =========================
# 圖 1：群組長條圖
# =========================
score_cols = [
    "format_compliance",
    "task_alignment",
    "naturalness",
    "insurance_realism",
    "hallucination_risk",
]

plot_df = df.set_index("model")[score_cols]

ax = plot_df.plot(kind="bar", figsize=(12, 6))
ax.set_title("Model Comparison on Average Scores")
ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_ylim(0, 5.5)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "grouped_bar_avg_scores.png", dpi=200)
plt.close()

# =========================
# 圖 2：熱力圖
# 不用 seaborn，純 matplotlib
# =========================
heatmap_cols = [
    "format_compliance",
    "task_alignment",
    "naturalness",
    "insurance_realism",
    "hallucination_risk",
]

heatmap_data = df.set_index("model")[heatmap_cols]

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(heatmap_data.values, aspect="auto")

ax.set_xticks(range(len(heatmap_cols)))
ax.set_xticklabels(heatmap_cols, rotation=20, ha="right")
ax.set_yticks(range(len(heatmap_data.index)))
ax.set_yticklabels(heatmap_data.index)

ax.set_title("Heatmap of Average Scores")

# 在格子中標數字
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        val = heatmap_data.iloc[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center")

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(OUT_DIR / "heatmap_avg_scores.png", dpi=200)
plt.close()

# =========================
# 圖 3：Hallucination Control 長條圖
# 越高越好，比 risk 更直觀
# =========================
control_df = df.set_index("model")[["hallucination_control"]]

ax = control_df.plot(kind="bar", figsize=(8, 5), legend=False)
ax.set_title("Hallucination Control (Higher is Better)")
ax.set_xlabel("Model")
ax.set_ylabel("Control Score")
ax.set_ylim(0, 5.5)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUT_DIR / "hallucination_control.png", dpi=200)
plt.close()

print("已完成繪圖，輸出資料夾：", OUT_DIR)
