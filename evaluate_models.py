from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ollama import Client


# =========================
# 設定區
# =========================

OLLAMA_HOST = "http://10.67.75.157:11434"
JUDGE_MODEL = "gpt-oss:20b"

FORMS_DIR = Path("forms")
GENERATED_FORMS_DIR = Path("generated_forms")
EVAL_REPORTS_DIR = Path("eval_reports")
PER_SAMPLE_DIR = EVAL_REPORTS_DIR / "per_sample"

# 若只想測部分 sample，可填 ["001", "002"]
TARGET_SAMPLES: list[str] | None = None

# 若只想測部分模型資料夾，可填 ["qwen2.5_7b", "gpt_oss_20b"]
TARGET_MODEL_DIRS: list[str] | None = None

PRINT_PROGRESS = True
PRINT_JUDGE_RAW = False


# =========================
# Prompt
# =========================

EVAL_PROMPT_TEMPLATE = """
你是一位保險公司內部審核人員，請比較「模型生成會辦單」與「原始會辦單」的差異。

【評估目標】
請根據以下標準進行評估，並提供精確、簡短的評論。

【評估項目】
1. 個資擷取正確性（姓名、電話、身分證、生日、地址、保單號碼）
2. 主問題理解（是否抓到客戶真正需求）
3. 條列式會辦單品質（是否清楚、精簡、可用）
4. 幻覺（是否出現原始內容未提及的資訊）

【評分規則】
- score 範圍只能是 0 到 1
- 可使用小數，例如 0.0, 0.5, 0.8, 1.0
- hallucination 的 score 定義如下：
  - 1.0 = 幾乎沒有幻覺
  - 0.0 = 幻覺很多

【輸出格式（JSON）】
{
  "personal_info_accuracy": {
    "score": 0.0,
    "comment": ""
  },
  "main_issue_accuracy": {
    "score": 0.0,
    "comment": ""
  },
  "bullet_quality": {
    "score": 0.0,
    "comment": ""
  },
  "hallucination": {
    "score": 0.0,
    "comment": ""
  },
  "overall_comment": ""
}

【限制】
- 每個 comment 不可超過 20 字
- overall_comment 不可超過 30 字
- 不可解釋評估過程
- 不可輸出 JSON 以外內容
- 若資訊不足，也要根據可見內容給分

【原始會辦單】
{reference_form}

【模型生成會辦單】
{generated_form}
""".strip()


# =========================
# 資料結構
# =========================

@dataclass
class EvalResult:
    sample_id: str
    model_dir: str
    judge_model: str
    elapsed_seconds: float
    parse_success: bool
    raw_output: str
    parsed_output: dict[str, Any] | None
    error: str | None


# =========================
# 工具函式
# =========================

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_eval_prompt(reference_form: str, generated_form: str) -> str:
    return (
        EVAL_PROMPT_TEMPLATE
        .replace("{reference_form}", reference_form)
        .replace("{generated_form}", generated_form)
    )


def extract_json_block(text: str) -> dict[str, Any] | None:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def clamp_score(value: Any) -> float | None:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None

    if num < 0:
        return 0.0
    if num > 1:
        return 1.0
    return round(num, 4)


def normalize_eval_result(parsed: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None

    def normalize_item(key: str) -> dict[str, Any]:
        item = parsed.get(key, {})
        if not isinstance(item, dict):
            item = {}

        score = clamp_score(item.get("score"))
        comment = item.get("comment")
        if comment is None:
            comment = ""
        comment = str(comment).strip()

        return {
            "score": score,
            "comment": comment,
        }

    normalized = {
        "personal_info_accuracy": normalize_item("personal_info_accuracy"),
        "main_issue_accuracy": normalize_item("main_issue_accuracy"),
        "bullet_quality": normalize_item("bullet_quality"),
        "hallucination": normalize_item("hallucination"),
        "overall_comment": str(parsed.get("overall_comment", "")).strip(),
    }
    return normalized


def call_judge_model(client: Client, prompt: str) -> tuple[str, float]:
    start = time.perf_counter()

    response = client.chat(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    elapsed = time.perf_counter() - start
    raw_text = response["message"]["content"]

    return raw_text, elapsed


def get_model_dirs(base_dir: Path) -> list[Path]:
    dirs = [p for p in base_dir.iterdir() if p.is_dir()]
    dirs = sorted(dirs, key=lambda p: p.name)

    if TARGET_MODEL_DIRS is not None:
        dirs = [p for p in dirs if p.name in TARGET_MODEL_DIRS]

    return dirs


def get_sample_ids(forms_dir: Path) -> list[str]:
    ids = sorted(p.stem for p in forms_dir.glob("*.txt"))

    if TARGET_SAMPLES is not None:
        ids = [sid for sid in ids if sid in TARGET_SAMPLES]

    return ids


def save_per_sample_result(result: EvalResult) -> None:
    out_dir = PER_SAMPLE_DIR / result.model_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{result.sample_id}.json"
    payload = {
        "meta": {
            "sample_id": result.sample_id,
            "model_dir": result.model_dir,
            "judge_model": result.judge_model,
            "elapsed_seconds": result.elapsed_seconds,
            "parse_success": result.parse_success,
            "error": result.error,
        },
        "parsed_output": result.parsed_output,
        "raw_output": result.raw_output,
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def compute_summary(all_results: list[EvalResult]) -> dict[str, Any]:
    grouped: dict[str, list[EvalResult]] = {}
    for r in all_results:
        grouped.setdefault(r.model_dir, []).append(r)

    summary: dict[str, Any] = {}

    for model_dir, results in grouped.items():
        total = len(results)
        parse_success_count = sum(1 for r in results if r.parse_success)
        avg_elapsed = sum(r.elapsed_seconds for r in results) / total if total else 0.0

        def collect_avg(metric_key: str) -> float | None:
            scores: list[float] = []
            for r in results:
                if not r.parsed_output:
                    continue
                item = r.parsed_output.get(metric_key, {})
                score = item.get("score")
                if isinstance(score, (float, int)):
                    scores.append(float(score))
            if not scores:
                return None
            return round(sum(scores) / len(scores), 4)

        summary[model_dir] = {
            "total_samples": total,
            "parse_success_count": parse_success_count,
            "parse_success_rate": round(parse_success_count / total, 4) if total else 0.0,
            "avg_elapsed_seconds": round(avg_elapsed, 4),
            "avg_personal_info_accuracy": collect_avg("personal_info_accuracy"),
            "avg_main_issue_accuracy": collect_avg("main_issue_accuracy"),
            "avg_bullet_quality": collect_avg("bullet_quality"),
            "avg_hallucination": collect_avg("hallucination"),
        }

    return summary


def write_summary_json(summary: dict[str, Any]) -> None:
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_REPORTS_DIR / "summary.json"
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_summary_csv(summary: dict[str, Any]) -> None:
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_REPORTS_DIR / "summary.csv"

    fieldnames = [
        "model_dir",
        "total_samples",
        "parse_success_count",
        "parse_success_rate",
        "avg_elapsed_seconds",
        "avg_personal_info_accuracy",
        "avg_main_issue_accuracy",
        "avg_bullet_quality",
        "avg_hallucination",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for model_dir, item in summary.items():
            row = {"model_dir": model_dir}
            row.update(item)
            writer.writerow(row)


def validate_paths() -> None:
    if not FORMS_DIR.exists():
        raise FileNotFoundError(f"找不到資料夾：{FORMS_DIR}")
    if not GENERATED_FORMS_DIR.exists():
        raise FileNotFoundError(f"找不到資料夾：{GENERATED_FORMS_DIR}")


# =========================
# 主程式
# =========================

def main() -> None:
    validate_paths()

    client = Client(host=OLLAMA_HOST)
    EVAL_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PER_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    sample_ids = get_sample_ids(FORMS_DIR)
    model_dirs = get_model_dirs(GENERATED_FORMS_DIR)

    if not sample_ids:
        print("[ERROR] forms/ 中沒有找到 sample。")
        return

    if not model_dirs:
        print("[ERROR] generated_forms/ 中沒有找到模型資料夾。")
        return

    if PRINT_PROGRESS:
        print(f"[INFO] judge model: {JUDGE_MODEL}")
        print(f"[INFO] samples: {sample_ids}")
        print(f"[INFO] model_dirs: {[p.name for p in model_dirs]}")

    all_results: list[EvalResult] = []

    for model_dir in model_dirs:
        for sample_id in sample_ids:
            ref_path = FORMS_DIR / f"{sample_id}.txt"
            gen_path = model_dir / f"{sample_id}.txt"

            if not gen_path.exists():
                if PRINT_PROGRESS:
                    print(f"[WARN] 缺少生成檔案：{gen_path}")
                continue

            reference_form = load_text(ref_path)
            generated_form = load_text(gen_path)
            prompt = build_eval_prompt(reference_form, generated_form)

            if PRINT_PROGRESS:
                print(f"[INFO] evaluating sample={sample_id}, model_dir={model_dir.name}")

            try:
                raw_output, elapsed = call_judge_model(client, prompt)

                if PRINT_JUDGE_RAW:
                    print("[JUDGE RAW]")
                    print(raw_output)
                    print("=" * 60)

                parsed = extract_json_block(raw_output)
                normalized = normalize_eval_result(parsed)
                parse_success = normalized is not None

                result = EvalResult(
                    sample_id=sample_id,
                    model_dir=model_dir.name,
                    judge_model=JUDGE_MODEL,
                    elapsed_seconds=elapsed,
                    parse_success=parse_success,
                    raw_output=raw_output,
                    parsed_output=normalized,
                    error=None,
                )
            except Exception as exc:
                result = EvalResult(
                    sample_id=sample_id,
                    model_dir=model_dir.name,
                    judge_model=JUDGE_MODEL,
                    elapsed_seconds=0.0,
                    parse_success=False,
                    raw_output="",
                    parsed_output=None,
                    error=f"{type(exc).__name__}: {exc}",
                )

            save_per_sample_result(result)
            all_results.append(result)

    summary = compute_summary(all_results)
    write_summary_json(summary)
    write_summary_csv(summary)

    print("\n===== 評估完成 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
