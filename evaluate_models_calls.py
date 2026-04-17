from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ollama import Client

# =========================
# 基本設定
# =========================

OLLAMA_HOST = "http://10.67.75.157:11434"
JUDGE_MODEL = "gpt-oss:20b"   # 固定裁判模型
INPUT_DIR = Path("outputs")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

client = Client(host=OLLAMA_HOST)


# =========================
# 工具函式
# =========================

def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def extract_json_object(text: str) -> dict[str, Any]:
    """
    嘗試從模型輸出中抽出第一個 JSON 物件
    """
    text = text.strip()

    # 先處理 markdown code fence
    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 直接 parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 抓第一個 {...}
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("無法從模型輸出中找到 JSON 物件")

    return json.loads(match.group(0))


def normalize_line(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip())


# =========================
# Rule-based 評估
# =========================

@dataclass
class RuleEvalResult:
    line_count: int
    valid_line_prefix_ratio: float
    strictly_alternating: bool
    contains_extra_meta: bool
    mentions_transfer_specialist: bool
    mentions_identity_verification: bool
    mentions_claim_progress: bool
    issues: list[str]


def rule_evaluate_transcript(text: str) -> RuleEvalResult:
    lines = [normalize_line(x) for x in text.splitlines() if x.strip()]
    issues: list[str] = []

    if not lines:
        return RuleEvalResult(
            line_count=0,
            valid_line_prefix_ratio=0.0,
            strictly_alternating=False,
            contains_extra_meta=True,
            mentions_transfer_specialist=False,
            mentions_identity_verification=False,
            mentions_claim_progress=False,
            issues=["輸出為空"],
        )

    valid_prefix_count = 0
    speaker_seq: list[str] = []
    contains_extra_meta = False

    for line in lines:
        if line.startswith("L:"):
            valid_prefix_count += 1
            speaker_seq.append("L")
        elif line.startswith("R:"):
            valid_prefix_count += 1
            speaker_seq.append("R")
        else:
            contains_extra_meta = True
            issues.append(f"非 L:/R: 開頭行：{line[:40]}")

    valid_ratio = valid_prefix_count / len(lines)

    strictly_alternating = True
    if speaker_seq:
        for i in range(1, len(speaker_seq)):
            if speaker_seq[i] == speaker_seq[i - 1]:
                strictly_alternating = False
                issues.append(f"發話人未交替：第 {i} 與第 {i+1} 行皆為 {speaker_seq[i]}")
                break

    if not (10 <= len(lines) <= 14):
        issues.append(f"輪數不符要求：目前 {len(lines)} 行，預期 10~14 行")

    # 轉專員偵測
    transfer_keywords = [
        "轉專員", "轉給專員", "轉由專員", "協助轉接", "幫您轉接", "轉接專員"
    ]
    mentions_transfer_specialist = any(k in text for k in transfer_keywords)
    if mentions_transfer_specialist:
        issues.append("題目要求不轉專員，但內容疑似提到轉專員")

    # 核對資料偵測
    verify_keywords = [
        "核對", "確認一下您的資料", "身分證", "末四碼", "保單號碼", "生日", "請問是"
    ]
    mentions_identity_verification = any(k in text for k in verify_keywords)
    if not mentions_identity_verification:
        issues.append("題目要求需核對資料，但內容中核對跡象不足")

    # 理賠進度查詢偵測
    claim_keywords = [
        "理賠", "案件進度", "申請進度", "審核中", "受理", "賠案"
    ]
    mentions_claim_progress = any(k in text for k in claim_keywords)
    if not mentions_claim_progress:
        issues.append("主題不像理賠進度查詢")

    return RuleEvalResult(
        line_count=len(lines),
        valid_line_prefix_ratio=round(valid_ratio, 4),
        strictly_alternating=strictly_alternating,
        contains_extra_meta=contains_extra_meta,
        mentions_transfer_specialist=mentions_transfer_specialist,
        mentions_identity_verification=mentions_identity_verification,
        mentions_claim_progress=mentions_claim_progress,
        issues=issues,
    )


# =========================
# LLM Judge Prompt
# =========================

JUDGE_SYSTEM_PROMPT = """
你是一位嚴格、保守、重視可驗證依據的保險客服通話資料評審員。
你的任務是評估一段「合成的繁體中文保險客服 STT 逐字稿」是否符合指定條件。

評分原則：
1. 不要因為語句漂亮就給高分，請重視「是否符合需求」。
2. 若逐字稿自行加入題目未要求、且沒有明確根據的細節，視為 hallucination / 不必要捏造。
3. 若違反格式規範（L:/R:、輪數、交替、額外說明）要明確扣分。
4. 請輸出 JSON，不要有任何額外說明。

分數區間：
- 1 = 很差 / 嚴重不符
- 2 = 偏弱 / 有明顯問題
- 3 = 普通 / 可接受但不穩
- 4 = 好 / 大致合理
- 5 = 很好 / 穩定且貼近真實

hallucination_risk 分數意義：
- 1 = 幾乎沒有不必要捏造
- 5 = 捏造很多，風險高
"""

def build_judge_user_prompt(
    system_prompt: str,
    user_prompt: str,
    transcript: str,
    rule_eval: RuleEvalResult,
) -> str:
    schema_hint = """
請輸出以下 JSON 結構：
{
  "scores": {
    "format_compliance": 1,
    "task_alignment": 1,
    "naturalness": 1,
    "insurance_realism": 1,
    "hallucination_risk": 1
  },
  "hallucinations": [
    {
      "type": "規格幻覺/情境幻覺/業務邏輯幻覺/細節幻覺",
      "evidence": "具體證據"
    }
  ],
  "strengths": ["..."],
  "weaknesses": ["..."],
  "summary": "..."
}
"""

    return f"""
以下是生成任務資訊：

[System Prompt]
{system_prompt}

[User Prompt]
{user_prompt}

[逐字稿]
{transcript}

[Rule-based 檢查結果]
{json.dumps(asdict(rule_eval), ensure_ascii=False, indent=2)}

請你依照上面資訊評估這段逐字稿。
注意：
- 若 rule-based 已經發現硬性違規，format_compliance 與 task_alignment 不應給高分
- 請指出「有根據」的 hallucination，不要空泛評論
- 優點缺點要具體
- 只輸出 JSON

{schema_hint}
""".strip()


def judge_one_sample(
    judge_model: str,
    system_prompt: str,
    user_prompt: str,
    transcript: str,
    rule_eval: RuleEvalResult,
) -> dict[str, Any]:
    judge_prompt = build_judge_user_prompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        transcript=transcript,
        rule_eval=rule_eval,
    )

    response = client.chat(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": judge_prompt},
        ],
        stream=False,
        options={"temperature": 0},
        keep_alive="5m",
    )

    content = response.message.content
    parsed = extract_json_object(content)

    # 把 judge 自己的效能資訊也存起來
    parsed["_judge_meta"] = {
        "judge_model": judge_model,
        "total_duration": getattr(response, "total_duration", None),
        "eval_count": getattr(response, "eval_count", None),
        "prompt_eval_count": getattr(response, "prompt_eval_count", None),
    }
    return parsed


# =========================
# 主流程
# =========================

def collect_samples(input_dir: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for path in sorted(input_dir.rglob("*.json")):
        data = read_json(path)
        data["_path"] = str(path)
        samples.append(data)
    return samples


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}

    for item in results:
        model = item["model"]
        by_model.setdefault(model, []).append(item)

    summary: dict[str, Any] = {}

    score_keys = [
        "format_compliance",
        "task_alignment",
        "naturalness",
        "insurance_realism",
        "hallucination_risk",
    ]

    for model, items in by_model.items():
        score_sum = {k: 0.0 for k in score_keys}
        all_strengths: list[str] = []
        all_weaknesses: list[str] = []
        all_rule_issues: list[str] = []

        for item in items:
            scores = item["judge_result"]["scores"]
            for k in score_keys:
                score_sum[k] += float(scores[k])

            all_strengths.extend(item["judge_result"].get("strengths", []))
            all_weaknesses.extend(item["judge_result"].get("weaknesses", []))
            all_rule_issues.extend(item["rule_eval"].get("issues", []))

        n = len(items)
        avg_scores = {k: round(score_sum[k] / n, 3) for k in score_keys}

        summary[model] = {
            "num_samples": n,
            "avg_scores": avg_scores,
            "common_rule_issues": all_rule_issues,
            "strength_examples": all_strengths[:10],
            "weakness_examples": all_weaknesses[:10],
        }

    return summary


def main() -> None:
    samples = collect_samples(INPUT_DIR)
    if not samples:
        raise SystemExit("找不到任何輸入 JSON，請先把生成結果放進 outputs/")

    all_results: list[dict[str, Any]] = []

    for sample in samples:
        model = sample["model"]
        transcript = sample["output"]
        system_prompt = sample["system_prompt"]
        user_prompt = sample["user_prompt"]

        rule_eval = rule_evaluate_transcript(transcript)

        judge_result = judge_one_sample(
            judge_model=JUDGE_MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            transcript=transcript,
            rule_eval=rule_eval,
        )

        record = {
            "model": model,
            "sample_id": sample.get("sample_id"),
            "source_file": sample["_path"],
            "rule_eval": asdict(rule_eval),
            "judge_result": judge_result,
        }
        all_results.append(record)

    summary = summarize_results(all_results)

    write_json(REPORT_DIR / "llm_eval.json", all_results)
    write_json(REPORT_DIR / "summary.json", summary)

    print("評估完成")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
