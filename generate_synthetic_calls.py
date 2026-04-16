from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ollama import Client


# =========================
# 設定區
# =========================

OLLAMA_HOST = "http://10.67.75.157:11434"
GEN_MODEL = "gpt-oss:20b"

OUTPUT_BASE_DIR = Path("synthetic_data")
CALLS_DIR = OUTPUT_BASE_DIR / "synthetic_calls"
LABELS_DIR = OUTPUT_BASE_DIR / "synthetic_labels"

NUM_SAMPLES = 20
START_INDEX = 1

PRINT_PROGRESS = True
PRINT_RAW_OUTPUT = False

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


# =========================
# Prompt Template
# =========================

PROMPT_TEMPLATE = """
你是一位保險客服對話模擬助手。

請根據以下條件，生成一段合理、自然、精簡的客服通話內容。

【任務要求】
- 輸出客戶（R）與客服（L）的對話
- 對話需符合保險客服情境
- 必須包含指定資訊與流程
- 對話內容應自然，但不可冗長

【重要規則】
1. 只能根據提供條件生成，不可新增未被允許的具體事實
2. 不可自行補出完整身分證字號、詳細地址、完整保單號碼，除非條件中明確要求
3. 若條件未要求某項個資，不可硬加
4. 對話需使用繁體中文
5. 使用以下格式：
   L: 客服
   R: 客戶
6. 不要加入旁白、說明、分析或標題
7. 對話長度控制在 8 到 16 句之間

【條件】
客戶目的：
{customer_intent}

必須提及資訊：
{required_info}

可選提及資訊：
{optional_info}

服務流程：
{service_flow}

情境難度：
{difficulty}

客戶語氣：
{customer_tone}

【輸出】
直接輸出對話內容
""".strip()


# =========================
# Scenario 定義
# =========================

INTENTS = [
    "查詢保單借款還款紀錄",
    "詢問保費是否已入帳",
    "查詢保單狀態是否有效",
    "詢問理賠進度",
    "詢問理賠金額",
    "申請補發繳費證明",
    "查詢生存金或滿期金給付狀況",
    "詢問信用卡扣款失敗原因",
    "詢問銀行轉帳扣款問題",
    "詢問保單變更申請進度",
]

REQUIRED_INFO_CANDIDATES = [
    ["姓名", "電話"],
    ["姓名", "電話", "生日"],
    ["姓名", "電話", "身分證後四碼"],
    ["姓名", "電話", "保單號碼末四碼"],
    ["姓名"],
]

OPTIONAL_INFO_CANDIDATES = [
    [],
    ["生日"],
    ["身分證後四碼"],
    ["保單號碼末四碼"],
    ["聯絡時段"],
]

SERVICE_FLOW_LIBRARY = {
    "查詢保單借款還款紀錄": ["確認身份", "查詢還款紀錄", "回覆查詢結果"],
    "詢問保費是否已入帳": ["確認身份", "查詢入帳狀態", "回覆查詢結果"],
    "查詢保單狀態是否有效": ["確認身份", "查詢保單狀態", "回覆查詢結果"],
    "詢問理賠進度": ["確認身份", "查詢理賠進度", "回覆目前進度"],
    "詢問理賠金額": ["確認身份", "查詢理賠資料", "回覆金額或處理狀況"],
    "申請補發繳費證明": ["確認身份", "確認補發需求", "說明後續處理方式"],
    "查詢生存金或滿期金給付狀況": ["確認身份", "查詢給付狀況", "回覆結果"],
    "詢問信用卡扣款失敗原因": ["確認身份", "查詢扣款紀錄", "說明可能原因與處理方式"],
    "詢問銀行轉帳扣款問題": ["確認身份", "查詢扣款紀錄", "說明處理方式"],
    "詢問保單變更申請進度": ["確認身份", "查詢申請進度", "回覆結果"],
}

DIFFICULTIES = ["簡單", "中等", "偏複雜"]
CUSTOMER_TONES = ["平靜", "急切", "有點困惑", "簡短直接"]


@dataclass
class Scenario:
    scenario_id: str
    customer_intent: str
    required_info: list[str]
    optional_info: list[str]
    service_flow: list[str]
    difficulty: str
    customer_tone: str


# =========================
# 工具函式
# =========================

def build_prompt(s: Scenario) -> str:
    return (
        PROMPT_TEMPLATE
        .replace("{customer_intent}", s.customer_intent)
        .replace("{required_info}", "、".join(s.required_info) if s.required_info else "無")
        .replace("{optional_info}", "、".join(s.optional_info) if s.optional_info else "無")
        .replace("{service_flow}", " → ".join(s.service_flow))
        .replace("{difficulty}", s.difficulty)
        .replace("{customer_tone}", s.customer_tone)
    )


def generate_scenarios(num_samples: int, start_index: int = 1) -> list[Scenario]:
    scenarios: list[Scenario] = []

    for i in range(start_index, start_index + num_samples):
        intent = random.choice(INTENTS)
        required_info = random.choice(REQUIRED_INFO_CANDIDATES)
        optional_info = random.choice(OPTIONAL_INFO_CANDIDATES)
        service_flow = SERVICE_FLOW_LIBRARY[intent]
        difficulty = random.choice(DIFFICULTIES)
        customer_tone = random.choice(CUSTOMER_TONES)

        scenario = Scenario(
            scenario_id=f"{i:03d}",
            customer_intent=intent,
            required_info=required_info,
            optional_info=optional_info,
            service_flow=service_flow,
            difficulty=difficulty,
            customer_tone=customer_tone,
        )
        scenarios.append(scenario)

    return scenarios


def validate_dialogue(text: str) -> tuple[bool, str]:
    text = text.strip()
    if not text:
        return False, "empty output"

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 4:
        return False, "too few lines"

    bad_prefix = [line for line in lines if not (line.startswith("L:") or line.startswith("R:"))]
    if bad_prefix:
        return False, "invalid speaker prefix"

    return True, ""


def call_model(client: Client, prompt: str) -> tuple[str, float]:
    start = time.perf_counter()

    response = client.chat(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    )

    elapsed = time.perf_counter() - start
    content = response["message"]["content"]
    return content, elapsed


def save_outputs(
    scenario: Scenario,
    dialogue_text: str,
    elapsed_seconds: float,
    error: str | None,
) -> None:
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)

    call_path = CALLS_DIR / f"{scenario.scenario_id}.txt"
    label_path = LABELS_DIR / f"{scenario.scenario_id}.json"

    call_path.write_text(dialogue_text, encoding="utf-8")

    payload = {
        "meta": {
            "scenario_id": scenario.scenario_id,
            "generation_model": GEN_MODEL,
            "elapsed_seconds": round(elapsed_seconds, 4),
            "error": error,
        },
        "scenario": asdict(scenario),
        "target_summary": {
            "customer_intent": scenario.customer_intent,
            "required_info": scenario.required_info,
            "optional_info": scenario.optional_info,
            "service_flow": scenario.service_flow,
        },
        "dialogue_file": str(call_path),
    }

    label_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_generation_summary(results: list[dict[str, Any]]) -> None:
    summary_path = OUTPUT_BASE_DIR / "generation_summary.json"
    summary_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================
# 主程式
# =========================

def main() -> None:
    client = Client(host=OLLAMA_HOST)
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = generate_scenarios(num_samples=NUM_SAMPLES, start_index=START_INDEX)
    generation_results: list[dict[str, Any]] = []

    if PRINT_PROGRESS:
        print(f"[INFO] model={GEN_MODEL}")
        print(f"[INFO] num_samples={len(scenarios)}")

    for scenario in scenarios:
        prompt = build_prompt(scenario)

        if PRINT_PROGRESS:
            print(f"[INFO] generating scenario={scenario.scenario_id} | intent={scenario.customer_intent}")

        try:
            output, elapsed = call_model(client, prompt)

            if PRINT_RAW_OUTPUT:
                print("[RAW OUTPUT]")
                print(output)
                print("=" * 60)

            valid, reason = validate_dialogue(output)
            error = None if valid else reason

            save_outputs(
                scenario=scenario,
                dialogue_text=output,
                elapsed_seconds=elapsed,
                error=error,
            )

            generation_results.append({
                "scenario_id": scenario.scenario_id,
                "intent": scenario.customer_intent,
                "elapsed_seconds": round(elapsed, 4),
                "valid_dialogue": valid,
                "error": error,
            })

        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

            save_outputs(
                scenario=scenario,
                dialogue_text="",
                elapsed_seconds=0.0,
                error=error,
            )

            generation_results.append({
                "scenario_id": scenario.scenario_id,
                "intent": scenario.customer_intent,
                "elapsed_seconds": 0.0,
                "valid_dialogue": False,
                "error": error,
            })

    save_generation_summary(generation_results)

    ok_count = sum(1 for r in generation_results if r["valid_dialogue"])
    print("\n===== 生成完成 =====")
    print(f"成功筆數：{ok_count}/{len(generation_results)}")
    print(f"輸出資料夾：{OUTPUT_BASE_DIR.resolve()}")


if __name__ == "__main__":
    main()
