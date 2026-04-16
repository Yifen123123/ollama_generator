import json
from pathlib import Path
from typing import Dict, Any
from ollama import Client

OLLAMA_HOST = "http://10.67.75.157:11434"
MODEL_NAME = "gpt-oss:20b"

client = Client(host=OLLAMA_HOST)

INPUT_PATH = Path("outputs/all_pair_analysis.json")
OUTPUT_PATH = Path("outputs/final_rules.json")


RULE_SUMMARY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_goal": {"type": "string"},
        "common_fields": {
            "type": "array",
            "items": {"type": "string"}
        },
        "workflow": {
            "type": "array",
            "items": {"type": "string"}
        },
        "field_mapping_rules": {
            "type": "array",
            "items": {"type": "string"}
        },
        "writing_rules": {
            "type": "array",
            "items": {"type": "string"}
        },
        "template_patterns": {
            "type": "array",
            "items": {"type": "string"}
        },
        "sop": {
            "type": "array",
            "items": {"type": "string"}
        },
        "llm_instruction": {"type": "string"}
    },
    "required": [
        "overall_goal",
        "common_fields",
        "workflow",
        "field_mapping_rules",
        "writing_rules",
        "template_patterns",
        "sop",
        "llm_instruction"
    ]
}


def build_summary_prompt(analyses: list[dict]) -> str:
    schema_text = json.dumps(RULE_SUMMARY_SCHEMA, ensure_ascii=False, indent=2)
    analyses_text = json.dumps(analyses, ensure_ascii=False, indent=2)

    return f"""
你現在是一位客服知識整理專家。

以下是一批「通話紀錄 -> 會辦單」的逐筆分析結果。
請你從中整理出客服人員生成會辦單的共同規則。

請依照下面 schema 輸出 JSON。
不要輸出額外文字。

JSON Schema:
{schema_text}

請總結：
1. 客服生成會辦單的整體目標
2. 常見欄位
3. 生成流程
4. 欄位映射規則
5. 書寫規則
6. 常見模板句型
7. SOP
8. 最後整理成一段可給 LLM 使用的 instruction

資料如下：
{analyses_text}
""".strip()


def main():
    if not INPUT_PATH.exists():
        print(f"找不到輸入檔案: {INPUT_PATH.resolve()}")
        return

    analyses = json.loads(INPUT_PATH.read_text(encoding="utf-8"))

    prompt = build_summary_prompt(analyses)

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位客服知識整理專家。"
                    "你必須輸出符合 schema 的 JSON。"
                    "不得輸出額外說明。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        format=RULE_SUMMARY_SCHEMA,
        options={
            "temperature": 0.1,
        },
    )

    result = json.loads(response["message"]["content"])

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"完成 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
