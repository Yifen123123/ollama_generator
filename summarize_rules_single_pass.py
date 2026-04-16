import json
from pathlib import Path
from typing import Dict, Any, List
from ollama import Client

OLLAMA_HOST = "http://10.67.75.157:11434"
MODEL_NAME = "gpt-oss:120b"

INPUT_PATH = Path("outputs/all_pair_analysis.json")
OUTPUT_PATH = Path("outputs/final_rules.json")

client = Client(host=OLLAMA_HOST)

FINAL_SCHEMA: Dict[str, Any] = {
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


def load_data() -> List[dict]:
    return json.loads(INPUT_PATH.read_text(encoding="utf-8"))


def compress_one(item: dict) -> dict:
    """
    只保留最有助於抽規則的欄位，避免把整份分析原封不動塞給模型。
    """
    return {
        "file_id": item.get("file_id", ""),
        "form_goal": item.get("form_goal", ""),
        "key_fields": [
            {
                "field_name": f.get("field_name", ""),
                "transformation_rule": f.get("transformation_rule", "")
            }
            for f in item.get("key_fields", [])
        ],
        "writing_steps": item.get("writing_steps", []),
        "writing_rules": item.get("writing_rules", []),
        "template_like_phrases": item.get("template_like_phrases", [])
    }


def compress_all(data: List[dict]) -> List[dict]:
    return [compress_one(x) for x in data]


def build_prompt(compressed_data: List[dict]) -> str:
    schema_text = json.dumps(FINAL_SCHEMA, ensure_ascii=False, indent=2)
    data_text = json.dumps(compressed_data, ensure_ascii=False, separators=(",", ":"))

    return f"""
你現在是一位客服知識整理專家。

以下是多筆「通話紀錄 -> 會辦單」分析後的壓縮資料。
請你直接根據全部資料，歸納出客服撰寫會辦單的共同規則。

請遵守以下要求：
1. 僅根據提供資料歸納，不要虛構不存在的流程。
2. 找出高頻出現的欄位、流程、映射規則、書寫規則、模板句型。
3. 最後整理成可供 LLM 直接使用的 llm_instruction。
4. 只能輸出符合 schema 的 JSON。

JSON Schema:
{schema_text}

資料：
{data_text}
""".strip()


def main():
    if not INPUT_PATH.exists():
        print(f"找不到檔案: {INPUT_PATH.resolve()}")
        return

    data = load_data()
    compressed_data = compress_all(data)

    prompt = build_prompt(compressed_data)

    print(f"總筆數: {len(data)}")
    print(f"壓縮後字元數: {len(json.dumps(compressed_data, ensure_ascii=False))}")

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位客服知識整理專家。"
                    "你必須從全部資料中歸納會辦單撰寫規則。"
                    "只能輸出符合 schema 的 JSON。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        format=FINAL_SCHEMA,
        options={
            "temperature": 0.1,
            "num_ctx": 65536,    # 官方建議大上下文任務至少 64k
            "num_predict": 1500, # 控制輸出長度
        },
        keep_alive=-1,
    )

    result = json.loads(response["message"]["content"])

    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"完成 -> {OUTPUT_PATH}")
    print("prompt_eval_count:", response.get("prompt_eval_count"))
    print("eval_count:", response.get("eval_count"))
    print("total_duration:", response.get("total_duration"))
    print("load_duration:", response.get("load_duration"))


if __name__ == "__main__":
    main()
