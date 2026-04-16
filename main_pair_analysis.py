import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from ollama import Client


# =========================
# 1. 基本設定
# =========================
OLLAMA_HOST = "http://10.67.75.157:11434"
MODEL_NAME = "gpt-oss:120b"

CALL_DIR = Path("calls")
FORM_DIR = Path("forms")
OUTPUT_DIR = Path("outputs")
PAIR_OUTPUT_DIR = OUTPUT_DIR / "pair_analysis"

OUTPUT_DIR.mkdir(exist_ok=True)
PAIR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

client = Client(host=OLLAMA_HOST)


# =========================
# 2. JSON Schema
#    Ollama 支援 format 傳入 schema
# =========================
PAIR_ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "file_id": {"type": "string"},
        "call_summary": {"type": "string"},
        "form_goal": {"type": "string"},
        "key_fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                    "field_value": {"type": "string"},
                    "from_call_evidence": {"type": "string"},
                    "transformation_rule": {"type": "string"}
                },
                "required": [
                    "field_name",
                    "field_value",
                    "from_call_evidence",
                    "transformation_rule"
                ]
            }
        },
        "writing_steps": {
            "type": "array",
            "items": {"type": "string"}
        },
        "writing_rules": {
            "type": "array",
            "items": {"type": "string"}
        },
        "template_like_phrases": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": [
        "file_id",
        "call_summary",
        "form_goal",
        "key_fields",
        "writing_steps",
        "writing_rules",
        "template_like_phrases"
    ]
}


# =========================
# 3. 工具函式
# =========================
def read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def clean_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def find_matching_form(call_file: Path) -> Optional[Path]:
    """
    用 stem 配對 forms 資料夾中的對應檔案。
    例如 calls/abc123.txt -> forms/abc123.txt
    """
    stem = call_file.stem
    candidates = list(FORM_DIR.glob(f"{stem}.*"))
    if not candidates:
        return None
    return candidates[0]


def pair_files(call_dir: Path) -> List[tuple[str, Path, Path]]:
    """
    回傳 [(file_id, call_path, form_path), ...]
    """
    pairs = []
    for call_file in sorted(call_dir.glob("*.txt")):
        form_file = find_matching_form(call_file)
        if form_file is not None:
            pairs.append((call_file.stem, call_file, form_file))
    return pairs


def build_prompt(file_id: str, call_text: str, form_text: str) -> str:
    schema_text = json.dumps(PAIR_ANALYSIS_SCHEMA, ensure_ascii=False, indent=2)

    return f"""
你現在是一位資深客服流程分析師。

你的任務不是重寫會辦單，
而是分析「客服人員如何從通話紀錄產生這份會辦單」。

請你根據下面的 JSON schema 輸出結果。
只能輸出 JSON，不要輸出任何額外說明。

JSON Schema:
{schema_text}

分析要求：
1. 摘要通話重點。
2. 說明這份會辦單的目的。
3. 列出會辦單中的重要欄位與內容。
4. 說明每個欄位是從通話中的哪段資訊整理而來。
5. 說明口語內容如何被改寫成正式書面語。
6. 歸納客服撰寫會辦單的步驟。
7. 找出疑似模板化、慣用的句型。

【檔案ID】
{file_id}

【通話紀錄】
{call_text}

【對應會辦單】
{form_text}
""".strip()


def ask_ollama_for_pair_analysis(prompt: str) -> dict:
    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位資深客服流程分析師。"
                    "你必須嚴格輸出符合 schema 的 JSON。"
                    "不得輸出額外說明文字。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        format=PAIR_ANALYSIS_SCHEMA,
        options={
            "temperature": 0.1,
        },
    )

    content = response["message"]["content"]
    return json.loads(content)


def process_one_pair(file_id: str, call_path: Path, form_path: Path) -> dict:
    call_text = clean_text(read_text_file(call_path))
    form_text = clean_text(read_text_file(form_path))

    prompt = build_prompt(file_id, call_text, form_text)
    result = ask_ollama_for_pair_analysis(prompt)
    return result


# =========================
# 4. 主流程
# =========================
def main():
    if not CALL_DIR.exists():
        print(f"找不到 calls 資料夾: {CALL_DIR.resolve()}")
        return

    if not FORM_DIR.exists():
        print(f"找不到 forms 資料夾: {FORM_DIR.resolve()}")
        return

    pairs = pair_files(CALL_DIR)

    if not pairs:
        print("沒有找到任何可配對的 call/form 檔案。")
        return

    print(f"成功配對 {len(pairs)} 組資料。")

    all_results = []
    failed_files = []

    total_start = time.time()

    for idx, (file_id, call_path, form_path) in enumerate(pairs, start=1):
        print(f"[{idx}/{len(pairs)}] 處理中: {file_id}")

        try:
            start = time.time()
            result = process_one_pair(file_id, call_path, form_path)
            elapsed = time.time() - start

            # 額外附上處理時間，方便你後續觀察
            result["_meta"] = {
                "call_file": str(call_path),
                "form_file": str(form_path),
                "elapsed_seconds": round(elapsed, 2),
            }

            # 單筆輸出
            single_output_path = PAIR_OUTPUT_DIR / f"{file_id}.json"
            with open(single_output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            all_results.append(result)
            print(f"    完成 -> {single_output_path}")

        except Exception as e:
            print(f"    失敗 -> {file_id}: {e}")
            failed_files.append({
                "file_id": file_id,
                "error": str(e),
            })

    # 全部集合輸出
    all_output_path = OUTPUT_DIR / "all_pair_analysis.json"
    with open(all_output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    if failed_files:
        failed_output_path = OUTPUT_DIR / "failed_files.json"
        with open(failed_output_path, "w", encoding="utf-8") as f:
            json.dump(failed_files, f, ensure_ascii=False, indent=2)
        print(f"失敗清單已寫入 -> {failed_output_path}")

    total_elapsed = time.time() - total_start

    print("\n=== 全部完成 ===")
    print(f"成功筆數: {len(all_results)}")
    print(f"總耗時(秒): {round(total_elapsed, 2)}")
    print(f"總集合輸出: {all_output_path}")


if __name__ == "__main__":
    main()
