import json
import time
from pathlib import Path
from typing import Dict, Any
from ollama import Client


# =========================
# 1. 基本設定
# =========================
OLLAMA_HOST = "http://10.67.75.157:11434"
MODEL_NAME = "gpt-oss:20b"

CALL_DIR = Path("calls")
RULE_PATH = Path("outputs/final_rules.json")
OUTPUT_DIR = Path("generated_forms")

OUTPUT_DIR.mkdir(exist_ok=True)

client = Client(host=OLLAMA_HOST)


# =========================
# 2. 生成會辦單的輸出格式
#    你之後可以依你的實際欄位再調整
# =========================
FORM_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "案件類型": {"type": "string"},
        "客戶需求": {"type": "string"},
        "處理經過": {"type": "string"},
        "處理結果": {"type": "string"},
        "是否需轉辦": {"type": "string"},
        "轉辦部門": {"type": "string"},
        "備註": {"type": "string"}
    },
    "required": [
        "案件類型",
        "客戶需求",
        "處理經過",
        "處理結果",
        "是否需轉辦",
        "轉辦部門",
        "備註"
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


def load_rules() -> dict:
    if not RULE_PATH.exists():
        raise FileNotFoundError(f"找不到規則檔案: {RULE_PATH.resolve()}")
    return json.loads(RULE_PATH.read_text(encoding="utf-8"))


def build_generation_prompt(call_text: str, rules: dict) -> str:
    return f"""
你現在是一位資深客服人員，負責根據通話紀錄撰寫會辦單。

以下是根據歷史資料整理出的會辦單生成規則，請務必遵守。

【生成規則】
{json.dumps(rules, ensure_ascii=False, indent=2)}

請依照以下要求生成會辦單：
1. 使用正式、精簡、書面的客服紀錄語氣。
2. 優先依照規則中的 workflow、field_mapping_rules、writing_rules、sop。
3. 若通話中沒有明確提到某項資訊，請填「未提及」。
4. 不可憑空捏造具體事實。
5. 若是否需轉辦無法判定，請填「未提及」。
6. 若轉辦部門無法判定，請填「未提及」。
7. 僅輸出符合指定 schema 的 JSON。

【通話紀錄】
{call_text}
""".strip()


def generate_form(call_text: str, rules: dict) -> dict:
    prompt = build_generation_prompt(call_text, rules)

    response = client.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一位專業客服人員，擅長將通話紀錄整理成正式會辦單。"
                    "你必須嚴格輸出符合 schema 的 JSON，不得輸出任何額外說明。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        stream=False,
        format=FORM_SCHEMA,
        options={
            "temperature": 0.2,
        },
    )

    content = response["message"]["content"]
    return json.loads(content)


def save_result(file_stem: str, result: dict, elapsed_seconds: float) -> Path:
    output_path = OUTPUT_DIR / f"{file_stem}.json"

    payload = {
        "file_id": file_stem,
        "generated_form": result,
        "_meta": {
            "elapsed_seconds": round(elapsed_seconds, 2)
        }
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


# =========================
# 4. 主流程：批次讀 calls/*.txt
# =========================
def main():
    if not CALL_DIR.exists():
        print(f"找不到 calls 資料夾: {CALL_DIR.resolve()}")
        return

    call_files = sorted(CALL_DIR.glob("*.txt"))
    if not call_files:
        print("calls 資料夾中沒有 .txt 檔案")
        return

    rules = load_rules()

    print(f"共找到 {len(call_files)} 份通話紀錄")
    print(f"使用模型: {MODEL_NAME}")
    print("-" * 50)

    failed_files = []

    total_start = time.time()

    for idx, call_file in enumerate(call_files, start=1):
        print(f"[{idx}/{len(call_files)}] 處理中: {call_file.name}")

        try:
            call_text = clean_text(read_text_file(call_file))

            start = time.time()
            result = generate_form(call_text, rules)
            elapsed = time.time() - start

            output_path = save_result(call_file.stem, result, elapsed)
            print(f"    完成 -> {output_path}")

        except Exception as e:
            print(f"    失敗 -> {call_file.name}: {e}")
            failed_files.append({
                "file_name": call_file.name,
                "error": str(e),
            })

    total_elapsed = time.time() - total_start

    if failed_files:
        failed_path = OUTPUT_DIR / "failed_files.json"
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_files, f, ensure_ascii=False, indent=2)
        print(f"\n失敗清單已寫入 -> {failed_path}")

    print("\n=== 批次生成完成 ===")
    print(f"總耗時(秒): {round(total_elapsed, 2)}")
    print(f"輸出資料夾: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
