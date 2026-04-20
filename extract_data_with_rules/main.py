from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ollama import Client

from src.prompt_builder import (
    build_call_prompt,
    build_form_prompt,
    build_output_schema,
)


# =========================
# 可依專案調整的設定
# =========================
BASE_DIR = Path(__file__).resolve().parent

RULES_PATH = BASE_DIR / "data" / "final_rules_ch.json"
FORMS_DIR = BASE_DIR / "data" / "forms"
CALLS_DIR = BASE_DIR / "data" / "calls"
PROMPTS_DIR = BASE_DIR / "prompts"
OUTPUT_DIR = BASE_DIR / "outputs"

FORM_PROMPT_PATH = PROMPTS_DIR / "normalize_form.prompt"
CALL_PROMPT_PATH = PROMPTS_DIR / "extract_call.prompt"

OLLAMA_HOST = "http://10.67.75.157:11434"
MODEL_NAME = "gpt-oss:120b"


# =========================
# 基本 I/O
# =========================
def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def list_txt_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(folder.glob("*.txt"))


# =========================
# LLM 呼叫
# =========================
def chat_json(
    client: Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    temperature: float = 0.0,
    num_ctx: int = 32768,
    num_predict: int = 1200,
) -> dict[str, Any]:
    """
    呼叫 Ollama 並要求輸出符合 schema 的 JSON。
    """
    response = client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stream=False,
        format=schema,
        options={
            "temperature": temperature,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
        },
        keep_alive=-1,
    )

    content = response["message"]["content"]
    return json.loads(content)


# =========================
# 核心流程
# =========================
def normalize_forms(
    client: Client,
    model_name: str,
    rules: dict[str, Any],
    forms_dir: Path,
    prompt_path: Path,
    output_path: Path,
) -> None:
    common_fields = rules["common_fields"]
    schema = build_output_schema(common_fields)

    results: list[dict[str, Any]] = []

    for file_path in list_txt_files(forms_dir):
        file_id = file_path.stem
        form_text = read_text(file_path)

        user_prompt = build_form_prompt(
            prompt_path=prompt_path,
            file_id=file_id,
            form_text=form_text,
            rules=rules,
        )

        result = chat_json(
            client=client,
            model_name=model_name,
            system_prompt=(
                "你是一個嚴格的資訊整理系統。"
                "你只能根據文本內容輸出 JSON。"
                "不可猜測、不可補寫、不可新增欄位。"
                "若未明確提及，必須填 null。"
            ),
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.0,
        )

        results.append(result)
        print(f"[FORM] done: {file_id}")

    write_json(output_path, results)
    print(f"[FORM] saved to: {output_path}")


def extract_calls(
    client: Client,
    model_name: str,
    rules: dict[str, Any],
    calls_dir: Path,
    prompt_path: Path,
    output_path: Path,
) -> None:
    common_fields = rules["common_fields"]
    schema = build_output_schema(common_fields)

    results: list[dict[str, Any]] = []

    for file_path in list_txt_files(calls_dir):
        file_id = file_path.stem
        call_text = read_text(file_path)

        user_prompt = build_call_prompt(
            prompt_path=prompt_path,
            file_id=file_id,
            call_text=call_text,
            rules=rules,
        )

        result = chat_json(
            client=client,
            model_name=model_name,
            system_prompt=(
                "你是一個嚴格的資訊擷取系統。"
                "你只能根據通話內容輸出 JSON。"
                "不可猜測、不可補寫、不可根據常識推論。"
                "若未明確提及，必須填 null。"
                "若某欄位通常來自系統資料，但通話中沒出現，也必須填 null。"
            ),
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.0,
        )

        results.append(result)
        print(f"[CALL] done: {file_id}")

    write_json(output_path, results)
    print(f"[CALL] saved to: {output_path}")


# =========================
# CLI
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize forms and/or extract fields from calls."
    )
    parser.add_argument(
        "--mode",
        choices=["form", "call", "both"],
        default="both",
        help="Run form normalization, call extraction, or both.",
    )
    parser.add_argument(
        "--rules",
        type=str,
        default=str(RULES_PATH),
        help="Path to final_rules_ch.json",
    )
    parser.add_argument(
        "--forms_dir",
        type=str,
        default=str(FORMS_DIR),
        help="Folder containing form .txt files",
    )
    parser.add_argument(
        "--calls_dir",
        type=str,
        default=str(CALLS_DIR),
        help="Folder containing call .txt files",
    )
    parser.add_argument(
        "--form_prompt",
        type=str,
        default=str(FORM_PROMPT_PATH),
        help="Path to normalize_form.prompt",
    )
    parser.add_argument(
        "--call_prompt",
        type=str,
        default=str(CALL_PROMPT_PATH),
        help="Path to extract_call.prompt",
    )
    parser.add_argument(
        "--form_output",
        type=str,
        default=str(OUTPUT_DIR / "normalized_forms.json"),
        help="Output JSON for normalized forms",
    )
    parser.add_argument(
        "--call_output",
        type=str,
        default=str(OUTPUT_DIR / "extracted_calls.json"),
        help="Output JSON for extracted calls",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=OLLAMA_HOST,
        help="Ollama host",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Ollama model name",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rules_path = Path(args.rules)
    forms_dir = Path(args.forms_dir)
    calls_dir = Path(args.calls_dir)
    form_prompt_path = Path(args.form_prompt)
    call_prompt_path = Path(args.call_prompt)
    form_output_path = Path(args.form_output)
    call_output_path = Path(args.call_output)

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    rules = read_json(rules_path)

    if "common_fields" not in rules:
        raise KeyError("final_rules_ch.json 缺少 common_fields")

    client = Client(host=args.host)

    if args.mode in ("form", "both"):
        normalize_forms(
            client=client,
            model_name=args.model,
            rules=rules,
            forms_dir=forms_dir,
            prompt_path=form_prompt_path,
            output_path=form_output_path,
        )

    if args.mode in ("call", "both"):
        extract_calls(
            client=client,
            model_name=args.model,
            rules=rules,
            calls_dir=calls_dir,
            prompt_path=call_prompt_path,
            output_path=call_output_path,
        )


if __name__ == "__main__":
    main()
