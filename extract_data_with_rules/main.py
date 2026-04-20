from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ollama import Client

from src.prompt_builder import (
    build_call_prompt,from __future__ import annotations

import argparse
import json
import re
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


def normalize_text(text: str) -> str:
    """
    基礎文字清理：
    - 去掉前後空白
    - 把連續空白壓成一個空格
    """
    return " ".join(text.strip().split())


# =========================
# JSON 安全解析
# =========================
def safe_json_loads(text: str) -> dict[str, Any]:
    """
    從模型輸出中安全抽取 JSON。
    支援：
    1. 純 JSON
    2. 前後夾雜說明文字
    3. markdown code block
    """
    if text is None:
        raise ValueError("Model response is None.")

    text = text.strip()

    if not text:
        raise ValueError("Model response is empty.")

    # 1) 先直接 parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2) 去掉 ```json ... ```
    code_block_match = re.search(
        r"```(?:json)?\s*(\{.*\})\s*```",
        text,
        re.DOTALL,
    )
    if code_block_match:
        candidate = code_block_match.group(1).strip()
        return json.loads(candidate)

    # 3) 抓第一個 JSON object
    obj_match = re.search(r"\{.*\}", text, re.DOTALL)
    if obj_match:
        candidate = obj_match.group(0).strip()
        return json.loads(candidate)

    raise ValueError(f"No JSON object found in model response:\n{text}")


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
    show_raw_output: bool = True,
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

    content = response.get("message", {}).get("content", "")

    if show_raw_output:
        print("\n===== RAW MODEL OUTPUT START =====")
        print(content if content else "[EMPTY RESPONSE]")
        print("===== RAW MODEL OUTPUT END =====\n")

    return safe_json_loads(content)


# =========================
# 單筆結果驗證
# =========================
def validate_result_shape(
    result: dict[str, Any],
    file_id: str,
    source_type: str,
    common_fields: list[str],
) -> dict[str, Any]:
    """
    額外做一層保底驗證，避免模型少欄位或亂欄位。
    缺的欄位補 None，多的欄位移除。
    """
    if not isinstance(result, dict):
        raise ValueError("Model output is not a dict.")

    result["file_id"] = str(result.get("file_id", file_id))
    result["source_type"] = source_type

    structured_fields = result.get("structured_fields", {})
    if not isinstance(structured_fields, dict):
        structured_fields = {}

    fixed_fields: dict[str, Any] = {}
    for field in common_fields:
        value = structured_fields.get(field, None)
        if value is not None and not isinstance(value, str):
            value = str(value)
        fixed_fields[field] = value

    result["structured_fields"] = fixed_fields
    return result


# =========================
# 錯誤紀錄
# =========================
def build_error_record(
    file_id: str,
    file_path: Path,
    source_type: str,
    error: Exception,
) -> dict[str, Any]:
    return {
        "file_id": file_id,
        "file_path": str(file_path),
        "source_type": source_type,
        "error_type": type(error).__name__,
        "error_message": str(error),
    }


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
    error_output_path: Path,
    limit: int | None = None,
) -> None:
    common_fields = rules["common_fields"]
    schema = build_output_schema(common_fields)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    files = list_txt_files(forms_dir)
    if limit is not None:
        files = files[:limit]

    for idx, file_path in enumerate(files, start=1):
        file_id = file_path.stem
        print(f"[FORM] processing {idx}/{len(files)}: {file_id}")

        try:
            form_text = normalize_text(read_text(file_path))

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

            result = validate_result_shape(
                result=result,
                file_id=file_id,
                source_type="form",
                common_fields=common_fields,
            )

            results.append(result)
            print(f"[FORM] success: {file_id}")

        except Exception as e:
            print(f"[FORM] failed: {file_id} -> {type(e).__name__}: {e}")
            errors.append(
                build_error_record(
                    file_id=file_id,
                    file_path=file_path,
                    source_type="form",
                    error=e,
                )
            )

    write_json(output_path, results)
    write_json(error_output_path, errors)

    print(f"[FORM] success saved to: {output_path}")
    print(f"[FORM] error log saved to: {error_output_path}")
    print(f"[FORM] total success: {len(results)}")
    print(f"[FORM] total failed: {len(errors)}")


def extract_calls(
    client: Client,
    model_name: str,
    rules: dict[str, Any],
    calls_dir: Path,
    prompt_path: Path,
    output_path: Path,
    error_output_path: Path,
    limit: int | None = None,
) -> None:
    common_fields = rules["common_fields"]
    schema = build_output_schema(common_fields)

    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    files = list_txt_files(calls_dir)
    if limit is not None:
        files = files[:limit]

    for idx, file_path in enumerate(files, start=1):
        file_id = file_path.stem
        print(f"[CALL] processing {idx}/{len(files)}: {file_id}")

        try:
            call_text = normalize_text(read_text(file_path))

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

            result = validate_result_shape(
                result=result,
                file_id=file_id,
                source_type="call",
                common_fields=common_fields,
            )

            results.append(result)
            print(f"[CALL] success: {file_id}")

        except Exception as e:
            print(f"[CALL] failed: {file_id} -> {type(e).__name__}: {e}")
            errors.append(
                build_error_record(
                    file_id=file_id,
                    file_path=file_path,
                    source_type="call",
                    error=e,
                )
            )

    write_json(output_path, results)
    write_json(error_output_path, errors)

    print(f"[CALL] success saved to: {output_path}")
    print(f"[CALL] error log saved to: {error_output_path}")
    print(f"[CALL] total success: {len(results)}")
    print(f"[CALL] total failed: {len(errors)}")


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
        "--form_error_output",
        type=str,
        default=str(OUTPUT_DIR / "normalized_forms_errors.json"),
        help="Error JSON for normalized forms",
    )
    parser.add_argument(
        "--call_error_output",
        type=str,
        default=str(OUTPUT_DIR / "extracted_calls_errors.json"),
        help="Error JSON for extracted calls",
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
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N files for debugging.",
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
    form_error_output_path = Path(args.form_error_output)
    call_error_output_path = Path(args.call_error_output)

    if not rules_path.exists():
        raise FileNotFoundError(f"Rules file not found: {rules_path}")

    if not form_prompt_path.exists():
        raise FileNotFoundError(f"Form prompt file not found: {form_prompt_path}")

    if not call_prompt_path.exists():
        raise FileNotFoundError(f"Call prompt file not found: {call_prompt_path}")

    rules = read_json(rules_path)

    if "common_fields" not in rules:
        raise KeyError("final_rules_ch.json 缺少 common_fields")

    common_fields = rules["common_fields"]
    if not isinstance(common_fields, list) or not common_fields:
        raise ValueError("common_fields 必須是非空 list")

    client = Client(host=args.host)

    if args.mode in ("form", "both"):
        normalize_forms(
            client=client,
            model_name=args.model,
            rules=rules,
            forms_dir=forms_dir,
            prompt_path=form_prompt_path,
            output_path=form_output_path,
            error_output_path=form_error_output_path,
            limit=args.limit,
        )

    if args.mode in ("call", "both"):
        extract_calls(
            client=client,
            model_name=args.model,
            rules=rules,
            calls_dir=calls_dir,
            prompt_path=call_prompt_path,
            output_path=call_output_path,
            error_output_path=call_error_output_path,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
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
