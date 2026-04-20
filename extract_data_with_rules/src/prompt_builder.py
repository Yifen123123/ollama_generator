from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_prompt_template(path: str | Path) -> str:
    """
    讀取 prompt 模板檔案內容。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def fill_prompt_template(template: str, **kwargs: Any) -> str:
    """
    將模板中的 {key} 以 kwargs[key] 取代。
    非字串值會先轉成 JSON 字串。
    """
    result = template

    for key, value in kwargs.items():
        if value is None:
            replacement = "null"
        elif isinstance(value, str):
            replacement = value
        else:
            replacement = json.dumps(value, ensure_ascii=False, indent=2)

        result = result.replace(f"{{{key}}}", replacement)

    return result


def build_output_schema(common_fields: list[str]) -> dict[str, Any]:
    """
    建立統一輸出格式 schema。
    """
    if not common_fields:
        raise ValueError("common_fields is empty.")

    return {
        "type": "object",
        "properties": {
            "file_id": {"type": "string"},
            "source_type": {"type": "string"},
            "structured_fields": {
                "type": "object",
                "properties": {
                    field: {"type": ["string", "null"]}
                    for field in common_fields
                },
                "required": common_fields,
                "additionalProperties": False,
            },
        },
        "required": ["file_id", "source_type", "structured_fields"],
        "additionalProperties": False,
    }


def build_form_prompt(
    prompt_path: str | Path,
    file_id: str,
    form_text: str,
    rules: dict[str, Any],
) -> str:
    """
    建立會辦單 normalization 用的 prompt。
    """
    template = load_prompt_template(prompt_path)

    common_fields = rules.get("common_fields", [])
    field_mapping_rules = rules.get("field_mapping_rules", [])
    writing_rules = rules.get("writing_rules", [])

    return fill_prompt_template(
        template,
        file_id=file_id,
        form_text=form_text,
        common_fields=common_fields,
        field_mapping_rules=field_mapping_rules,
        writing_rules=writing_rules,
    )


def build_call_prompt(
    prompt_path: str | Path,
    file_id: str,
    call_text: str,
    rules: dict[str, Any],
) -> str:
    """
    建立通話紀錄 extraction 用的 prompt。
    """
    template = load_prompt_template(prompt_path)

    common_fields = rules.get("common_fields", [])
    field_mapping_rules = rules.get("field_mapping_rules", [])
    llm_instruction = rules.get("llm_instruction", "")

    return fill_prompt_template(
        template,
        file_id=file_id,
        call_text=call_text,
        common_fields=common_fields,
        field_mapping_rules=field_mapping_rules,
        llm_instruction=llm_instruction,
    )
