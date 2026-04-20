from pathlib import Path
from config import RULES_PATH, FORMS_DIR, OUTPUT_DIR, OLLAMA_HOST, MODEL_NAME
from io_utils import read_json, read_text, write_json, list_txt_files
from llm_client import OllamaExtractor
from prompt_builder import build_output_schema, build_form_prompt


def main():
    rules = read_json(RULES_PATH)
    common_fields = rules["common_fields"]
    schema = build_output_schema(common_fields)

    extractor = OllamaExtractor(OLLAMA_HOST, MODEL_NAME)

    results = []

    for path in list_txt_files(FORMS_DIR):
        file_id = path.stem
        form_text = read_text(path)

        user_prompt = build_form_prompt(
            file_id=file_id,
            form_text=form_text,
            rules=rules,
        )

        result = extractor.chat_json(
            system_prompt=(
                "你是一位會辦單欄位整理專家。"
                "你只能根據文本內容輸出 JSON，不可猜測。"
            ),
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.0,
        )
        results.append(result)
        print(f"[FORM] done: {file_id}")

    write_json(OUTPUT_DIR / "normalized_forms.json", results)


if __name__ == "__main__":
    main()
