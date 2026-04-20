from pathlib import Path
from config import RULES_PATH, CALLS_DIR, OUTPUT_DIR, OLLAMA_HOST, MODEL_NAME
from io_utils import read_json, read_text, write_json, list_txt_files
from llm_client import OllamaExtractor
from prompt_builder import build_output_schema, build_call_prompt


def main():
    rules = read_json(RULES_PATH)
    common_fields = rules["common_fields"]
    schema = build_output_schema(common_fields)

    extractor = OllamaExtractor(OLLAMA_HOST, MODEL_NAME)

    results = []

    for path in list_txt_files(CALLS_DIR):
        file_id = path.stem
        call_text = read_text(path)

        user_prompt = build_call_prompt(
            file_id=file_id,
            call_text=call_text,
            rules=rules,
        )

        result = extractor.chat_json(
            system_prompt=(
                "你是一位客服通話資訊擷取專家。"
                "你只能根據通話內容輸出 JSON，不可補寫未提及內容。"
            ),
            user_prompt=user_prompt,
            schema=schema,
            temperature=0.0,
        )
        results.append(result)
        print(f"[CALL] done: {file_id}")

    write_json(OUTPUT_DIR / "extracted_calls.json", results)


if __name__ == "__main__":
    main()
