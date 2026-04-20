from config import OUTPUT_DIR
from io_utils import read_json, write_json


def index_by_file_id(items: list[dict]) -> dict[str, dict]:
    return {item["file_id"]: item for item in items}


def main():
    forms = read_json(OUTPUT_DIR / "normalized_forms.json")
    calls = read_json(OUTPUT_DIR / "extracted_calls.json")

    form_map = index_by_file_id(forms)
    call_map = index_by_file_id(calls)

    all_ids = sorted(set(form_map) & set(call_map))
    compare_results = []

    for file_id in all_ids:
        form_fields = form_map[file_id]["structured_fields"]
        call_fields = call_map[file_id]["structured_fields"]

        field_compare = {}
        for field in form_fields.keys():
            field_compare[field] = {
                "form_value": form_fields.get(field),
                "call_value": call_fields.get(field),
                "is_equal": form_fields.get(field) == call_fields.get(field),
            }

        compare_results.append({
            "file_id": file_id,
            "field_compare": field_compare
        })

    write_json(OUTPUT_DIR / "compare_results.json", compare_results)


if __name__ == "__main__":
    main()
