import json
from pathlib import Path


def json_to_txt(input_path: str, output_path: str):
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    lines = []

    for unit, info in data.items():
        categories = info.get("categories", [])
        fields = info.get("fields_union", [])

        lines.append(f"【{unit}】")

        # 類別
        if categories:
            lines.append(f"類別: {', '.join(categories)}")
        else:
            lines.append("類別: 無")

        lines.append("欄位:")

        if fields:
            for field in fields:
                lines.append(f"- {field}")
        else:
            lines.append("- 無")

        lines.append("\n" + "-" * 40 + "\n")

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"輸出完成: {output_path}")


if __name__ == "__main__":
    # 👉 改這裡就好
    json_to_txt(
        input_path="outputs/unit_form_fields_summary.json",
        output_path="outputs/unit_form_fields.txt"
    )

    json_to_txt(
        input_path="outputs/unit_call_fields_summary.json",
        output_path="outputs/unit_call_fields.txt"
    )
