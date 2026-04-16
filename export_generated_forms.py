import json
from pathlib import Path


INPUT_DIR = Path("outputs")
OUTPUT_DIR = Path("generated_forms")


def format_text(parsed: dict) -> str:
    ci = parsed.get("customer_info", {})
    cs = parsed.get("call_summary", {})
    bf = parsed.get("bullet_form", [])

    lines = []

    # 客戶資訊
    lines.append("【客戶資訊】")
    lines.append(f"姓名：{ci.get('name')}")
    lines.append(f"電話：{ci.get('phone')}")
    lines.append(f"身分證：{ci.get('id_number')}")
    lines.append(f"生日：{ci.get('birthday')}")
    lines.append(f"地址：{ci.get('address')}")
    lines.append(f"保單號碼：{ci.get('policy_number')}")
    lines.append("")

    # 主問題
    lines.append("【主問題】")
    lines.append(cs.get("main_issue", ""))
    lines.append("")

    # 會辦單
    lines.append("【會辦單】")
    if isinstance(bf, list):
        for item in bf:
            lines.append(f"- {item}")
    else:
        lines.append(str(bf))

    return "\n".join(lines)


def main():
    for model_dir in INPUT_DIR.iterdir():
        if not model_dir.is_dir():
            continue

        out_model_dir = OUTPUT_DIR / model_dir.name
        out_model_dir.mkdir(parents=True, exist_ok=True)

        for json_file in model_dir.glob("*.json"):
            data = json.loads(json_file.read_text(encoding="utf-8"))

            parsed = data.get("parsed_output")
            if not parsed:
                continue

            text = format_text(parsed)

            out_path = out_model_dir / f"{json_file.stem}.txt"
            out_path.write_text(text, encoding="utf-8")

    print("✅ 全部轉換完成")


if __name__ == "__main__":
    main()
