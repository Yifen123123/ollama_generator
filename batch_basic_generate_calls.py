from __future__ import annotations

import json
from pathlib import Path
from ollama import Client

client = Client(host="http://10.67.75.157:11434")

MODELS = [
    "qwen2.5:7b",
    "qwen3:8b",
    "qwen2.5:14b",
    "gpt-oss:20b",
]

N_PER_MODEL = 3
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

system_prompt = """
你是一個專門生成繁體中文保險業客服通話STT逐字稿的合成資料生成器。

## 語言風格規範

**L（客服專員）：**
- 開場固定用語：「[保險公司名稱]人壽您好 ，我是 [數字01-99]號專員，敝姓[客服姓氏]，很高興為您服務」
- 語氣：專業、有禮、耐心，遇到客戶情緒激動時會安撫
- 常用語：「感謝您的來電」「請問您的保單號碼是？」「我幫您確認一下」
  「稍等一下喔」「方便留您的電話嗎？」「這部分我幫您轉給專員處理」
- 會重複確認重要資訊（姓名、身分證字號末四碼、保單號）
- 句子簡短，避免一次說太多

**R（保戶/客戶）：**
- 口語化、句子不完整是正常的
- 常見情緒：困惑、擔心、急迫、偶有不滿
- 常見問題類型：理賠進度查詢、保費繳交問題、保單內容確認、
  受益人變更、停效復效、醫療收據補件
- 會使用台灣日常用語：「就是說」「這樣子」「對啊」「那個」「你們那邊」
- 有時會搞錯術語，例如把「理賠」說成「申請錢」「出險」

## 格式規範
- 嚴格使用 L: 和 R: 作為發話人標記
- 每輪一行，L 和 R 交替出現
- 不加時間戳記、不加任何額外標記
- STT 風格：無標點或標點不完整是允許的，但為了可讀性可保留逗號和句號
""".strip()

user_prompt = """
請生成一段繁體中文保險客服通話STT逐字稿。

條件：
- 問題類型：理賠進度查詢
- 客戶情緒：有點著急但不失禮
- 情境難度：中等
- 是否需核對資料：是
- 是否需轉專員：否
- 通話長度：10~14輪
- 加入自然口語、停頓感、資訊重複確認
- 對話要接近真實客服中心通話，不要太工整，不要像作文

只輸出逐字稿內容。
""".strip()

for model in MODELS:
    model_dir = OUT_DIR / model.replace(":", "_").replace(".", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, N_PER_MODEL + 1):
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            keep_alive="5m",
            options={"temperature": 0.9},
        )

        record = {
            "model": model,
            "sample_id": f"{model}_{i}",
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "output": response.message.content,
            "meta": {
                "total_duration": getattr(response, "total_duration", None),
                "eval_count": getattr(response, "eval_count", None),
                "prompt_eval_count": getattr(response, "prompt_eval_count", None),
            }
        }

        with open(model_dir / f"sample_{i:02d}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"saved: {model} sample_{i:02d}")
