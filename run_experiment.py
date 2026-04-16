from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from ollama import Client


# =========================
# 設定區
# =========================

OLLAMA_HOST = "http://10.67.75.157:11434"

MODELS = [
    "qwen2.5:7b",
    "qwen3:8b",
    "qwen2.5:14b",
    "gpt-oss:20b",
]

CALLS_DIR = Path("calls")
FORMS_DIR = Path("forms")
PROMPT_PATH = Path("prompts/form_generation.prompt")
OUTPUTS_DIR = Path("outputs")
REPORTS_DIR = Path("reports")

# 先只跑少量樣本
MAX_SAMPLES = 3

# 是否串流印出模型輸出
PRINT_STREAM = True


# =========================
# 資料結構
# =========================

@dataclass
class Sample:
    stem: str
    call_text: str
    form_text: str


@dataclass
class RunResult:
    model: str
    sample_id: str
    prompt: str
    raw_output: str
    parsed_output: dict[str, Any] | None
    parse_success: bool
    elapsed_seconds: float
    error: str | None


# =========================
# 工具函式
# =========================

def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_prompt(template: str, call_text: str) -> str:
    return template.replace("{call_text}", call_text)


def find_samples(calls_dir: Path, forms_dir: Path, max_samples: int | None = None) -> list[Sample]:
    call_files = sorted(calls_dir.glob("*.txt"))
    samples: list[Sample] = []

    for call_file in call_files:
        form_file = forms_dir / call_file.name
        if not form_file.exists():
            print(f"[WARN] 找不到對應 form：{form_file}")
            continue

        samples.append(
            Sample(
                stem=call_file.stem,
                call_text=load_text(call_file),
                form_text=load_text(form_file),
            )
        )

    if max_samples is not None:
        samples = samples[:max_samples]

    return samples


def extract_json_block(text: str) -> dict[str, Any] | None:
    """
    嘗試從模型輸出中抓出第一個 JSON 物件。
    """
    text = text.strip()

    # 先移除 markdown code fence
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 直接 parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 嘗試擷取第一個大括號區塊
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    return None


def call_model(client: Client, model_name: str, prompt: str, print_stream: bool = False) -> tuple[str, float]:
    start = time.perf_counter()
    chunks: list[str] = []

    stream = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for part in stream:
        content = part.message.content or ""
        chunks.append(content)
        if print_stream:
            print(content, end="", flush=True)

    if print_stream:
        print()

    elapsed = time.perf_counter() - start
    full_text = "".join(chunks)
    return full_text, elapsed


def save_result(output_dir: Path, result: RunResult, original_form_text: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "meta": {
            "model": result.model,
            "sample_id": result.sample_id,
            "elapsed_seconds": result.elapsed_seconds,
            "parse_success": result.parse_success,
            "error": result.error,
        },
        "prompt": result.prompt,
        "parsed_output": result.parsed_output,
        "raw_output": result.raw_output,
        "reference_form": original_form_text,
    }

    out_path = output_dir / f"{result.sample_id}.json"
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize_results(all_results: list[RunResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    by_model: dict[str, list[RunResult]] = {}
    for r in all_results:
        by_model.setdefault(r.model, []).append(r)

    for model, results in by_model.items():
        total = len(results)
        parse_ok = sum(1 for r in results if r.parse_success)
        avg_time = sum(r.elapsed_seconds for r in results) / total if total else 0.0

        summary[model] = {
            "total_samples": total,
            "parse_success_count": parse_ok,
            "parse_success_rate": round(parse_ok / total, 4) if total else 0.0,
            "avg_elapsed_seconds": round(avg_time, 4),
        }

    return summary


# =========================
# 主程式
# =========================

def main() -> None:
    client = Client(host=OLLAMA_HOST)

    prompt_template = load_text(PROMPT_PATH)
    samples = find_samples(CALLS_DIR, FORMS_DIR, max_samples=MAX_SAMPLES)

    if not samples:
        print("[ERROR] 沒有找到可用樣本。")
        return

    OUTPUTS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    all_results: list[RunResult] = []

    print(f"[INFO] 本次共測試 {len(samples)} 筆樣本，模型數量 {len(MODELS)}")

    for model_name in MODELS:
        print(f"\n===== 開始測試模型：{model_name} =====")
        model_dir = OUTPUTS_DIR / sanitize_model_name(model_name)

        for sample in samples:
            print(f"\n[INFO] sample={sample.stem}, model={model_name}")

            prompt = build_prompt(prompt_template, sample.call_text)

            try:
                raw_output, elapsed = call_model(
                    client=client,
                    model_name=model_name,
                    prompt=prompt,
                    print_stream=PRINT_STREAM,
                )

                parsed = extract_json_block(raw_output)
                parse_success = parsed is not None

                result = RunResult(
                    model=model_name,
                    sample_id=sample.stem,
                    prompt=prompt,
                    raw_output=raw_output,
                    parsed_output=parsed,
                    parse_success=parse_success,
                    elapsed_seconds=elapsed,
                    error=None,
                )

            except Exception as exc:
                result = RunResult(
                    model=model_name,
                    sample_id=sample.stem,
                    prompt=prompt,
                    raw_output="",
                    parsed_output=None,
                    parse_success=False,
                    elapsed_seconds=0.0,
                    error=f"{type(exc).__name__}: {exc}",
                )

            save_result(model_dir, result, sample.form_text)
            all_results.append(result)

    summary = summarize_results(all_results)
    summary_path = REPORTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 測試完成 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model_name)


if __name__ == "__main__":
    main()
