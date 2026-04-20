from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

RULES_PATH = BASE_DIR / "data" / "final_rules_ch.json"
FORMS_DIR = BASE_DIR / "data" / "forms"
CALLS_DIR = BASE_DIR / "data" / "calls"
OUTPUT_DIR = BASE_DIR / "outputs"

OLLAMA_HOST = "http://10.67.75.157:11434"
MODEL_NAME = "gpt-oss:20b"
