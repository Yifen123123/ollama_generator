import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def list_txt_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.txt"))
