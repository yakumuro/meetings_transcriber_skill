from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any

from common import (
    OUTPUTS_DIR, REGISTRY_PATH, atomic_write_text, ensure_directories,
    load_config, resolve_path,
)

_CONFIG = load_config(required=False)
OUTPUTS_DIR = resolve_path(_CONFIG.get("paths", {}).get("outputs", "outputs"))
REGISTRY_PATH = OUTPUTS_DIR / "REGISTRY.md"

HEADER = "| Дата | Папка | Ключевые темы | Ключевые решения |"
SEPARATOR = "|------|-------|---------------|------------------|"


def clean_cell(value: str, field: str) -> str:
    value = re.sub(r"\s+", " ", value).strip()
    if not value:
        raise RuntimeError(f"Поле «{field}» не должно быть пустым.")
    return value.replace("|", "\\|")


def validate_folder(folder: str) -> Path:
    if not folder or Path(folder).name != folder or folder in {".", ".."}:
        raise RuntimeError("Папка должна быть именем одного каталога внутри outputs.")
    meeting_dir = OUTPUTS_DIR / folder
    if not (meeting_dir / "transcript.md").is_file():
        raise RuntimeError(f"В outputs/{folder} отсутствует завершённый transcript.md.")
    return meeting_dir


def artifact_paths(meeting_dir: Path) -> dict[str, str | None]:
    return {
        name: str(meeting_dir / f"{name}.md") if (meeting_dir / f"{name}.md").is_file() else None
        for name in ("transcript", "summary", "email")
    }


def add_entry(folder: str, meeting_date: str, topics: str, decisions: str) -> dict[str, Any]:
    ensure_directories()
    meeting_dir = validate_folder(folder)
    try:
        date.fromisoformat(meeting_date)
    except ValueError as error:
        raise RuntimeError("Дата должна иметь формат ГГГГ-ММ-ДД.") from error
    topics = clean_cell(topics, "Темы")
    decisions = clean_cell(decisions, "Решения")
    if not REGISTRY_PATH.exists():
        atomic_write_text(
            REGISTRY_PATH,
            "# Реестр встреч\n\n## Реестр\n\n" + HEADER + "\n" + SEPARATOR + "\n",
        )
    content = REGISTRY_PATH.read_text(encoding="utf-8")
    if f"`{folder}`" in content:
        raise RuntimeError(f"Папка '{folder}' уже присутствует в REGISTRY.md.")
    lines = content.splitlines()
    separator_index = next(
        (index for index, line in enumerate(lines) if line.strip() == SEPARATOR), None
    )
    if separator_index is None:
        raise RuntimeError(f"В REGISTRY.md отсутствует ожидаемый разделитель таблицы: {SEPARATOR}")
    entry = f"| {meeting_date} | `{folder}` | {topics} | {decisions} |"
    lines.insert(separator_index + 1, entry)
    atomic_write_text(REGISTRY_PATH, "\n".join(lines) + "\n")
    return {
        "success": True,
        "registry": str(REGISTRY_PATH),
        "folder": folder,
        **artifact_paths(meeting_dir),
        "entry": entry,
    }


def parse_entries() -> list[dict[str, str]]:
    if not REGISTRY_PATH.exists():
        return []
    entries = []
    for line in REGISTRY_PATH.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|") or line.strip() in {HEADER, SEPARATOR}:
            continue
        cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line.strip().strip("|"))]
        if len(cells) != 4:
            continue
        folder = cells[1].strip("`")
        entries.append({
            "date": cells[0], "folder": folder, "topics": cells[2],
            "decisions": cells[3], "raw": line,
        })
    return entries


def search_entries(query: str = "", limit: int = 20) -> dict[str, Any]:
    if limit < 1:
        raise RuntimeError("Лимит должен быть положительным числом.")
    terms = query.casefold().split()
    matches = []
    for entry in parse_entries():
        haystack = " ".join(entry.values()).casefold()
        if all(term in haystack for term in terms):
            folder = entry["folder"]
            meeting_dir = OUTPUTS_DIR / folder
            matches.append({
                **{key: entry[key] for key in ("date", "folder", "topics", "decisions")},
                "summary": str(meeting_dir / "summary.md") if (meeting_dir / "summary.md").is_file() else None,
                "transcript": str(meeting_dir / "transcript.md") if (meeting_dir / "transcript.md").is_file() else None,
            })
        if len(matches) >= limit:
            break
    return {
        "query": query,
        "registry": str(REGISTRY_PATH),
        "count": len(matches),
        "matches": matches,
        "note": "Отсутствие совпадений не доказывает, что тема не обсуждалась. Используйте summary, затем transcript. Email в поиске не применяется.",
    }
