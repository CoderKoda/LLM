"""Local Koda model registry."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "models.json"


def load() -> dict:
    if REGISTRY.exists():
        try:
            return json.loads(REGISTRY.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {"models": []}


def save(data: dict) -> None:
    REGISTRY.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def register(name: str, path: str, kind: str = "koda", notes: str = "") -> None:
    data = load()
    items = [m for m in data["models"] if m.get("name") != name]
    items.append({
        "name": name,
        "path": path,
        "kind": kind,
        "notes": notes,
        "registered_at": datetime.now().isoformat(timespec="seconds"),
    })
    data["models"] = sorted(items, key=lambda x: x["name"].lower())
    save(data)
    print(f"registered: {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Koda model registry")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("add")
    r.add_argument("name")
    r.add_argument("path")
    r.add_argument("--kind", default="koda")
    r.add_argument("--notes", default="")
    i = sub.add_parser("info")
    i.add_argument("name")
    a = p.parse_args()
    data = load()
    if a.cmd == "list":
        for m in data["models"]:
            print(f"{m['name']:<24} {m['kind']:<10} {m['path']}")
        return
    if a.cmd == "add":
        register(a.name, a.path, a.kind, a.notes)
        return
    match = next((m for m in data["models"] if m.get("name") == a.name), None)
    if not match:
        raise SystemExit(f"model not found: {a.name}")
    print(json.dumps(match, indent=2))


if __name__ == "__main__":
    main()
