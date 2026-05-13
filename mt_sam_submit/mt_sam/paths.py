from __future__ import annotations

import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_asset_path(path_text: str, packaged_relative: str) -> Path:
    path = Path(path_text).expanduser()
    if path.exists():
        return path.resolve()
    packaged = package_root() / packaged_relative
    if packaged.exists():
        return packaged.resolve()
    return path


def ensure_third_party_imports() -> None:
    third_party = package_root() / "third_party"
    if third_party.exists():
        third_party_text = str(third_party.resolve())
        if third_party_text not in sys.path:
            sys.path.insert(0, third_party_text)
