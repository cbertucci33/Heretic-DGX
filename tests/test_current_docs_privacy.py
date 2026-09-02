# SPDX-License-Identifier: AGPL-3.0-or-later

import re
from pathlib import Path


CURRENT_ROOTS = (
    Path("src"),
    Path("docs"),
    Path("tests"),
    Path("scripts"),
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
PROHIBITED_PRIVATE_PATHS = (
    re.compile(r"[A-Z]:\\Users\\[^\\/]+\\", re.IGNORECASE),
    re.compile(r"/home/[a-z_][a-z0-9_-]*/", re.IGNORECASE),
    re.compile(r"/mnt/[a-z]/Users/[^/]+/", re.IGNORECASE),
)


def test_current_tree_contains_no_concrete_user_home_paths() -> None:
    violations: list[str] = []
    for root in CURRENT_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if any(pattern.search(text) for pattern in PROHIBITED_PRIVATE_PATHS):
                violations.append(str(path))

    assert violations == []
