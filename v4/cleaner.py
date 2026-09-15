"""Turn scraped HTML into training-friendly prose.

Trafilatura removes scripts, navigation, headers, footers, ads and other
boilerplate before we apply a few conservative language-quality checks.
"""

from __future__ import annotations

import re

from trafilatura import extract


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
SPACE_RE = re.compile(r"[ \t\u00a0]+")


def clean_text(html: str) -> str:
    text = extract(
        html,
        output_format="txt",
        include_comments=False,
        include_tables=False,
        include_links=False,
        favor_precision=True,
        deduplicate=True,
    ) or ""

    paragraphs: list[str] = []
    for raw in text.splitlines():
        line = SPACE_RE.sub(" ", raw).strip()
        line = URL_RE.sub("", line).strip()
        if not line:
            continue
        if looks_like_sentence(line):
            paragraphs.append(line)

    # Collapse repeated blank lines and avoid tiny fragments that are mostly UI.
    return "\n\n".join(paragraphs)


def looks_like_sentence(text: str) -> bool:
    if len(text) < 30:
        return False

    words = text.split()
    if len(words) < 5:
        return False

    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(1, len(text)) < 0.55:
        return False

    # Navigation-like lines are unusually link/menu shaped.
    if text.count("|") >= 3 or text.count("›") >= 3:
        return False

    # Keep normal headings and prose, but reject obvious isolated UI labels.
    if len(words) <= 10 and not re.search(r"[.!?:;]$", text):
        return False

    return True
