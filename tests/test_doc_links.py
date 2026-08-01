"""Every relative link in the repo's Markdown must point at something real.

This repo navigates by cross-reference: `roadmap.md` sends you to
`implementation-plan.md`, which sends you to a dated note in `notes/`, which
cites `architecture.md` by section. That web is how a decision's evidence gets
found six weeks later, and a dead link silently turns "here is the proof" into
"trust me".

It has already rotted once. Archiving the stale docs into `docs/archive/`
(2026-08-01) broke five sibling references -- `roadmap.md` resolves to
`docs/archive/roadmap.md` from inside that directory, which does not exist.
Nothing noticed, because nothing was looking: there was no link gate at all,
only a belief that there was one.

Only *relative* links are checked. External URLs are deliberately out of scope
-- verifying them needs the network, which would make this test flaky and slow
for a failure mode nobody on this project has hit.
"""

from __future__ import annotations

import pathlib
import re

# Inline code and fenced blocks are stripped before scanning. Without that,
# `cute.compile[options](fn, ...)` and `fused_moe_kernel[grid](...)` both parse
# as Markdown links and report as broken -- two false positives that were real
# when this checker was first run by hand.
_FENCED = re.compile(r"```.*?```", re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_LINK = re.compile(r"\[[^\]]*\]\(([^)]+)\)")

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _markdown_files() -> list[pathlib.Path]:
    return sorted(
        set(_REPO_ROOT.glob("*.md"))
        | set(_REPO_ROOT.glob("docs/**/*.md"))
        | set(_REPO_ROOT.glob("notes/**/*.md"))
    )


def _relative_targets(text: str) -> list[str]:
    text = _INLINE_CODE.sub(" ", _FENCED.sub(" ", text))
    targets = []
    for match in _LINK.finditer(text):
        # `[text](path "title")` -- the title is not part of the path.
        target = match.group(1).split()[0].strip() if match.group(1).strip() else ""
        if not target or target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(target)
    return targets


def test_relative_markdown_links_resolve():
    broken: list[str] = []
    checked = 0
    for md in _markdown_files():
        for target in _relative_targets(md.read_text(encoding="utf-8", errors="replace")):
            # Anchors are not validated -- only that the file being pointed
            # at exists. Validating anchors means reimplementing GitHub's
            # heading-slug rules (which differ for CJK headings, and this
            # repo's docs are largely Chinese), for a much rarer failure.
            path = target.split("#")[0]
            if not path:
                continue
            checked += 1
            if not (md.parent / path).resolve().exists():
                broken.append(f"{md.relative_to(_REPO_ROOT)} -> {target}")

    assert checked > 100, (
        f"only {checked} relative links found; the scanner probably stopped "
        "matching this repo's Markdown rather than the docs having shrunk"
    )
    assert not broken, "broken relative links:\n  " + "\n  ".join(broken)
