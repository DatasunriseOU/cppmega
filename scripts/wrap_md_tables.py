#!/usr/bin/env python3
"""
Reflow GitHub-flavored markdown tables so each row fits within a target width,
wrapping long cells onto continuation lines via py-markdown-table.

Usage:
    wrap_md_tables.py FILE [-w MAX_ROW] [-c MAX_COL] [-i] [-o OUT]

Defaults: max-row 80, max-col auto-distributed proportionally to the natural
column widths. Tables are detected as a header row, a `|---|...|` separator,
and one or more body rows.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from py_markdown_table.markdown_table import markdown_table


SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [normalize_ascii(c.strip()) for c in s.split("|")]


def is_table_row(line: str) -> bool:
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and s.count("|") >= 2


def longest_word(text: str, delim: str = " ") -> int:
    if not text:
        return 0
    return max((len(w) for w in text.split(delim)), default=0)


def natural_width(text: str) -> int:
    if not text:
        return 0
    return max(len(line) for line in text.splitlines() or [text])


# Map non-ASCII glyphs that monospace terminals/markdown renderers commonly
# misalign to safe ASCII equivalents. Applied to cell content before width
# calculation so the codepoint count == visual cell count.
ASCII_REPLACEMENTS = {
    "✓": "v",
    "✔": "v",
    "✗": "x",
    "✘": "x",
    "→": "->",
    "←": "<-",
    "↑": "^",
    "↓": "v",
    "⇒": "=>",
    "⇐": "<=",
    "—": "--",
    "–": "-",
    "−": "-",
    "•": "*",
    "·": ".",
    "…": "...",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
    "×": "x",
    "÷": "/",
    "≈": "~=",
    "≠": "!=",
    "≤": "<=",
    "≥": ">=",
    "±": "+/-",
    "©": "(c)",
    "®": "(R)",
    "™": "(TM)",
    "\u00a0": " ",  # NBSP
    "\u200b": "",   # zero-width space
}


_HASH_NUM_RE = re.compile(r"(?<!#)#(\d{3,})")


def normalize_ascii(text: str) -> str:
    if not text:
        return text
    for k, v in ASCII_REPLACEMENTS.items():
        if k in text:
            text = text.replace(k, v)
    # Double the `#` in `#NNN+` (PR/issue refs) so markdown editors don't parse
    # it as a CSS color — the color-swatch indicator pushes cell widths off.
    text = _HASH_NUM_RE.sub(r"##\1", text)
    return text


def break_long_words(text: str, width: int, delim: str = " ") -> str:
    """Hard-break any word in `text` longer than `width` by inserting `delim`,
    so the multiline renderer can wrap the cell to exactly `width`."""
    if width <= 0 or not text:
        return text
    out: list[str] = []
    for w in text.split(delim):
        while len(w) > width:
            out.append(w[:width])
            w = w[width:]
        out.append(w)
    return delim.join(out)


def allocate_widths(
    headers: list[str],
    rows: list[list[str]],
    max_row: int,
    max_col: int | None,
) -> dict[str, int]:
    n = len(headers)
    # padding_width=1 → each cell gets 2 padding chars; (n+1) border pipes.
    overhead = (n + 1) + 2 * n
    budget = max(n, max_row - overhead)

    cols = list(zip(*([headers] + rows))) if rows else [(h,) for h in headers]
    naturals = [max(1, max(natural_width(c) for c in col)) for col in cols]

    # Per-column floor: protect the header's longest word and any body word up
    # to BREAK_THRESHOLD. Beyond that (URLs, hashes, long identifiers), allow
    # hard-breaks so the column can shrink to fit the row budget.
    BREAK_THRESHOLD = 16
    mins: list[int] = []
    for i, col in enumerate(cols):
        head_w = longest_word(headers[i])
        body_w = max((longest_word(c) for c in col[1:]), default=0)
        mins.append(max(1, head_w, min(body_w, BREAK_THRESHOLD)))

    if max_col is not None:
        caps = [max(mins[i], max_col) for i in range(n)]
    else:
        caps = [budget for _ in range(n)]

    total_nat = sum(naturals) or 1
    widths = [
        max(mins[i], min(caps[i], round(budget * naturals[i] / total_nat)))
        for i in range(n)
    ]

    def shrink_one() -> bool:
        order = sorted(range(n), key=lambda i: widths[i] - mins[i], reverse=True)
        for i in order:
            if widths[i] > mins[i]:
                widths[i] -= 1
                return True
        return False

    def grow_one() -> bool:
        under_nat = [i for i in range(n) if widths[i] < naturals[i]]
        pool = under_nat or [i for i in range(n) if widths[i] < caps[i]]
        if not pool:
            return False
        i = max(pool, key=lambda j: naturals[j])
        widths[i] += 1
        return True

    while sum(widths) > budget and shrink_one():
        pass
    while sum(widths) < budget and grow_one():
        pass

    return {h: widths[i] for i, h in enumerate(headers)}


def render_table(
    headers: list[str],
    rows: list[list[str]],
    max_row: int,
    max_col: int | None,
) -> str:
    # py-markdown-table rejects duplicate keys, so disambiguate header collisions.
    keys: list[str] = []
    seen: dict[str, int] = {}
    for h in headers:
        base = h or " "
        if base in seen:
            seen[base] += 1
            keys.append(f"{base}\u200b{seen[base]}")
        else:
            seen[base] = 0
            keys.append(base)

    data = [{keys[i]: (row[i] if i < len(row) else "") for i in range(len(keys))} for row in rows]
    if not data:
        data = [{k: "" for k in keys}]

    widths = allocate_widths(keys, [list(d.values()) for d in data], max_row, max_col)

    # Hard-break any token longer than its column width so the multiline
    # renderer can honor the requested width exactly.
    keys_broken = [break_long_words(k, widths[k]) for k in keys]
    rename = dict(zip(keys, keys_broken))
    widths = {rename[k]: v for k, v in widths.items()}
    data = [{rename[k]: break_long_words(v, widths[rename[k]]) for k, v in row.items()} for row in data]

    md = (
        markdown_table(data)
        .set_params(
            row_sep="markdown",
            padding_width=1,
            padding_weight="right",
            multiline=widths,
            multiline_strategy="rows_and_header",
            quote=False,
        )
        .get_markdown()
    )
    # Strip the zero-width disambiguation marker from the rendered header.
    return md.replace("\u200b", "").rstrip("\n")


def reflow(text: str, max_row: int, max_col: int | None) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        # Try to detect: header row, separator row.
        if (
            i + 1 < n
            and is_table_row(lines[i])
            and SEP_RE.match(lines[i + 1])
        ):
            header = split_row(lines[i])
            j = i + 2
            rows: list[list[str]] = []
            while j < n and is_table_row(lines[j]):
                cells = split_row(lines[j])
                if len(cells) < len(header):
                    cells += [""] * (len(header) - len(cells))
                elif len(cells) > len(header):
                    cells = cells[: len(header)]
                rows.append(cells)
                j += 1
            try:
                out.append(render_table(header, rows, max_row, max_col))
            except Exception as e:
                print(
                    f"warn: table at line {i + 1} not reflowed ({e}); kept original",
                    file=sys.stderr,
                )
                out.extend(lines[i:j])
            i = j
            continue
        out.append(lines[i])
        i += 1
    trailing_nl = "\n" if text.endswith("\n") else ""
    return "\n".join(out) + trailing_nl


def main() -> int:
    doc_lines = (__doc__ or "").strip().splitlines()
    ap = argparse.ArgumentParser(description=doc_lines[0] if doc_lines else "")
    ap.add_argument("file", type=Path, help="markdown file to process")
    ap.add_argument("-w", "--max-row", type=int, default=80, help="target max row width (default 80)")
    ap.add_argument(
        "-c",
        "--max-col",
        type=int,
        default=None,
        help="per-column hard cap (default: auto from max-row, proportional)",
    )
    ap.add_argument("-i", "--in-place", action="store_true", help="rewrite the file in place")
    ap.add_argument("-o", "--output", type=Path, help="write result to OUTPUT instead of stdout")
    args = ap.parse_args()

    src = args.file.read_text(encoding="utf-8")
    result = reflow(src, args.max_row, args.max_col)

    if args.in_place:
        args.file.write_text(result, encoding="utf-8")
    elif args.output:
        args.output.write_text(result, encoding="utf-8")
    else:
        sys.stdout.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
