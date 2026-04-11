"""Patch causal-conv1d / mamba-ssm setup.py in-place for sm_100-only builds.

Copied into the Modal image at /tmp/patch_setup.py and invoked during the
mamba-ssm image layer build. See scripts/modal_cutile_b200.py::_image_with_mamba.

Rationale: both packages hardcode a full `cc_flag` list covering compute_75
through compute_121 in their setup.py. On Modal's builder that translates to
40+ min of compile time PLUS Modal's container log rate limit killing the
build because ptxas emits ~10k info lines per kernel. We only need sm_100
(B200) so we strip everything else and also drop `--ptxas-options=-v`.
"""
from __future__ import annotations

import sys


def _indent_of(line: str) -> int:
    i = 0
    while i < len(line) and line[i] == " ":
        i += 1
    return i


def patch(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()
    out: list[str] = []
    i = 0
    kept_archs: list[str] = []
    while i < len(lines):
        ln = lines[i]
        # Match the `cc_flag.append("-gencode")` line paired with the next
        # `cc_flag.append("arch=compute_N,code=sm_N")` line.
        if "cc_flag.append(\"-gencode\")" in ln and i + 1 < len(lines):
            nxt = lines[i + 1]
            if "arch=compute_" in nxt:
                if "arch=compute_100" in nxt:
                    out.append(ln)
                    out.append(nxt)
                    kept_archs.append("100")
                # else: drop both lines (skip the non-sm_100 arch pair)
                i += 2
                continue
        # Drop --ptxas-options=-v so the build doesn't spam the log.
        if "--ptxas-options=-v" in ln:
            i += 1
            continue
        out.append(ln)
        i += 1

    # Second pass: insert `pass` statements to fill any empty `if:` / `else:` /
    # `elif:` blocks that our gencode deletion may have emptied. We detect
    # these by looking for an `if/else/elif` line whose indented body is
    # entirely missing (i.e. the next non-blank line is at the same or lesser
    # indent level). Covers `causal-conv1d` and `mamba-ssm` setup.py shapes.
    fixed: list[str] = []
    n = len(out)
    j = 0
    while j < n:
        ln = out[j]
        fixed.append(ln)
        stripped = ln.lstrip()
        if (
            stripped.startswith(("if ", "elif ", "else:", "for ", "while "))
            and stripped.rstrip().endswith(":")
        ):
            header_indent = _indent_of(ln)
            body_indent = header_indent + 4
            # Look ahead at the next non-blank line. If it's at <= header
            # indent (or end of file), the block is empty.
            k = j + 1
            while k < n and out[k].strip() == "":
                k += 1
            if k >= n or _indent_of(out[k]) <= header_indent:
                fixed.append(" " * body_indent + "pass")
        j += 1

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(fixed) + "\n")

    print(f"[patch_setup] rewrote {path}: kept_archs={kept_archs}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_setup.py <setup.py>")
    patch(sys.argv[1])
