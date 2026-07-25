#!/usr/bin/env python3
"""Extract cross-file dependency graphs from repos using DRY-RUN build commands.

For each repo, detects the build system and runs the appropriate dry-run /
configure-only command to produce a full cross-file dependency tree.  No
compilation ever happens.

Supported build systems:
  - CMake   → cmake --graphviz (configure only)
  - Make    → make -p -n (print rule database, no execution)
  - Bazel   → bazel query --output=graph (or static BUILD parse fallback)
  - Ninja   → ninja -t graph (or static parse of build.ninja)
  - Shell   → shellcheck -x --format=json (or manual source resolution)

Output: outputs/build_graphs/{repo_name}.json per repo.

SAFETY: DRY RUN ONLY.  Never compile.  Never run `make` without `-n`.
Never run `cmake --build`.  Timeout 60s per repo.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIMEOUT_S = 60
DEFAULT_REPOS_DIR = "outputs/source_cache/code"
OUTPUT_DIR = "outputs/build_graphs"

# Build system detection markers (checked in priority order)
BUILD_SYSTEM_MARKERS = [
    ("bazel", ["WORKSPACE", "WORKSPACE.bazel", "MODULE.bazel"]),
    ("cmake", ["CMakeLists.txt"]),
    ("ninja", ["build.ninja"]),
    ("make", ["GNUmakefile", "makefile", "Makefile"]),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Node:
    id: str
    type: str = "unknown"
    file: str = ""

    def to_dict(self) -> dict:
        d: dict[str, str] = {"id": self.id, "type": self.type}
        if self.file:
            d["file"] = self.file
        return d


@dataclass
class Edge:
    src: str
    dst: str
    kind: str

    def to_dict(self) -> dict:
        return {"from": self.src, "to": self.dst, "kind": self.kind}


@dataclass
class Diagnostic:
    file: str
    line: int
    severity: str
    message: str

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "severity": self.severity,
            "message": self.message,
        }


@dataclass
class GraphResult:
    repo: str
    build_system: str
    graph_file: str = ""
    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    diagnostics: list[Diagnostic] = field(default_factory=list)
    cross_file_includes: list[str] = field(default_factory=list)
    configure_success: bool = False

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "build_system": self.build_system,
            "graph_file": self.graph_file,
            "nodes": [n.to_dict() for n in self.nodes],
            "edges": [e.to_dict() for e in self.edges],
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "cross_file_includes": self.cross_file_includes,
            "configure_success": self.configure_success,
        }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def tool_available(name: str) -> bool:
    """Check if a CLI tool is on PATH."""
    return shutil.which(name) is not None


def run_cmd(
    cmd: list[str],
    *,
    cwd: str | None = None,
    timeout: int = TIMEOUT_S,
    env_extra: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    """Run a command with timeout.  Returns (returncode, stdout, stderr)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired:
        return -1, "", f"TIMEOUT after {timeout}s"
    except FileNotFoundError:
        return -2, "", f"Command not found: {cmd[0]}"


def detect_build_system(repo_path: Path) -> str | None:
    """Detect build system by marker files.  Returns system name or None."""
    for system, markers in BUILD_SYSTEM_MARKERS:
        for marker in markers:
            if (repo_path / marker).exists():
                return system
    # Check for shell scripts with source statements
    sh_files = list(repo_path.glob("*.sh")) + list(repo_path.glob("scripts/*.sh"))
    if sh_files:
        return "shell"
    return None


def find_cross_file_includes_cmake(repo_path: Path) -> list[str]:
    """Find all CMakeLists.txt and .cmake files referenced (cross-file)."""
    includes: list[str] = []
    for p in sorted(repo_path.rglob("CMakeLists.txt")):
        rel = str(p.relative_to(repo_path))
        if rel != "CMakeLists.txt":
            includes.append(rel)
    for p in sorted(repo_path.rglob("*.cmake")):
        includes.append(str(p.relative_to(repo_path)))
    return includes


# ---------------------------------------------------------------------------
# CMake extractor
# ---------------------------------------------------------------------------


def extract_cmake(repo_path: Path, repo_name: str, tmp_dir: str) -> GraphResult:
    """Run cmake configure with --graphviz to get dependency DOT graph."""
    result = GraphResult(repo=repo_name, build_system="cmake")
    build_dir = os.path.join(tmp_dir, "build")
    dot_file = os.path.join(tmp_dir, "deps.dot")
    log_file = os.path.join(tmp_dir, "cmake.log")

    os.makedirs(build_dir, exist_ok=True)

    cmd = [
        "cmake",
        "-B", build_dir,
        "-S", str(repo_path),
        f"--graphviz={dot_file}",
    ]

    rc, stdout, stderr = run_cmd(cmd, cwd=str(repo_path))
    result.configure_success = rc == 0

    # Capture diagnostics from stderr/stdout
    combined_log = stdout + "\n" + stderr
    _parse_cmake_diagnostics(combined_log, result)

    # Parse the DOT file (cmake may produce deps.dot or deps.dot.<target>)
    dot_path = Path(dot_file)
    if dot_path.exists():
        result.graph_file = "deps.dot"
        _parse_dot_file(dot_path, result)
    else:
        # cmake graphviz sometimes names files differently
        dot_candidates = list(Path(tmp_dir).glob("deps.dot*"))
        if dot_candidates:
            result.graph_file = dot_candidates[0].name
            _parse_dot_file(dot_candidates[0], result)

    # Parse CMakeLists.txt files for source file references (target→source edges)
    _parse_cmake_sources(repo_path, result)

    result.cross_file_includes = find_cross_file_includes_cmake(repo_path)
    return result


def _parse_cmake_diagnostics(log_text: str, result: GraphResult) -> None:
    """Extract CMake Error/Warning lines from configure log."""
    patterns = [
        (r"CMake Error at (.+?):(\d+)", "error"),
        (r"CMake Warning at (.+?):(\d+)", "warning"),
        (r"CMake Error:", "error"),
        (r"CMake Warning:", "warning"),
    ]
    for line in log_text.splitlines():
        for pat, severity in patterns:
            m = re.search(pat, line)
            if m:
                fname = m.group(1) if m.lastindex and m.lastindex >= 1 else ""
                lineno = int(m.group(2)) if m.lastindex and m.lastindex >= 2 else 0
                # Message is the rest of the line after the match
                msg = line[m.end():].strip().lstrip(":").strip()
                if not msg:
                    msg = line.strip()
                result.diagnostics.append(
                    Diagnostic(file=fname, line=lineno, severity=severity, message=msg)
                )
                break


def _parse_cmake_sources(repo_path: Path, result: GraphResult) -> None:
    """Parse CMakeLists.txt files for add_executable/add_library source lists."""
    # Match: add_executable(name src1 src2 ...) or add_library(name [STATIC|SHARED|...] src1 ...)
    target_re = re.compile(
        r"add_(?:executable|library)\s*\(\s*(\w+)\s+(.*?)\)",
        re.DOTALL,
    )
    # Source file extensions we care about
    src_ext_re = re.compile(r"\.(cpp|c|cxx|cc|m|mm|s|S)$")

    for cml in sorted(repo_path.rglob("CMakeLists.txt")):
        try:
            text = cml.read_text(errors="replace")
        except OSError:
            continue

        rel_cml = str(cml.relative_to(repo_path))
        cml_dir = cml.parent

        for m in target_re.finditer(text):
            target_name = m.group(1)
            args_str = m.group(2)

            # Remove comments
            args_str = re.sub(r"#.*$", "", args_str, flags=re.MULTILINE)
            # Remove keywords
            args_str = re.sub(
                r"\b(STATIC|SHARED|MODULE|INTERFACE|OBJECT|IMPORTED|ALIAS|"
                r"WIN32|MACOSX_BUNDLE|EXCLUDE_FROM_ALL)\b",
                "", args_str,
            )

            # Extract source file tokens
            tokens = args_str.split()
            for tok in tokens:
                tok = tok.strip()
                if not tok or tok.startswith("$"):
                    continue
                if src_ext_re.search(tok):
                    # Resolve relative to the CMakeLists.txt directory
                    rel_src = str((cml_dir / tok).relative_to(repo_path)) \
                        if not tok.startswith("/") else tok
                    result.edges.append(Edge(
                        src=target_name, dst=rel_src, kind="target_source"
                    ))
                    # Add source node if not present
                    if not any(n.id == rel_src for n in result.nodes):
                        result.nodes.append(Node(
                            id=rel_src, type="source", file=rel_cml
                        ))


def _parse_dot_file(dot_path: Path, result: GraphResult) -> None:
    """Parse a Graphviz DOT file produced by cmake --graphviz."""
    try:
        text = dot_path.read_text(errors="replace")
    except OSError:
        return
    _parse_dot_text(text, result)


def _parse_dot_text(text: str, result: GraphResult) -> None:
    """Parse DOT graph text into nodes and edges."""
    # CMake graphviz format:
    #   "node0" [ label = "core_lib", shape = octagon ];
    #   "node2" -> "node0" [ style = dotted ] // myapp -> core_lib
    node_re = re.compile(
        r'"([^"]+)"\s*\[([^\]]*)\]', re.MULTILINE
    )
    edge_re = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"')

    # Shape → type mapping (cmake graphviz conventions)
    shape_types = {
        "egg": "executable",
        "octagon": "static_library",
        "doubleoctagon": "shared_library",
        "tripleoctagon": "module_library",
        "pentagon": "interface_library",
        "hexagon": "object_library",
        "septagon": "unknown_library",
        "box": "custom_target",
    }

    # Build node ID → label mapping
    id_to_label: dict[str, str] = {}
    seen_nodes: set[str] = set()

    for m in node_re.finditer(text):
        node_id = m.group(1)
        attrs = m.group(2)

        # Skip legend nodes
        if node_id.startswith("legendNode"):
            continue

        # Only process first occurrence of a node ID (edge lines also match
        # the pattern, e.g. "node3" [ style = dotted ], but lack a label)
        if node_id in id_to_label:
            continue

        # Extract label
        label_m = re.search(r'label\s*=\s*"([^"]*)"', attrs)
        label = label_m.group(1) if label_m else node_id

        # Skip legend labels
        if label in ("Legend", "Executable", "Static Library", "Shared Library",
                     "Module Library", "Interface Library", "Object Library",
                     "Unknown Library", "Custom Target"):
            continue

        id_to_label[node_id] = label

        if label in seen_nodes:
            continue
        seen_nodes.add(label)

        # Determine type from shape
        node_type = "unknown"
        shape_m = re.search(r'shape\s*=\s*(\w+)', attrs)
        if shape_m:
            shape = shape_m.group(1)
            node_type = shape_types.get(shape, "target")

        # Fallback heuristics from label
        if node_type == "unknown":
            if re.search(r"\.(cpp|c|cxx|cc|h|hpp)$", label):
                node_type = "source"
            elif "lib" in label.lower():
                node_type = "library"

        result.nodes.append(Node(id=label, type=node_type))

    # Parse edges, resolving node IDs to labels
    seen_edges: set[tuple[str, str]] = set()
    for m in edge_re.finditer(text):
        src_id, dst_id = m.group(1), m.group(2)

        # Skip legend edges
        if src_id.startswith("legendNode") or dst_id.startswith("legendNode"):
            continue

        src = id_to_label.get(src_id, src_id)
        dst = id_to_label.get(dst_id, dst_id)

        edge_key = (src, dst)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        # Classify edge kind
        kind = "link_dep"
        if re.search(r"\.(cpp|c|cxx|cc|h|hpp|txt)$", dst):
            kind = "target_source"
        elif re.search(r"(pthread|Threads)", dst):
            kind = "link_dep"

        result.edges.append(Edge(src=src, dst=dst, kind=kind))


# ---------------------------------------------------------------------------
# Make extractor
# ---------------------------------------------------------------------------


def extract_make(repo_path: Path, repo_name: str, tmp_dir: str) -> GraphResult:
    """Run make -p -n to dump the rule database without executing."""
    result = GraphResult(repo=repo_name, build_system="make")

    # Find the makefile
    makefile = None
    for name in ("GNUmakefile", "makefile", "Makefile"):
        candidate = repo_path / name
        if candidate.exists():
            makefile = name
            break

    if makefile is None:
        result.diagnostics.append(
            Diagnostic(file="", line=0, severity="error", message="No Makefile found")
        )
        return result

    result.graph_file = makefile

    # make -p -n: print database, no execution
    cmd = ["make", "-p", "-n", "-f", makefile]
    rc, stdout, stderr = run_cmd(cmd, cwd=str(repo_path))

    if rc == -2:
        result.diagnostics.append(
            Diagnostic(file="", line=0, severity="error", message="make not installed")
        )
        return result

    # make -p may return non-zero (e.g. rc=2) if it can't satisfy the build,
    # but the database is still printed to stdout.  Consider it a success if
    # we got rule data.
    result.configure_success = rc == 0 or (stdout and ":" in stdout)

    # Parse rule lines: target: prerequisites
    # In make -p output, rules appear as lines like:
    #   target: dep1 dep2 dep3
    # Skip comments, variables, recipe lines (start with tab), and pattern rules.
    rule_re = re.compile(r"^([^#\t][^:=]*?)\s*:\s*([^=].*?)?$")
    seen_edges: set[tuple[str, str]] = set()

    # Make internal variables to skip (these appear as "VAR = value" or "VAR := value")
    make_internals = {
        "MAKEFILE_LIST", "CURDIR", "MAKEFILEPATH", "MAKE_VERSION",
        "MAKE_COMMAND", "MAKEFILES", "SUFFIXES", "MAKELEVEL",
        "MAKE", "MAKEFLAGS", "MAKEOVERRIDES", "MFLAGS",
        ".DEFAULT_GOAL", ".FEATURES", ".INCLUDE_DIRS", ".VARIABLES",
        ".RECIPEPREFIX", ".EXTRA_PREREQS",
    }

    for line in stdout.splitlines():
        # Skip recipe lines (start with tab), comments
        if line.startswith("\t") or line.startswith("#"):
            continue

        # Skip variable assignments: VAR = val, VAR := val, VAR ?= val, VAR += val
        # A rule line has ":" NOT followed by "=" (i.e., not ":=")
        if re.match(r"^[A-Za-z_.][\w.]*\s*[:?+]?=", line):
            continue
        # Also skip lines like "VAR := ..." where colon is part of assignment
        if re.match(r"^[^:]+:= ", line) or re.match(r"^[^:]+\?= ", line):
            continue

        m = rule_re.match(line)
        if not m:
            continue

        target = m.group(1).strip()
        prereqs_str = (m.group(2) or "").strip()

        # Skip pattern rules (contain %)
        if "%" in target:
            continue

        # Skip make internal targets and special targets
        if target.startswith(".") and target in (
            ".PHONY", ".SUFFIXES", ".DEFAULT", ".PRECIOUS",
            ".INTERMEDIATE", ".SECONDARY", ".SECONDEXPANSION",
            ".DELETE_ON_ERROR", ".IGNORE", ".LOW_RESOLUTION_TIME",
            ".SILENT", ".EXPORT_ALL_VARIABLES", ".NOTPARALLEL",
            ".ONESHELL", ".POSIX",
        ):
            continue

        # Skip make internal variables that look like rules
        if target in make_internals:
            continue

        # Skip other dot-targets and archive syntax
        if target.startswith(".") or target.startswith("("):
            continue

        # Skip targets that are just ":" or empty
        if not target or target == ":":
            continue

        # Add target as node
        if not any(n.id == target for n in result.nodes):
            ntype = "target"
            if re.search(r"\.(o|obj)$", target):
                ntype = "object"
            elif re.search(r"\.(a|so|dylib)$", target):
                ntype = "library"
            elif re.search(r"\.(c|cpp|cxx|cc|h|hpp)$", target):
                ntype = "source"
            elif not re.search(r"\.", target):
                ntype = "executable"
            result.nodes.append(Node(id=target, type=ntype))

        # Parse prerequisites
        if prereqs_str:
            # Remove order-only prereqs (after |)
            prereqs_str = prereqs_str.split("|")[0].strip()
            prereqs = prereqs_str.split()
            for prereq in prereqs:
                prereq = prereq.strip()
                if not prereq or prereq.startswith(".") or "%" in prereq:
                    continue
                # Skip RCS/SCCS built-in prereqs
                if prereq.startswith(("RCS/", "SCCS/", "s.")):
                    continue
                edge_key = (target, prereq)
                if edge_key not in seen_edges:
                    seen_edges.add(edge_key)
                    kind = "depends_on"
                    if re.search(r"\.(c|cpp|cxx|cc|s|S)$", prereq):
                        kind = "target_source"
                    elif re.search(r"\.(h|hpp|hxx)$", prereq):
                        kind = "header_dep"
                    elif re.search(r"\.(o|obj)$", prereq):
                        kind = "object_dep"
                    elif re.search(r"\.(a|so|dylib)$", prereq) or prereq.startswith("-l"):
                        kind = "link_dep"
                    result.edges.append(Edge(src=target, dst=prereq, kind=kind))

    # Find included makefiles
    include_re = re.compile(r"^include\s+(.+)$|^sinclude\s+(.+)$", re.MULTILINE)
    makefile_text = (repo_path / makefile).read_text(errors="replace")
    for m in include_re.finditer(makefile_text):
        inc = m.group(1) or m.group(2)
        for part in inc.strip().split():
            if part and not part.startswith("$"):
                result.cross_file_includes.append(part)

    return result


# ---------------------------------------------------------------------------
# Bazel extractor
# ---------------------------------------------------------------------------


def extract_bazel(repo_path: Path, repo_name: str, tmp_dir: str) -> GraphResult:
    """Run bazel query or fall back to static BUILD file parsing."""
    result = GraphResult(repo=repo_name, build_system="bazel")

    if tool_available("bazel"):
        cmd = ["bazel", "query", "deps(//...)", "--output=graph"]
        rc, stdout, stderr = run_cmd(cmd, cwd=str(repo_path))
        result.configure_success = rc == 0
        if rc == 0 and stdout.strip():
            result.graph_file = "bazel_query.dot"
            _parse_dot_file_content(stdout, result)
            return result

    # Fallback: static parse of BUILD / BUILD.bazel files
    result.configure_success = True  # static parse always "succeeds"
    _parse_bazel_build_files(repo_path, result)
    return result


def _parse_dot_file_content(text: str, result: GraphResult) -> None:
    """Parse DOT content from a string (used by bazel/ninja query output)."""
    _parse_dot_text(text, result)


def _parse_bazel_build_files(repo_path: Path, result: GraphResult) -> None:
    """Statically parse BUILD/BUILD.bazel files for rule→srcs, rule→deps."""
    build_files = list(repo_path.rglob("BUILD")) + list(repo_path.rglob("BUILD.bazel"))
    build_files = sorted(set(build_files))

    # Simple regex-based extraction of rule attributes
    rule_re = re.compile(
        r"(\w+)\s*\(\s*name\s*=\s*\"([^\"]+)\"", re.MULTILINE
    )
    srcs_re = re.compile(r"srcs\s*=\s*\[([^\]]*)\]", re.DOTALL)
    deps_re = re.compile(r"deps\s*=\s*\[([^\]]*)\]", re.DOTALL)
    string_re = re.compile(r'"([^"]+)"')

    for bf in build_files:
        rel_bf = str(bf.relative_to(repo_path))
        result.cross_file_includes.append(rel_bf)
        try:
            text = bf.read_text(errors="replace")
        except OSError:
            continue

        # Split into rule blocks (rough heuristic: split on rule_name()
        for rm in rule_re.finditer(text):
            rule_type = rm.group(1)
            rule_name = rm.group(2)
            # Get the block after the rule opening (up to next rule or EOF)
            block_start = rm.end()
            # Find matching closing paren (simplified: next rule or 2000 chars)
            next_rule = rule_re.search(text, block_start)
            block_end = next_rule.start() if next_rule else len(text)
            block = text[block_start:block_end]

            full_name = f"//{bf.parent.relative_to(repo_path)}:{rule_name}" if bf.parent != repo_path else f"//:{rule_name}"
            result.nodes.append(Node(
                id=full_name,
                type=rule_type,
                file=f"{rel_bf}",
            ))

            # Extract srcs
            sm = srcs_re.search(block)
            if sm:
                for s in string_re.finditer(sm.group(1)):
                    result.edges.append(Edge(
                        src=full_name, dst=s.group(1), kind="target_source"
                    ))

            # Extract deps
            dm = deps_re.search(block)
            if dm:
                for d in string_re.finditer(dm.group(1)):
                    result.edges.append(Edge(
                        src=full_name, dst=d.group(1), kind="link_dep"
                    ))


# ---------------------------------------------------------------------------
# Ninja extractor
# ---------------------------------------------------------------------------


def extract_ninja(repo_path: Path, repo_name: str, tmp_dir: str) -> GraphResult:
    """Run ninja -t graph or statically parse build.ninja."""
    result = GraphResult(repo=repo_name, build_system="ninja")
    ninja_file = repo_path / "build.ninja"
    result.graph_file = "build.ninja"

    if tool_available("ninja"):
        cmd = ["ninja", "-t", "graph", "-f", str(ninja_file)]
        rc, stdout, stderr = run_cmd(cmd, cwd=str(repo_path))
        if rc == 0 and stdout.strip():
            result.configure_success = True
            _parse_dot_file_content(stdout, result)
            return result

    # Fallback: static parse of build.ninja
    result.configure_success = True
    _parse_ninja_file(ninja_file, result)
    return result


def _parse_ninja_file(ninja_file: Path, result: GraphResult) -> None:
    """Statically parse build.ninja for build edges."""
    try:
        text = ninja_file.read_text(errors="replace")
    except OSError:
        result.diagnostics.append(
            Diagnostic(file="build.ninja", line=0, severity="error",
                       message="Cannot read build.ninja")
        )
        return

    # build statements: build output1 output2: rule input1 input2 | implicit || order
    build_re = re.compile(
        r"^build\s+([^:]+):\s+(\S+)\s+(.*)$", re.MULTILINE
    )

    for m in build_re.finditer(text):
        outputs_str = m.group(1).strip()
        rule = m.group(2).strip()
        inputs_str = m.group(3).strip()

        outputs = outputs_str.split()
        # Split inputs on | and || for implicit/order-only
        inputs_parts = re.split(r"\|\|?", inputs_str)
        inputs = inputs_parts[0].split() if inputs_parts else []

        for out in outputs:
            out = out.strip()
            if not out:
                continue
            result.nodes.append(Node(id=out, type="output"))
            for inp in inputs:
                inp = inp.strip()
                if not inp:
                    continue
                kind = "target_source" if rule in ("cc", "cxx", "compile") else "depends_on"
                result.edges.append(Edge(src=out, dst=inp, kind=kind))


# ---------------------------------------------------------------------------
# Shell extractor
# ---------------------------------------------------------------------------


def extract_shell(repo_path: Path, repo_name: str, tmp_dir: str) -> GraphResult:
    """Extract cross-file source dependencies from shell scripts."""
    result = GraphResult(repo=repo_name, build_system="shell")
    result.configure_success = True

    # Recursively find all shell scripts in the repo
    sh_files = sorted(set(repo_path.rglob("*.sh")))

    if not sh_files:
        return result

    if tool_available("shellcheck"):
        _extract_shell_shellcheck(sh_files, repo_path, result)
    else:
        _extract_shell_manual(sh_files, repo_path, result)

    return result


def _extract_shell_shellcheck(
    sh_files: list[Path], repo_path: Path, result: GraphResult
) -> None:
    """Use shellcheck -x --format=json for cross-file source resolution."""
    for sh_file in sh_files:
        rel = str(sh_file.relative_to(repo_path))
        result.nodes.append(Node(id=rel, type="script", file=rel))

        cmd = ["shellcheck", "-x", "--format=json", str(sh_file)]
        rc, stdout, stderr = run_cmd(cmd, cwd=str(repo_path))

        if rc in (0, 1) and stdout.strip():
            try:
                findings = json.loads(stdout)
                for f in findings:
                    if isinstance(f, dict):
                        result.diagnostics.append(Diagnostic(
                            file=f.get("file", rel),
                            line=f.get("line", 0),
                            severity=f.get("level", "info"),
                            message=f.get("message", ""),
                        ))
            except json.JSONDecodeError:
                pass

        # Also manually resolve source statements
        _resolve_shell_sources(sh_file, repo_path, result)


def _extract_shell_manual(
    sh_files: list[Path], repo_path: Path, result: GraphResult
) -> None:
    """Manually parse shell scripts for source/. statements."""
    for sh_file in sh_files:
        rel = str(sh_file.relative_to(repo_path))
        result.nodes.append(Node(id=rel, type="script", file=rel))
        _resolve_shell_sources(sh_file, repo_path, result)


def _resolve_shell_sources(
    sh_file: Path, repo_path: Path, result: GraphResult
) -> None:
    """Follow source/. statements in a shell script."""
    source_re = re.compile(
        r"""^\s*(?:source|\.)\s+["']?([^"'\s;]+)["']?""", re.MULTILINE
    )
    try:
        text = sh_file.read_text(errors="replace")
    except OSError:
        return

    rel_script = str(sh_file.relative_to(repo_path))

    for m in source_re.finditer(text):
        sourced = m.group(1)
        # Skip variable expansions
        if "$" in sourced:
            continue

        # Try resolving: first relative to repo root (common pattern when
        # scripts are invoked from the project root), then relative to script dir
        candidates = [
            (repo_path / sourced).resolve(),
            (sh_file.parent / sourced).resolve(),
        ]
        resolved = None
        for candidate in candidates:
            if candidate.exists():
                resolved = candidate
                break
        if resolved is None:
            # Use repo-root-relative as default even if file doesn't exist
            resolved = candidates[0]

        try:
            rel_resolved = str(resolved.relative_to(repo_path))
        except ValueError:
            rel_resolved = sourced

        result.edges.append(Edge(src=rel_script, dst=rel_resolved, kind="source_dep"))
        if rel_resolved not in result.cross_file_includes:
            result.cross_file_includes.append(rel_resolved)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

EXTRACTORS = {
    "cmake": extract_cmake,
    "make": extract_make,
    "bazel": extract_bazel,
    "ninja": extract_ninja,
    "shell": extract_shell,
}


def process_repo(repo_path: Path, repo_name: str, output_dir: Path) -> GraphResult | None:
    """Process a single repo: detect build system, extract graph, write JSON."""
    build_system = detect_build_system(repo_path)
    if build_system is None:
        print(f"  [SKIP] {repo_name}: no supported build system detected")
        return None

    extractor = EXTRACTORS.get(build_system)
    if extractor is None:
        print(f"  [SKIP] {repo_name}: no extractor for {build_system}")
        return None

    # Check tool availability for systems that need external tools
    if build_system == "cmake" and not tool_available("cmake"):
        print(f"  [SKIP] {repo_name}: cmake not installed")
        return None
    if build_system == "make" and not tool_available("make"):
        print(f"  [SKIP] {repo_name}: make not installed")
        return None

    print(f"  [{build_system.upper()}] {repo_name}")

    # Create temp dir for this repo
    tmp_dir = tempfile.mkdtemp(prefix=f"buildgraph_{repo_name.replace('/', '_')}_")
    try:
        result = extractor(repo_path, repo_name, tmp_dir)
    except Exception as exc:
        result = GraphResult(repo=repo_name, build_system=build_system)
        result.diagnostics.append(
            Diagnostic(file="", line=0, severity="error", message=f"Extractor crash: {exc}")
        )
    finally:
        # Always clean up temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = repo_name.replace("/", "_").replace("\\", "_")
    out_path = output_dir / f"{safe_name}.json"
    out_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n")
    print(f"    -> {out_path}  ({len(result.nodes)} nodes, {len(result.edges)} edges)")

    return result


def discover_repos(repos_dir: Path) -> list[tuple[str, Path]]:
    """Discover repos in the repos directory.  Each subdirectory is a repo."""
    repos = []
    if not repos_dir.is_dir():
        return repos
    for entry in sorted(repos_dir.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            repos.append((entry.name, entry))
    return repos


def extract_from_tar(
    tar_path: Path, repo_name: str, output_dir: Path
) -> GraphResult | None:
    """Extract a repo from a tarball, run graph extraction, delete source."""
    extract_tmp = tempfile.mkdtemp(prefix=f"tartmp_{repo_name.replace('/', '_')}_")
    try:
        # Extract tarball
        with tarfile.open(tar_path, "r:*") as tf:
            tf.extractall(extract_tmp, filter="data")

        # Find the repo root (might be nested one level)
        entries = list(Path(extract_tmp).iterdir())
        if len(entries) == 1 and entries[0].is_dir():
            repo_path = entries[0]
        else:
            repo_path = Path(extract_tmp)

        return process_repo(repo_path, repo_name, output_dir)
    finally:
        shutil.rmtree(extract_tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract cross-file dependency graphs from repos (DRY RUN only)."
    )
    parser.add_argument(
        "--repos-dir",
        type=Path,
        default=Path(DEFAULT_REPOS_DIR),
        help=f"Directory containing repo sources (default: {DEFAULT_REPOS_DIR})",
    )
    parser.add_argument(
        "--repo-list",
        type=Path,
        default=None,
        help="JSON file with list of repo names to process",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N repos",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just list what would be processed, don't run extraction",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(OUTPUT_DIR),
        help=f"Output directory for graph JSON files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--from-tar",
        type=Path,
        default=None,
        metavar="TARBALL",
        help="Extract repo from tarball, run extraction, delete source",
    )
    parser.add_argument(
        "--tar-repo-name",
        type=str,
        default=None,
        help="Repo name when using --from-tar (default: tarball filename)",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).resolve().parent.parent
    repos_dir = args.repos_dir
    if not repos_dir.is_absolute():
        repos_dir = project_root / repos_dir
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    # --from-tar mode
    if args.from_tar:
        tar_path = args.from_tar
        if not tar_path.is_absolute():
            tar_path = project_root / tar_path
        if not tar_path.exists():
            print(f"ERROR: tarball not found: {tar_path}", file=sys.stderr)
            sys.exit(1)
        repo_name = args.tar_repo_name or tar_path.stem.replace(".tar", "")
        print(f"Extracting from tarball: {tar_path}")
        result = extract_from_tar(tar_path, repo_name, output_dir)
        if result:
            print(f"Done: {len(result.nodes)} nodes, {len(result.edges)} edges")
        return

    # Discover repos
    if args.repo_list:
        repo_list_path = args.repo_list
        if not repo_list_path.is_absolute():
            repo_list_path = project_root / repo_list_path
        repo_names = json.loads(repo_list_path.read_text())
        repos = [(name, repos_dir / name) for name in repo_names if (repos_dir / name).is_dir()]
    else:
        repos = discover_repos(repos_dir)

    if args.limit:
        repos = repos[: args.limit]

    if not repos:
        print(f"No repos found in {repos_dir}")
        print("Hint: use --repos-dir to point to a directory with repo subdirectories")
        sys.exit(0)

    # --dry-run: just list
    if args.dry_run:
        print(f"Would process {len(repos)} repo(s) from {repos_dir}:")
        for name, path in repos:
            bs = detect_build_system(path)
            print(f"  {name:40s}  build_system={bs or 'none'}")
        return

    # Process repos one at a time (streaming)
    print(f"Processing {len(repos)} repo(s) from {repos_dir}")
    print(f"Output: {output_dir}")
    print()

    stats = {"processed": 0, "skipped": 0, "failed": 0}
    for name, path in repos:
        try:
            result = process_repo(path, name, output_dir)
            if result is None:
                stats["skipped"] += 1
            else:
                stats["processed"] += 1
        except Exception as exc:
            print(f"  [ERROR] {name}: {exc}")
            stats["failed"] += 1

    print()
    print(f"Done: {stats['processed']} processed, {stats['skipped']} skipped, {stats['failed']} failed")


if __name__ == "__main__":
    main()
