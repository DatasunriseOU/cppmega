"""Shell domain parser using tree-sitter AST (bash/zsh/powershell/batch)."""

from __future__ import annotations

from typing import Any

from cppmega.data.build_parsers.base import (
    ParsedDomainDocument,
    is_source_path,
    strip_quotes,
)
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)

_SHELL_LANGUAGE_MAP = {
    "bash": "bash",
    "sh": "bash",
    "ksh": "bash",
    "zsh": "zsh",
    "tcsh": "tcsh",
    "csh": "tcsh",
    "powershell": "powershell",
    "pwsh": "powershell",
    "ps1": "powershell",
    "batch": "batch",
    "cmd": "batch",
}

_SHELL_DOMAIN_MAP = {
    "bash": DomainKind.BASH,
    "sh": DomainKind.SH,
    "ksh": DomainKind.KSH,
    "zsh": DomainKind.ZSH,
    "tcsh": DomainKind.TCSH,
    "csh": DomainKind.TCSH,
    "powershell": DomainKind.POWERSHELL,
    "pwsh": DomainKind.POWERSHELL,
    "ps1": DomainKind.POWERSHELL,
    "batch": DomainKind.BATCH,
    "cmd": DomainKind.BATCH,
}

_PIPE_NODE_TYPES = {"pipe", "pipeline"}
_REDIR_OUT_TYPES = {"file_redirect", "redirect", "redirection", "out_redirect"}
_REDIR_IN_TYPES = {"in_redirect"}
_SOURCE_COMMANDS = {"source", "."}
_VAR_ASSIGN_TYPES = {"variable_assignment", "assignment"}
_VAR_EXPANSION_TYPES = {
    "variable_name",
    "expansion",
    "simple_expansion",
    "variable",
}

_FILE_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx", ".cu", ".cuh",
    ".s", ".asm", ".o", ".a", ".so", ".dylib", ".dll", ".exe",
    ".py", ".sh", ".bash", ".zsh", ".txt", ".md", ".json", ".yaml", ".yml",
    ".cmake", ".mk", ".ninja", ".build", ".bzl", ".meson", ".toml", ".cfg",
    ".conf", ".ini", ".log", ".out", ".err", ".dat", ".bin", ".tar", ".gz",
    ".zip", ".xz", ".bz2", ".zst", ".parquet", ".csv", ".tsv",
}


def _looks_like_file(text: str) -> bool:
    if "/" in text or "\\" in text:
        return True
    dot_idx = text.rfind(".")
    if dot_idx > 0 and dot_idx < len(text) - 1:
        ext = text[dot_idx:].lower()
        if ext in _FILE_EXTENSIONS or len(ext) <= 5:
            return True
    return False


def _get_parser(shell_kind: str):
    lang = _SHELL_LANGUAGE_MAP.get(shell_kind, "bash")
    if lang == "tcsh":
        try:
            return _get_tcsh_parser()
        except Exception:
            pass
    from tree_sitter_language_pack import get_parser

    return get_parser("bash" if lang == "tcsh" else lang)


def _get_tcsh_parser():
    """Load the bundled tree-sitter-tcsh grammar."""
    import ctypes
    import warnings
    from pathlib import Path

    import tree_sitter

    so_path = Path(__file__).resolve().parent.parent / "grammars" / "tree_sitter_tcsh.so"
    if not so_path.exists():
        raise FileNotFoundError(f"tcsh grammar not found: {so_path}")
    lib = ctypes.cdll.LoadLibrary(str(so_path))
    func = lib.tree_sitter_tcsh
    func.restype = ctypes.c_void_p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lang = tree_sitter.Language(func())
    return tree_sitter.Parser(lang)


def _node_text(node: Any, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _walk(node: Any):
    yield node
    for child in node.children:
        yield from _walk(child)


def _find_children_of_type(node: Any, types: set[str]) -> list[Any]:
    return [c for c in node.children if c.type in types]


def _extract_bash_edges(root: Any, source: bytes) -> list[tuple[int, int, int]]:
    """Extract shell edges from bash/zsh AST."""
    edges: list[tuple[int, int, int]] = []
    var_defs: dict[str, list[int]] = {}
    var_uses: list[tuple[str, int]] = []

    for node in _walk(root):
        if node.type in _PIPE_NODE_TYPES:
            cmds = [
                c
                for c in node.children
                if c.type in {"command", "subshell", "pipeline", "compound_statement"}
                or c.type.endswith("_command")
            ]
            for i in range(len(cmds) - 1):
                edges.append(
                    (cmds[i].start_byte, cmds[i + 1].start_byte, int(DomainEdgeKind.SHELL_PIPE))
                )

        elif node.type in _VAR_ASSIGN_TYPES:
            name_node = None
            for c in node.children:
                if c.type == "variable_name":
                    name_node = c
                    break
            if name_node is not None:
                name = _node_text(name_node, source)
                var_defs.setdefault(name, []).append(name_node.start_byte)

        elif node.type in _VAR_EXPANSION_TYPES:
            name = _node_text(node, source).lstrip("$").strip("{}")
            if name and name.isidentifier():
                var_uses.append((name, node.start_byte))

        elif node.type == "command":
            words = _find_children_of_type(node, {"command_name", "word", "string"})
            if words:
                cmd_name = _node_text(words[0], source)
                if cmd_name in _SOURCE_COMMANDS and len(words) > 1:
                    target = _node_text(words[1], source)
                    edges.append(
                        (
                            words[0].start_byte,
                            words[1].start_byte,
                            int(DomainEdgeKind.SHELL_COMMAND_FILE),
                        )
                    )
                else:
                    for w in words[1:]:
                        text = strip_quotes(_node_text(w, source))
                        if _looks_like_file(text):
                            edges.append(
                                (
                                    words[0].start_byte,
                                    w.start_byte,
                                    int(DomainEdgeKind.SHELL_COMMAND_FILE),
                                )
                            )

        elif node.type in _REDIR_OUT_TYPES:
            children = node.children
            if len(children) >= 2:
                edges.append(
                    (
                        children[0].start_byte,
                        children[-1].start_byte,
                        int(DomainEdgeKind.SHELL_REDIR_OUT),
                    )
                )

        elif node.type in _REDIR_IN_TYPES:
            children = node.children
            if len(children) >= 2:
                edges.append(
                    (
                        children[-1].start_byte,
                        children[0].start_byte,
                        int(DomainEdgeKind.SHELL_REDIR_IN),
                    )
                )

    for name, use_offset in var_uses:
        if name in var_defs:
            def_offset = var_defs[name][0]
            edges.append(
                (def_offset, use_offset, int(DomainEdgeKind.SHELL_VAR_DEF_USE))
            )

    return edges


def _extract_powershell_edges(root: Any, source: bytes) -> list[tuple[int, int, int]]:
    """Extract shell edges from PowerShell AST."""
    edges: list[tuple[int, int, int]] = []
    var_defs: dict[str, list[int]] = {}
    var_uses: list[tuple[str, int]] = []
    assignment_vars: set[int] = set()

    for node in _walk(root):
        if node.type == "pipeline_chain":
            parts = [
                c
                for c in node.children
                if c.type not in {"|", "command_argument_sep"}
                and not c.type.startswith("_")
            ]
            for i in range(len(parts) - 1):
                edges.append(
                    (parts[i].start_byte, parts[i + 1].start_byte, int(DomainEdgeKind.SHELL_PIPE))
                )

        elif node.type == "assignment_expression":
            for child in node.children:
                if child.type == "left_assignment_expression":
                    for var_node in _walk(child):
                        if var_node.type == "variable":
                            name = _node_text(var_node, source).lstrip("$")
                            var_defs.setdefault(name, []).append(var_node.start_byte)
                            assignment_vars.add(id(var_node))
                            break
                    break

        elif node.type == "variable":
            if id(node) not in assignment_vars:
                name = _node_text(node, source).lstrip("$")
                if name and name[0:1].isalpha():
                    var_uses.append((name, node.start_byte))

        elif node.type == "command":
            cmd_name_node = None
            arg_nodes: list[Any] = []
            for c in node.children:
                if c.type == "command_name":
                    cmd_name_node = c
                elif c.type == "command_elements":
                    for gc in c.children:
                        if gc.type == "generic_token":
                            arg_nodes.append(gc)
            if cmd_name_node is not None:
                cmd_name = _node_text(cmd_name_node, source)
                if cmd_name in {".", "Invoke-Expression", "iex"} and arg_nodes:
                    edges.append(
                        (
                            cmd_name_node.start_byte,
                            arg_nodes[0].start_byte,
                            int(DomainEdgeKind.SHELL_COMMAND_FILE),
                        )
                    )
                else:
                    for arg in arg_nodes:
                        text = strip_quotes(_node_text(arg, source))
                        if is_source_path(text) or "/" in text or "\\" in text:
                            edges.append(
                                (
                                    cmd_name_node.start_byte,
                                    arg.start_byte,
                                    int(DomainEdgeKind.SHELL_COMMAND_FILE),
                                )
                            )

    for name, use_offset in var_uses:
        if name in var_defs:
            def_offset = var_defs[name][0]
            edges.append(
                (def_offset, use_offset, int(DomainEdgeKind.SHELL_VAR_DEF_USE))
            )

    return edges


def _extract_batch_edges(root: Any, source: bytes) -> list[tuple[int, int, int]]:
    """Extract shell edges from Windows batch/cmd AST (limited grammar)."""
    edges: list[tuple[int, int, int]] = []
    var_defs: dict[str, int] = {}
    var_uses: list[tuple[str, int]] = []

    for node in _walk(root):
        if node.type == "variable_reference":
            name_node = None
            for c in node.children:
                if c.type == "variable_name":
                    name_node = c
                    break
            if name_node is not None:
                name = _node_text(name_node, source).upper()
                var_uses.append((name, node.start_byte))

        elif node.type == "keyword":
            kw_text = _node_text(node, source).lower()
            if kw_text == "set":
                next_sib = node.next_sibling
                if next_sib is not None:
                    for c in _walk(next_sib):
                        if c.type == "identifier":
                            name = _node_text(c, source).upper()
                            var_defs.setdefault(name, c.start_byte)
                            break

        elif node.type == "pipe":
            children = [c for c in node.children if c.type not in {"|", "ERROR"}]
            for i in range(len(children) - 1):
                edges.append(
                    (children[i].start_byte, children[i + 1].start_byte, int(DomainEdgeKind.SHELL_PIPE))
                )

    for name, use_offset in var_uses:
        if name in var_defs:
            edges.append(
                (var_defs[name], use_offset, int(DomainEdgeKind.SHELL_VAR_DEF_USE))
            )

    return edges


def _extract_tcsh_edges(root: Any, source: bytes) -> list[tuple[int, int, int]]:
    """Extract shell edges from native tcsh AST."""
    edges: list[tuple[int, int, int]] = []
    var_defs: dict[str, int] = {}
    var_uses: list[tuple[str, int]] = []

    for node in _walk(root):
        if node.type == "pipeline":
            cmds = [c for c in node.children if c.type == "command"]
            for i in range(len(cmds) - 1):
                edges.append(
                    (cmds[i].start_byte, cmds[i + 1].start_byte, int(DomainEdgeKind.SHELL_PIPE))
                )

        elif node.type == "redirection":
            op_node = None
            target_node = None
            for c in node.children:
                if c.type == "redirect_operator":
                    op_node = c
                elif c.type == "word":
                    target_node = c
            if op_node is not None and target_node is not None:
                op_text = _node_text(op_node, source)
                if "<" in op_text:
                    edges.append(
                        (target_node.start_byte, op_node.start_byte, int(DomainEdgeKind.SHELL_REDIR_IN))
                    )
                else:
                    edges.append(
                        (op_node.start_byte, target_node.start_byte, int(DomainEdgeKind.SHELL_REDIR_OUT))
                    )

        elif node.type == "simple_command":
            words = [c for c in node.children if c.type == "word"]
            if words:
                cmd_word = words[0]
                for w in words[1:]:
                    text = _node_text(w, source)
                    if _looks_like_file(text):
                        edges.append(
                            (cmd_word.start_byte, w.start_byte, int(DomainEdgeKind.SHELL_COMMAND_FILE))
                        )

        elif node.type == "setenv_command":
            parent = node.parent
            if parent is not None:
                siblings = parent.children
                found_cmd = False
                for sib in siblings:
                    if sib == node:
                        found_cmd = True
                        continue
                    if found_cmd and sib.type == "word":
                        for c in sib.children:
                            if c.type == "identifier":
                                name = _node_text(c, source)
                                var_defs.setdefault(name, c.start_byte)
                                break
                        break

        elif node.type == "variable_substitution":
            for c in node.children:
                if c.type == "identifier":
                    name = _node_text(c, source)
                    var_uses.append((name, node.start_byte))
                    break

    for name, use_offset in var_uses:
        if name in var_defs:
            edges.append(
                (var_defs[name], use_offset, int(DomainEdgeKind.SHELL_VAR_DEF_USE))
            )

    return edges


def _heuristic_assign_roles_and_edges(doc: ParsedDomainDocument, shell_kind: str) -> None:
    """Token-level heuristic role/edge assignment when tree-sitter is unavailable."""
    tokens = doc.tokens
    if not tokens:
        return

    keywords = _DIALECT_KEYWORDS.get(shell_kind, _SHELL_KEYWORDS)
    redir_out = {">", ">>", "2>", "&>"}
    redir_in = {"<"}

    next_entity = 1
    previous_command: int | None = None
    command_expected = True
    pending_redir: tuple[int, DomainEdgeKind] | None = None

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if idx > 0 and token.line != tokens[idx - 1].line:
            previous_command = None
            command_expected = True
            pending_redir = None
        value = token.text

        if value in keywords:
            doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
            next_entity += 1
            command_expected = False
            idx += 1
            continue

        if value == "|":
            doc.set_role(idx, DomainRoleKind.PIPE)
            command_expected = True
            pending_redir = None
            idx += 1
            continue

        if value in redir_out | redir_in:
            edge_kind = (
                DomainEdgeKind.SHELL_REDIR_OUT
                if value in redir_out
                else DomainEdgeKind.SHELL_REDIR_IN
            )
            doc.set_role(idx, DomainRoleKind.REDIRECT)
            pending_redir = (previous_command if previous_command is not None else idx, edge_kind)
            idx += 1
            continue

        if value.startswith("$"):
            doc.set_role(idx, DomainRoleKind.VARIABLE)
            idx += 1
            continue

        if pending_redir is not None:
            src, edge_kind = pending_redir
            doc.set_role(idx, DomainRoleKind.PATH, entity=next_entity)
            next_entity += 1
            doc.add_edge(src, idx, edge_kind)
            pending_redir = None
            idx += 1
            continue

        if command_expected and value not in {";", "&&", "||"}:
            old_command = previous_command
            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
            previous_command = idx
            next_entity += 1
            if old_command is not None and any(
                tokens[j].text == "|" for j in range(max(0, old_command + 1), idx)
            ):
                doc.add_edge(old_command, idx, DomainEdgeKind.SHELL_PIPE)
            command_expected = False
            idx += 1
            continue

        if "/" in value or ("." in value and not value.startswith("-") and len(value) > 2):
            doc.set_role(idx, DomainRoleKind.PATH, entity=next_entity)
            next_entity += 1
            if previous_command is not None:
                doc.add_edge(previous_command, idx, DomainEdgeKind.SHELL_COMMAND_FILE)

        if value in {";", "&&", "||"}:
            command_expected = True
        idx += 1


def parse_shell(text: str, shell_kind: str = "bash") -> ParsedDomainDocument:
    """Parse a shell script using tree-sitter and extract domain edges.

    Supports: bash, sh, ksh, zsh, powershell/pwsh/ps1, batch/cmd.
    Falls back to heuristic token-based parsing when tree-sitter is unavailable.
    """
    domain = _SHELL_DOMAIN_MAP.get(shell_kind, DomainKind.BASH)
    doc = ParsedDomainDocument.new(
        domain=domain,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"parser": "tree-sitter", "shell_kind": shell_kind},
    )

    try:
        parser = _get_parser(shell_kind)
    except Exception:
        doc.metadata["parser"] = "heuristic-fallback"
        _heuristic_assign_roles_and_edges(doc, shell_kind)
        return doc

    source_bytes = text.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    if root.has_error and root.child_count == 0:
        return doc.mark_raw("tree-sitter produced empty parse")

    lang = _SHELL_LANGUAGE_MAP.get(shell_kind, "bash")
    if lang == "powershell":
        raw_edges = _extract_powershell_edges(root, source_bytes)
    elif lang == "tcsh":
        raw_edges = _extract_tcsh_edges(root, source_bytes)
    elif lang == "batch":
        raw_edges = _extract_batch_edges(root, source_bytes)
    else:
        raw_edges = _extract_bash_edges(root, source_bytes)

    _attach_edges_to_doc(doc, raw_edges)
    _assign_roles(doc, root, source_bytes, lang, shell_kind)

    doc.metadata["tree_sitter_language"] = lang
    doc.metadata["ast_root_type"] = root.type
    doc.metadata["ast_has_error"] = root.has_error

    is_fallback = shell_kind in ("tcsh", "csh") and root.type == "program"
    if root.has_error and not is_fallback:
        doc.mark_raw(f"malformed_{shell_kind}_shell")
    elif not is_fallback and _has_unbalanced_control_flow(doc, shell_kind):
        doc.mark_raw(f"malformed_{shell_kind}_shell")

    return doc


def _has_unbalanced_control_flow(doc: ParsedDomainDocument, shell_kind: str) -> bool:
    """Detect unmatched shell control-flow closers in the token stream."""
    if shell_kind in ("tcsh", "csh"):
        openers = {"if": "endif", "foreach": "end", "while": "end", "switch": "endsw"}
        closers = {"end", "endsw", "endif"}
    else:
        openers = {"if": "fi", "case": "esac", "for": "done", "while": "done", "until": "done"}
        if shell_kind in ("bash", "zsh", "ksh"):
            openers["select"] = "done"
        closers = {"fi", "esac", "done"}

    expected: list[str] = []
    in_comment_line = -1
    for tok in doc.tokens:
        if tok.text == "#":
            in_comment_line = tok.line
            continue
        if tok.line == in_comment_line:
            continue
        if tok.text.startswith(('"', "'")):
            continue
        word = tok.text.lower()
        if word in openers:
            expected.append(openers[word])
        elif word in closers:
            if not expected or expected.pop() != word:
                return True
    return bool(expected)


_SHELL_KEYWORDS = {
    "if", "then", "else", "elif", "fi", "for", "do", "done", "case",
    "esac", "while", "until", "export", "readonly", "unset",
}
_BASH_KEYWORDS = _SHELL_KEYWORDS | {"declare", "local", "mapfile", "readarray", "select", "shopt"}
_ZSH_KEYWORDS = _SHELL_KEYWORDS | {"autoload", "emulate", "setopt", "unsetopt", "zmodload"}
_KSH_KEYWORDS = _SHELL_KEYWORDS | {"compound", "function", "integer", "nameref", "typeset"}
_TCSH_KEYWORDS = {"setenv", "unsetenv", "alias", "foreach", "endif", "switch", "endsw", "end"}

_DIALECT_KEYWORDS = {
    "bash": _BASH_KEYWORDS,
    "sh": _SHELL_KEYWORDS,
    "zsh": _ZSH_KEYWORDS,
    "ksh": _KSH_KEYWORDS,
    "tcsh": _TCSH_KEYWORDS,
    "csh": _TCSH_KEYWORDS,
    "powershell": set(),
    "batch": set(),
}


def _assign_roles(
    doc: ParsedDomainDocument, root: Any, source: bytes, lang: str, shell_kind: str = ""
) -> None:
    """Assign DomainRoleKind roles to tokens based on AST node types."""
    tokens = doc.tokens
    if not tokens:
        return

    def byte_to_token_idx(byte_offset: int) -> int | None:
        best: int | None = None
        for idx, tok in enumerate(tokens):
            if tok.start <= byte_offset < tok.end:
                return idx
            if tok.start <= byte_offset:
                best = idx
            else:
                break
        return best

    next_entity = 1
    command_types = {"command_name"}
    var_types = {"variable_name", "variable", "expansion", "simple_expansion"}
    pipe_types = {"|", "pipe"}
    redir_types = {"file_redirect", "redirect", "redirection"}

    if lang == "powershell":
        command_types = {"command_name"}
        var_types = {"variable", "variable_name"}
    elif lang == "tcsh":
        command_types = {"setenv_command", "unsetenv_command", "echo_command"}
        var_types = {"variable_substitution"}
        redir_types = {"redirect_operator"}
    elif lang == "batch":
        command_types = {"keyword"}
        var_types = {"variable_name", "variable_reference"}

    keywords = _DIALECT_KEYWORDS.get(shell_kind, _DIALECT_KEYWORDS.get(lang, _SHELL_KEYWORDS))

    for node in _walk(root):
        if node.type == "declaration_command":
            for c in node.children:
                if c.type in ("typeset", "declare", "export", "local", "readonly", "unset"):
                    idx = byte_to_token_idx(c.start_byte)
                    if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                        text = _node_text(c, source)
                        if text in keywords:
                            doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
                        else:
                            doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
                        next_entity += 1
                elif c.type == "variable_assignment":
                    for vc in c.children:
                        if vc.type == "variable_name":
                            idx = byte_to_token_idx(vc.start_byte)
                            if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                                doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
                                next_entity += 1
                            break
        elif lang == "tcsh" and node.type == "simple_command":
            has_builtin = any(c.type == "builtin_command" for c in node.children)
            if not has_builtin:
                for c in node.children:
                    if c.type == "word":
                        for sub in c.children:
                            if sub.type == "identifier":
                                idx = byte_to_token_idx(sub.start_byte)
                                if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                                    text = _node_text(sub, source)
                                    if text in keywords:
                                        doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
                                    else:
                                        doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
                                    next_entity += 1
                                break
                        break
        elif node.type in command_types:
            idx = byte_to_token_idx(node.start_byte)
            if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                text = _node_text(node, source)
                if text in keywords:
                    doc.set_role(idx, DomainRoleKind.KEYWORD, entity=next_entity)
                else:
                    doc.set_role(idx, DomainRoleKind.COMMAND, entity=next_entity)
                next_entity += 1
        elif node.type in _VAR_ASSIGN_TYPES:
            for c in node.children:
                if c.type == "variable_name":
                    idx = byte_to_token_idx(c.start_byte)
                    if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                        doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
                        next_entity += 1
                    break
        elif node.type in var_types:
            idx = byte_to_token_idx(node.start_byte)
            if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                doc.set_role(idx, DomainRoleKind.VARIABLE)
        elif node.type in pipe_types and node.child_count == 0:
            idx = byte_to_token_idx(node.start_byte)
            if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                doc.set_role(idx, DomainRoleKind.PIPE)
        elif node.type in redir_types:
            idx = byte_to_token_idx(node.start_byte)
            if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                doc.set_role(idx, DomainRoleKind.REDIRECT)
        elif node.type == "word":
            text = _node_text(node, source)
            if "/" in text or (
                "." in text
                and not text.startswith("-")
                and not text.startswith("$")
                and len(text) > 2
            ):
                idx = byte_to_token_idx(node.start_byte)
                if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                    doc.set_role(idx, DomainRoleKind.PATH)

    if shell_kind in ("tcsh", "csh"):
        for node in _walk(root):
            if node.type == "setenv_command" or (
                node.type == "command_name"
                and _node_text(node, source) in ("setenv", "unsetenv")
            ):
                # Native grammar: setenv_command → parent builtin_command → grandparent simple_command
                # Fallback bash grammar: command_name → parent command
                container = node.parent
                if container is not None and container.type == "builtin_command":
                    container = container.parent
                if container is not None:
                    siblings = container.children
                    found_cmd = False
                    for sib in siblings:
                        if sib == node or (sib.type == "builtin_command" and node.parent == sib):
                            found_cmd = True
                            continue
                        if found_cmd and sib.type == "word":
                            idx = byte_to_token_idx(sib.start_byte)
                            if idx is not None and doc.role_ids[idx] == int(DomainRoleKind.NONE):
                                doc.set_role(idx, DomainRoleKind.ENVIRONMENT, entity=next_entity)
                                next_entity += 1
                            break


def _attach_edges_to_doc(
    doc: ParsedDomainDocument, raw_edges: list[tuple[int, int, int]]
) -> None:
    """Map byte-offset edges to token-index edges on the ParsedDomainDocument."""
    tokens = doc.tokens
    if not tokens:
        return

    def byte_to_token_idx(byte_offset: int) -> int | None:
        best: int | None = None
        for idx, tok in enumerate(tokens):
            if tok.start <= byte_offset < tok.end:
                return idx
            if tok.start <= byte_offset:
                best = idx
            else:
                break
        return best

    seen: set[tuple[int, int, int]] = set()
    for src_byte, dst_byte, kind in raw_edges:
        src_idx = byte_to_token_idx(src_byte)
        dst_idx = byte_to_token_idx(dst_byte)
        if src_idx is None or dst_idx is None:
            continue
        if src_idx == dst_idx:
            continue
        triple = (src_idx, dst_idx, kind)
        if triple not in seen:
            seen.add(triple)
            doc.edges.append(triple)


__all__ = ["parse_shell"]
