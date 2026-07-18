"""Make / Automake domain parser."""

from __future__ import annotations

from cppmega.data.build_parsers.base import ParsedDomainDocument, is_option
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


def parse_make(text: str, *, domain: DomainKind = DomainKind.MAKE, build_kind: str = "make") -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=domain,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": build_kind, "parser_adapter": build_kind},
    )
    next_entity = 1
    current_target: int | None = None

    for line_no, raw_line in enumerate(text.splitlines()):
        line_indices = doc.token_indices_on_line(line_no)
        comment_offset = next(
            (
                idx
                for idx, char in enumerate(raw_line)
                if char == "#" and (idx == 0 or raw_line[idx - 1] != "\\")
            ),
            len(raw_line),
        )
        line_indices = [
            idx for idx in line_indices if doc.tokens[idx].column < comment_offset
        ]
        if not line_indices:
            continue
        stripped = raw_line.lstrip()
        first_idx = line_indices[0]
        if raw_line.startswith("\t") or stripped.startswith(("@", "-", "+")):
            command_idx = first_idx
            if doc.tokens[command_idx].text in {"@", "-", "+"} and len(line_indices) > 1:
                command_idx = line_indices[1]
            doc.set_role(command_idx, DomainRoleKind.COMMAND, entity=next_entity)
            command_entity = next_entity
            next_entity += 1
            if current_target is not None:
                doc.add_edge(current_target, command_idx, DomainEdgeKind.BUILD_RULE_COMMAND)
            for arg_idx in line_indices:
                value = doc.tokens[arg_idx].text
                if arg_idx == command_idx:
                    continue
                if is_option(value):
                    doc.set_role(arg_idx, DomainRoleKind.OPTION, scope=command_entity)
                elif "/" in value or "." in value:
                    doc.set_role(arg_idx, DomainRoleKind.PATH, scope=command_entity)
            continue

        if "=" in [doc.tokens[i].text for i in line_indices]:
            eq_pos = next(i for i in line_indices if doc.tokens[i].text == "=")
            lhs = [i for i in line_indices if i < eq_pos]
            if lhs:
                doc.set_role(lhs[0], DomainRoleKind.VARIABLE, entity=next_entity)
                next_entity += 1
            continue

        colon_idx = next(
            (i for i in line_indices if doc.tokens[i].text in {":", "::"}),
            None,
        )
        if colon_idx is not None:
            targets = [i for i in line_indices if i < colon_idx]
            prereqs: list[int] = []
            for token_idx in (i for i in line_indices if i > colon_idx):
                value = doc.tokens[token_idx].text
                if value == ";":
                    break
                if value in {"|", "||", ".WAIT"}:
                    continue
                prereqs.append(token_idx)
            if targets:
                current_target = targets[0]
                doc.set_role(current_target, DomainRoleKind.TARGET, entity=next_entity)
                target_entity = next_entity
                next_entity += 1
                for prereq_idx in prereqs:
                    doc.set_role(prereq_idx, DomainRoleKind.PREREQUISITE, entity=next_entity, scope=target_entity)
                    next_entity += 1
                    doc.add_edge(current_target, prereq_idx, DomainEdgeKind.BUILD_TARGET_DEP)

    return doc


def parse_automake(text: str) -> ParsedDomainDocument:
    doc = parse_make(text, domain=DomainKind.AUTOMAKE, build_kind="automake")
    for quote in ('"', "'"):
        escaped = False
        count = 0
        for char in text:
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == quote and not escaped:
                count += 1
            escaped = False
        if count % 2:
            return doc.mark_raw("malformed_automake_syntax")
    return doc


__all__ = ["parse_automake", "parse_make"]
