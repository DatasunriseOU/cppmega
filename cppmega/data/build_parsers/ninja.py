"""Ninja manifest domain parser."""

from __future__ import annotations

from cppmega.data.build_parsers.base import ParsedDomainDocument, is_option
from cppmega.data.domain_schema import (
    DomainEdgeKind,
    DomainKind,
    DomainRoleKind,
    ParseConfidence,
)


def parse_ninja(text: str) -> ParsedDomainDocument:
    doc = ParsedDomainDocument.new(
        domain=DomainKind.NINJA,
        text=text,
        confidence=ParseConfidence.HEURISTIC,
        metadata={"build_kind": "ninja", "parser_adapter": "ninja"},
    )
    next_entity = 1
    current_rule: int | None = None

    for line_no, raw_line in enumerate(text.splitlines()):
        line_indices = doc.token_indices_on_line(line_no)
        comment_offset = next(
            (
                idx
                for idx, char in enumerate(raw_line)
                if char == "#" and (idx == 0 or raw_line[idx - 1] != "$")
            ),
            len(raw_line),
        )
        line_indices = [
            idx for idx in line_indices if doc.tokens[idx].column < comment_offset
        ]
        if not line_indices:
            continue
        first_text = doc.tokens[line_indices[0]].text
        if first_text == "rule" and len(line_indices) > 1:
            rule_idx = line_indices[1]
            current_rule = rule_idx
            doc.set_role(line_indices[0], DomainRoleKind.KEYWORD)
            doc.set_role(rule_idx, DomainRoleKind.RULE, entity=next_entity)
            next_entity += 1
        elif first_text == "build":
            colon_idx = next((i for i in line_indices if doc.tokens[i].text == ":"), None)
            if colon_idx is None:
                continue
            outputs = [
                i for i in line_indices
                if line_indices[0] < i < colon_idx
                and doc.tokens[i].text not in {"|", "||"}
            ]
            tail = [
                i for i in line_indices
                if i > colon_idx and doc.tokens[i].text not in {"|", "||"}
            ]
            if tail:
                rule_idx = tail[0]
                doc.set_role(rule_idx, DomainRoleKind.RULE)
            for out_idx in outputs:
                doc.set_role(out_idx, DomainRoleKind.OUTPUT, entity=next_entity)
                out_entity = next_entity
                next_entity += 1
                for inp_idx in tail[1:]:
                    doc.set_role(inp_idx, DomainRoleKind.INPUT, entity=next_entity, scope=out_entity)
                    next_entity += 1
                    doc.add_edge(out_idx, inp_idx, DomainEdgeKind.BUILD_ACTION_INPUT)
        elif first_text == "command":
            eq_idx = next((i for i in line_indices if doc.tokens[i].text == "="), None)
            if eq_idx is not None:
                command_tokens = [i for i in line_indices if i > eq_idx]
                if command_tokens:
                    cmd_idx = command_tokens[0]
                    doc.set_role(cmd_idx, DomainRoleKind.COMMAND, entity=next_entity)
                    command_entity = next_entity
                    next_entity += 1
                    if current_rule is not None:
                        doc.add_edge(current_rule, cmd_idx, DomainEdgeKind.BUILD_RULE_COMMAND)
                    for arg_idx in command_tokens[1:]:
                        if is_option(doc.tokens[arg_idx].text):
                            doc.set_role(arg_idx, DomainRoleKind.OPTION, scope=command_entity)

    return doc


__all__ = ["parse_ninja"]
