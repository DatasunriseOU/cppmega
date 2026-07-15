#!/usr/bin/env python3
"""Tier-2 PR ingest: render a pr_store record into a `pr_discussion` text block.

This is the bridge between the store (component 3/4) and the commit-doc docstring
(component 5). Given an assembled PR record it produces a single string that
build_docstring emits at the HEAD of the commit block (before PRE/POST/diff),
and that the extractor attaches to each record as ``record['pr_discussion']``.

Size discipline: the block is sanely bounded (per-section line/char caps) so a
pathological PR cannot blow up a single doc; the document token-limit +
route-by-fit length-bucketing handle the remaining size variance (a bigger
discussion -> more tokens -> a larger length bucket).
"""

from typing import Optional


def _clip(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    if len(text) > max_chars:
        return text[:max_chars].rstrip() + " ...[truncated]"
    return text


def render_discussion(
    rec: Optional[dict],
    max_comments: int = 30,
    max_reviews: int = 20,
    max_body_chars: int = 4000,
    max_item_chars: int = 1500,
    max_total_chars: int = 24000,
) -> str:
    """Render an assembled PR record to a plain-text discussion block.

    Returns '' when there is no usable discussion (so build_docstring can skip
    emitting an empty @discussion section). RAISES on a malformed record (a dict
    missing pr_number) rather than silently rendering garbage.
    """
    if not rec:
        return ""
    if "pr_number" not in rec:
        raise ValueError(f"render_discussion: record missing pr_number: {rec!r}")

    lines: list[str] = []
    title = (rec.get("pr_title") or "").strip()
    number = rec.get("pr_number")
    header = f"PR #{number}" + (f": {title}" if title else "")
    lines.append(header)

    body = _clip(rec.get("pr_body") or "", max_body_chars)
    if body:
        lines.append("")
        lines.append(body)

    comments = rec.get("comments") or []
    if comments:
        lines.append("")
        lines.append(f"--- Discussion ({len(comments)} comments) ---")
        for c in comments[:max_comments]:
            user = (c.get("user") or "?").strip()
            path = c.get("path")
            tag = f"@{user}" + (f" ({path})" if path else "")
            lines.append(f"{tag}: {_clip(c.get('body') or '', max_item_chars)}")
        if len(comments) > max_comments:
            lines.append(f"...[+{len(comments) - max_comments} more comments]")

    reviews = rec.get("reviews") or []
    review_items = [r for r in reviews if (r.get("body") or "").strip() or r.get("state")]
    if review_items:
        lines.append("")
        lines.append(f"--- Reviews ({len(review_items)}) ---")
        for r in review_items[:max_reviews]:
            user = (r.get("user") or "?").strip()
            state = (r.get("state") or "").strip()
            head = f"@{user}" + (f" [{state}]" if state else "")
            rbody = _clip(r.get("body") or "", max_item_chars)
            lines.append(f"{head}: {rbody}" if rbody else head)
        if len(review_items) > max_reviews:
            lines.append(f"...[+{len(review_items) - max_reviews} more reviews]")

    linked = rec.get("linked_issues") or []
    if linked:
        lines.append("")
        lines.append(f"--- Linked issues ({len(linked)}) ---")
        for li in linked:
            t = (li.get("title") or "").strip()
            lib = _clip(li.get("body") or "", max_item_chars)
            lines.append(f"#{li.get('number')} {t}".rstrip())
            if lib:
                lines.append(lib)

    text = "\n".join(lines).strip()
    if not text or text == header:
        # Only a bare header (no body/comments/reviews/issues): not useful.
        return ""
    return _clip(text, max_total_chars)
