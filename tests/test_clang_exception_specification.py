from __future__ import annotations

from types import SimpleNamespace

from tools.clang_indexer import index_project as ip


def test_unknown_clang_exception_specification_enum_is_kept_in_signature() -> None:
    type_info = SimpleNamespace(
        spelling="int ()",
        get_canonical=lambda: type_info,
    )
    result_info = SimpleNamespace(
        spelling="int",
        get_canonical=lambda: result_info,
    )

    class UnknownExceptionCursor:
        displayname = "route()"
        type = type_info
        result_type = result_info

        def get_arguments(self):
            return []

        @property
        def exception_specification_kind(self):
            raise ValueError("Unknown template argument kind 9")

    signature = ip._cursor_canonical_signature(UnknownExceptionCursor())

    assert "exception=UNKNOWN" in signature
    assert "9" in signature
