"""Cppmega tokenizer wrapper with M0.1 contract checks.

The loader intentionally refuses artifacts that do not exactly match the
documented M0.1 tokenizer contract.  This avoids silently training against a
nearby nanochat tokenizer with different reserved IDs.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
import operator
from pathlib import Path
from typing import Any

from cppmega.data.tokenizer_contract import (
    EXPECTED_VOCAB_SIZE,
    SPECIAL_TOKEN_IDS,
    validate_tokenizer_artifact_contract,
)
from cppmega.data.prompt_graph import (
    CppPromptTokenizerAdapter,
    normalize_cpp_whitespace_with_offsets as normalize_whitespace_with_offsets,
)

EXPECTED_SPECIAL_TOKENS: dict[str, int] = {
    f"<{role}>": SPECIAL_TOKEN_IDS[role]
    for role in (
        "BOS",
        "EOS",
        "FIM_PREFIX",
        "FIM_MIDDLE",
        "FIM_SUFFIX",
        "CODE_START",
        "CODE_END",
        "THINK_START",
        "THINK_END",
        "QUERY_TOOL",
        "TOOL_RESULT",
        "FIM_INSTRUCTION",
        "SPACE",
        "NL",
    )
}


class TokenizerContractError(ValueError):
    """Raised when a tokenizer artifact does not satisfy M0.1."""


class CppMegaTokenizer:
    """Thin wrapper around ``tokenizers.Tokenizer`` with stable cppmega APIs."""

    def __init__(self, tokenizer: Any, *, path: Path):
        self._tokenizer = tokenizer
        self.path = path
        self._prompt_adapter = CppPromptTokenizerAdapter(
            tokenizer, tokenizer_path=path
        )
        self._vocab: dict[str, int] = dict(tokenizer.get_vocab())
        self._id_to_token = {token_id: token for token, token_id in self._vocab.items()}
        self._space_token_id = EXPECTED_SPECIAL_TOKENS["<SPACE>"]
        self._nl_token_id = EXPECTED_SPECIAL_TOKENS["<NL>"]

    @property
    def vocab_size(self) -> int:
        return int(self._tokenizer.get_vocab_size(with_added_tokens=True))

    @property
    def bos_token_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<BOS>"]

    @property
    def eos_token_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<EOS>"]

    @property
    def fim_prefix_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_PREFIX>"]

    @property
    def fim_middle_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_MIDDLE>"]

    @property
    def fim_suffix_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_SUFFIX>"]

    @property
    def code_start_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<CODE_START>"]

    @property
    def code_end_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<CODE_END>"]

    @property
    def think_start_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<THINK_START>"]

    @property
    def think_end_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<THINK_END>"]

    @property
    def query_tool_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<QUERY_TOOL>"]

    @property
    def tool_result_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<TOOL_RESULT>"]

    @property
    def fim_instruction_id(self) -> int:
        return EXPECTED_SPECIAL_TOKENS["<FIM_INSTRUCTION>"]

    @property
    def space_token_id(self) -> int:
        return self._space_token_id

    @property
    def nl_token_id(self) -> int:
        return self._nl_token_id

    def get_vocab_size(self) -> int:
        return self.vocab_size

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Collapse whitespace runs to ``<SPACE>``/``<NL>`` sentinel tokens.

        String/char/raw-string-literal aware: whitespace inside a literal is
        preserved verbatim (``"a    b"`` stays ``"a    b"``), only whitespace
        outside any literal is collapsed.
        """
        normalized, _offsets = normalize_whitespace_with_offsets(text)
        return normalized

    def encode(
        self,
        text: str | Sequence[str],
        *,
        prepend: int | str | None = None,
        append: int | str | None = None,
    ) -> list[int] | list[list[int]]:
        prepend_id = self._resolve_optional_token(prepend)
        append_id = self._resolve_optional_token(append)

        if isinstance(text, str):
            normalized = self._normalize_whitespace(text)
            ids = list(self._tokenizer.encode(normalized).ids)
            return self._with_optional_tokens(ids, prepend_id, append_id)
        if isinstance(text, Sequence):
            normalized_batch = [self._normalize_whitespace(t) for t in text]
            rows = [
                list(encoded.ids)
                for encoded in self._tokenizer.encode_batch(normalized_batch)
            ]
            return [
                self._with_optional_tokens(row, prepend_id, append_id) for row in rows
            ]
        raise TypeError(f"text must be str or sequence[str], got {type(text).__name__}")

    def encode_batch(self, texts: Sequence[str]) -> list[list[int]]:
        normalized = [self._normalize_whitespace(t) for t in texts]
        return [list(encoded.ids) for encoded in self._tokenizer.encode_batch(normalized)]

    def encode_with_offsets(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode once and map tokenizer offsets back to the original source.

        The tokenizer operates on whitespace-normalized text. Every normalized
        character already carries its original source span, so this method keeps
        prompt graph coordinates exactly aligned with the IDs passed to the model.
        """

        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        try:
            return self._prompt_adapter.encode_with_offsets(text)
        except (TypeError, ValueError) as exc:
            raise TokenizerContractError(str(exc)) from exc

    def decode(self, ids: Iterable[int]) -> str:
        """Decode IDs by simple concat then replacing ``<SPACE>``/``<NL>`` sentinels.

        The encoder substitutes whitespace runs with ``<SPACE>``/``<NL>`` tokens so
        that BPE-split identifiers decode without spurious inter-token spaces.
        Byte-exact for any input whose internal whitespace runs are at most a
        single space or single newline (longer runs are collapsed by the
        normalizer). Matches the CUDA-side ``nanochat.cpp_tokenizer`` decode.
        """
        parts: list[str] = []
        for position, raw_id in enumerate(ids):
            if isinstance(raw_id, bool):
                raise TokenizerContractError(
                    f"decode id at position {position} must be an integer, got bool"
                )
            try:
                token_id = operator.index(raw_id)
            except TypeError as exc:
                raise TokenizerContractError(
                    f"decode id at position {position} must be an integer, "
                    f"got {type(raw_id).__name__}"
                ) from exc
            token = self._id_to_token.get(token_id)
            if token is None:
                raise TokenizerContractError(
                    f"decode id at position {position} is outside tokenizer vocab: "
                    f"{token_id}"
                )
            parts.append(token)
        s = "".join(parts)
        return s.replace("<SPACE>", " ").replace("<NL>", "\n")

    def token_for_id(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def id_for_token(self, token: str) -> int | None:
        return self._vocab.get(token)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.encode(*args, **kwargs)

    def _resolve_optional_token(self, token: int | str | None) -> int | None:
        if token is None:
            return None
        if isinstance(token, int) and not isinstance(token, bool):
            if token not in self._id_to_token:
                raise TokenizerContractError(
                    f"special token id {token} is outside tokenizer vocab"
                )
            return token
        if isinstance(token, str):
            token_id = self.id_for_token(token)
            if token_id is None:
                raise TokenizerContractError(f"unknown special token {token!r}")
            return token_id
        raise TypeError(f"token must be int or str, got {type(token).__name__}")

    @staticmethod
    def _with_optional_tokens(
        ids: list[int], prepend_id: int | None, append_id: int | None
    ) -> list[int]:
        if prepend_id is not None:
            ids.insert(0, prepend_id)
        if append_id is not None:
            ids.append(append_id)
        return ids


def load_cppmega_tokenizer(path: str | Path) -> CppMegaTokenizer:
    """Load a tokenizer only if it satisfies the M0.1 contract."""

    tokenizer_path = _resolve_tokenizer_path(path)
    try:
        validate_tokenizer_artifact_contract(
            tokenizer_path,
            require_frozen_artifact=True,
        )
    except (RuntimeError, ValueError) as exc:
        raise TokenizerContractError(str(exc)) from exc

    try:
        from tokenizers import Tokenizer  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise TokenizerContractError(
            "tokenizers package is required to load cppmega tokenizer artifacts"
        ) from exc

    return CppMegaTokenizer(Tokenizer.from_file(str(tokenizer_path)), path=tokenizer_path)


def _resolve_tokenizer_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "tokenizer.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"tokenizer artifact not found: {candidate}")
    return candidate


__all__ = [
    "CppMegaTokenizer",
    "EXPECTED_SPECIAL_TOKENS",
    "EXPECTED_VOCAB_SIZE",
    "TokenizerContractError",
    "load_cppmega_tokenizer",
    "normalize_whitespace_with_offsets",
]
