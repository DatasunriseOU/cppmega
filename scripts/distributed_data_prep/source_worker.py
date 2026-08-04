#!/usr/bin/env python3
"""Execute receipt-bound source jobs on transient local SSD.

Network sources are fetched with ``git clone --mirror``.  The worker records the
complete refs snapshot, HEAD, checkout tree, and all-object inventory before it
indexes the pinned commit.  Non-network sources must be immutable,
generation-pinned GCS tar.zst objects.  Workers deliberately do not receive the
tokenizer or a dedup database on the indexer command line: their output is a
canonical pre-global-dedup enriched stream, never a training-ready shard.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.parse
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, Sequence

if __package__ in {None, ""}:  # pragma: no cover - direct CLI execution
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))

from scripts.distributed_data_prep._common import (  # noqa: E402
    ContractError,
    atomic_write_json,
    canonical_json_bytes,
    canonical_sha256,
    gcs_join,
    load_json_object,
    require_exact_fields,
    require_git_object,
    require_int,
    require_sha256,
    run_checked,
    sha256_file,
    validate_gcs_uri,
)
from scripts.distributed_data_prep.source_manifest import (  # noqa: E402
    PRE_GLOBAL_SCHEMA,
    load_source_manifest,
    repositories_for_worker,
    validate_source_manifest,
)

SOURCE_WORKER_RECEIPT_SCHEMA = "cppmega.distributed_source_worker_receipt_v2"
ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA = (
    "cppmega.distributed_source_assignment_completion_receipt_v1"
)
CANONICAL_DOCUMENT_ORDER = "canonical_enriched_json_v1"
_SORT_FIELDS = (
    "repo",
    "filepath",
    "doc_type",
    "header_fragment_kind",
    "commit_hash",
    "file_local_commit_index",
)


class ObjectStore(Protocol):
    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]: ...

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]: ...

    def describe_if_present(
        self, uri: str, *, generation: str | None = None
    ) -> Mapping[str, object] | None: ...


class GcloudObjectStore:
    """Generation-aware GCS transport without adding a Python dependency."""

    def __init__(self, executable: str = "gcloud") -> None:
        self.executable = executable

    def describe(
        self, uri: str, *, generation: str | None = None
    ) -> dict[str, object]:
        validate_gcs_uri(uri, where="GCS object")
        target = f"{uri}#{generation}" if generation is not None else uri
        completed = run_checked(
            [
                self.executable,
                "storage",
                "objects",
                "describe",
                target,
                "--format=json",
            ]
        )
        try:
            raw = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ContractError(f"gcloud returned invalid object metadata for {uri}") from exc
        if not isinstance(raw, dict):
            raise ContractError(f"gcloud returned non-object metadata for {uri}")
        generation = str(raw.get("generation", ""))
        size = raw.get("size")
        if not generation.isdecimal() or int(generation) < 1:
            raise ContractError(f"GCS object has no valid generation: {uri}")
        try:
            size_int = int(size)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"GCS object has no valid size: {uri}") from exc
        return {
            "uri": uri,
            "generation": generation,
            "size_bytes": size_int,
            "crc32c": raw.get("crc32c"),
            "md5_hash": raw.get("md5Hash"),
        }

    def describe_if_present(
        self, uri: str, *, generation: str | None = None
    ) -> dict[str, object] | None:
        """Return exact metadata for one known object without listing a bucket.

        Slot completion receipts have deterministic names.  The worker service
        account is deliberately unable to list objects, so resume has to make a
        single object GET and distinguish an absent receipt from an operational
        error.  A 404 is the only non-error negative result.
        """

        validate_gcs_uri(uri, where="GCS object")
        bucket, object_name = uri[len("gs://") :].split("/", 1)
        token = run_checked(
            [self.executable, "auth", "print-access-token"]
        ).stdout.strip()
        if not token or any(character.isspace() for character in token):
            raise ContractError("gcloud returned an invalid access token")
        query: dict[str, str] = {}
        if generation is not None:
            query["generation"] = str(generation)
        endpoint = (
            "https://storage.googleapis.com/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/"
            f"{urllib.parse.quote(object_name, safe='')}"
        )
        if query:
            endpoint += "?" + urllib.parse.urlencode(query)
        with tempfile.TemporaryDirectory(prefix="cppmega-gcs-describe-") as raw_tmp:
            response = Path(raw_tmp) / "response.json"
            completed = subprocess.run(
                [
                    "curl",
                    "--config",
                    "-",
                    "--silent",
                    "--show-error",
                    "--location",
                    "--output",
                    str(response),
                    "--write-out",
                    "%{http_code}",
                    endpoint,
                ],
                input=f'header = "Authorization: Bearer {token}"\n',
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"GCS object metadata lookup failed for {uri}: "
                    f"{completed.stderr[-4000:]}"
                )
            status = completed.stdout.strip()
            if status == "404":
                return None
            if status != "200":
                detail = response.read_text(encoding="utf-8", errors="replace")
                raise RuntimeError(
                    f"GCS object metadata lookup returned HTTP {status} for {uri}: "
                    f"{detail[-4000:]}"
                )
            try:
                raw = json.loads(response.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ContractError(
                    f"GCS returned invalid object metadata for {uri}"
                ) from exc
        if not isinstance(raw, dict):
            raise ContractError(f"GCS returned non-object metadata for {uri}")
        resolved_generation = str(raw.get("generation", ""))
        size = raw.get("size")
        if not resolved_generation.isdecimal() or int(resolved_generation) < 1:
            raise ContractError(f"GCS object has no valid generation: {uri}")
        try:
            size_int = int(size)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"GCS object has no valid size: {uri}") from exc
        if generation is not None and resolved_generation != str(generation):
            raise ContractError(f"GCS generation selector drifted for {uri}")
        return {
            "uri": uri,
            "generation": resolved_generation,
            "size_bytes": size_int,
            "crc32c": raw.get("crc32c"),
            "md5_hash": raw.get("md5Hash"),
        }

    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]:
        validate_gcs_uri(uri, where="GCS publication URI")
        if not source.is_file():
            raise FileNotFoundError(source)
        bucket, object_name = uri[len("gs://") :].split("/", 1)
        token = run_checked(
            [self.executable, "auth", "print-access-token"]
        ).stdout.strip()
        if not token or any(character.isspace() for character in token):
            raise ContractError("gcloud returned an invalid access token")
        endpoint = (
            "https://storage.googleapis.com/upload/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o?"
            + urllib.parse.urlencode(
                {
                    "uploadType": "media",
                    "name": object_name,
                    "ifGenerationMatch": "0",
                }
            )
        )
        curl_config = (
            f'header = "Authorization: Bearer {token}"\n'
            'header = "Content-Type: application/octet-stream"\n'
        )
        command = [
            "curl",
            "--config",
            "-",
            "--fail-with-body",
            "--silent",
            "--show-error",
            "--request",
            "POST",
            "--upload-file",
            str(source),
            endpoint,
        ]
        completed = subprocess.run(
            command,
            input=curl_config,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            # A retry is safe only when the immutable object already has the exact
            # bytes.  Download and hash it; never turn an arbitrary 412/403 into a
            # successful publication.
            with tempfile.TemporaryDirectory(prefix="cppmega-gcs-verify-") as raw_tmp:
                existing = Path(raw_tmp) / "existing"
                try:
                    metadata = self.describe(uri)
                    self.download(
                        uri,
                        existing,
                        generation=str(metadata["generation"]),
                    )
                except Exception as verify_error:
                    raise RuntimeError(
                        f"immutable GCS publication failed for {uri}: "
                        f"{completed.stderr[-4000:]}"
                    ) from verify_error
                if sha256_file(existing) != sha256_file(source):
                    raise ContractError(
                        f"immutable GCS object already exists with different bytes: {uri}"
                    )
                return metadata
        metadata = self.describe(uri)
        if int(metadata["size_bytes"]) != source.stat().st_size:
            raise ContractError(f"published GCS object size mismatch: {uri}")
        return metadata

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]:
        validate_gcs_uri(uri, where="GCS download URI")
        bucket, object_name = uri[len("gs://") :].split("/", 1)
        token = run_checked(
            [self.executable, "auth", "print-access-token"]
        ).stdout.strip()
        if not token or any(character.isspace() for character in token):
            raise ContractError("gcloud returned an invalid access token")
        query = {"alt": "media"}
        if generation is not None:
            query["generation"] = str(generation)
        endpoint = (
            "https://storage.googleapis.com/download/storage/v1/b/"
            f"{urllib.parse.quote(bucket, safe='')}/o/"
            f"{urllib.parse.quote(object_name, safe='')}?"
            + urllib.parse.urlencode(query)
        )
        curl_config = f'header = "Authorization: Bearer {token}"\n'
        destination.parent.mkdir(parents=True, exist_ok=True)
        stage = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
        completed = subprocess.run(
            [
                "curl",
                "--config",
                "-",
                "--fail-with-body",
                "--silent",
                "--show-error",
                "--location",
                "--output",
                str(stage),
                endpoint,
            ],
            input=curl_config,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stage.unlink(missing_ok=True)
            raise RuntimeError(
                f"exact GCS download failed for {uri}: {completed.stderr[-4000:]}"
            )
        os.replace(stage, destination)
        metadata = self.describe(uri, generation=generation)
        if generation is not None and str(metadata["generation"]) != str(generation):
            raise ContractError(f"GCS generation selector drifted for {uri}")
        if destination.stat().st_size != int(metadata["size_bytes"]):
            raise ContractError(f"downloaded GCS object size mismatch: {uri}")
        return metadata


class LocalObjectStore:
    """Filesystem object store used by bounded smoke tests."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, uri: str) -> Path:
        validate_gcs_uri(uri, where="local object URI")
        relative = uri[len("gs://") :]
        return self.root / relative

    def publish_if_absent(self, source: Path, uri: str) -> Mapping[str, object]:
        destination = self._path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if sha256_file(destination) != sha256_file(source):
                raise ContractError(f"local immutable object collision: {uri}")
        else:
            stage = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            shutil.copyfile(source, stage)
            os.replace(stage, destination)
        return {
            "uri": uri,
            "generation": "1",
            "size_bytes": destination.stat().st_size,
            "crc32c": None,
            "md5_hash": None,
        }

    def download(
        self, uri: str, destination: Path, *, generation: str | None = None
    ) -> Mapping[str, object]:
        if generation not in {None, "1"}:
            raise ContractError(f"unknown local object generation for {uri}")
        source = self._path(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        return {
            "uri": uri,
            "generation": "1",
            "size_bytes": destination.stat().st_size,
            "crc32c": None,
            "md5_hash": None,
        }

    def describe_if_present(
        self, uri: str, *, generation: str | None = None
    ) -> Mapping[str, object] | None:
        if generation not in {None, "1"}:
            raise ContractError(f"unknown local object generation for {uri}")
        source = self._path(uri)
        if not source.is_file():
            return None
        return {
            "uri": uri,
            "generation": "1",
            "size_bytes": source.stat().st_size,
            "crc32c": None,
            "md5_hash": None,
        }


def _git(git_dir: Path, *args: str) -> str:
    return run_checked(["git", f"--git-dir={git_dir}", *args]).stdout.strip()


def _sorted_file_digest(source: Path, destination: Path) -> tuple[str, int, int, dict[str, int]]:
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    completed = subprocess.run(
        ["sort", "-o", str(destination), str(source)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"object inventory sort failed: {completed.stderr[-4000:]}")
    digest = hashlib.sha256()
    count = 0
    total_bytes = 0
    types: dict[str, int] = {}
    with destination.open("rb") as stream:
        for line in stream:
            digest.update(line)
            fields = line.rstrip(b"\n").split(b" ")
            if len(fields) != 3:
                raise ContractError("Git object inventory line is malformed")
            object_type = fields[1].decode("ascii")
            try:
                object_size = int(fields[2])
            except ValueError as exc:
                raise ContractError("Git object inventory size is malformed") from exc
            count += 1
            total_bytes += object_size
            types[object_type] = types.get(object_type, 0) + 1
    return digest.hexdigest(), count, total_bytes, types


def acquire_git_mirror(
    source: Mapping[str, object], scratch: Path
) -> tuple[Path, dict[str, object]]:
    """Clone a full mirror and materialize exactly the manifest-pinned commit."""

    remote = str(source["remote_url"])
    expected_commit = require_git_object(
        source["expected_commit"], where="git source expected_commit"
    )
    mirror = scratch / "mirror.git"
    checkout = scratch / "checkout"
    run_checked(["git", "clone", "--mirror", "--no-hardlinks", remote, mirror])
    if _git(mirror, "rev-parse", "--is-bare-repository") != "true":
        raise ContractError("git clone --mirror did not create a bare repository")
    resolved_commit = _git(mirror, "rev-parse", f"{expected_commit}^{{commit}}")
    if resolved_commit != expected_commit:
        raise ContractError(
            f"mirror resolved a different commit: {resolved_commit} != {expected_commit}"
        )
    tree = _git(mirror, "rev-parse", f"{expected_commit}^{{tree}}")
    expected_tree = source.get("expected_tree")
    if expected_tree is not None and tree != expected_tree:
        raise ContractError(f"pinned Git tree drifted: {tree} != {expected_tree}")

    fsck = subprocess.run(
        ["git", f"--git-dir={mirror}", "fsck", "--full", "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    if fsck.returncode != 0:
        raise ContractError(f"full mirror failed git fsck: {fsck.stderr[-8000:]}")

    refs_lines = sorted(
        line
        for line in _git(
            mirror,
            "for-each-ref",
            "--format=%(refname)%00%(objectname)%00%(objecttype)",
        ).splitlines()
        if line
    )
    refs_payload = ("\n".join(refs_lines) + ("\n" if refs_lines else "")).encode(
        "utf-8"
    )
    try:
        head_ref = _git(mirror, "symbolic-ref", "-q", "HEAD")
    except RuntimeError:
        head_ref = None
    try:
        head_commit = _git(mirror, "rev-parse", "HEAD^{commit}")
    except RuntimeError:
        head_commit = None

    unordered = scratch / "objects.unsorted"
    ordered = scratch / "objects.sorted"
    with unordered.open("wb") as stream:
        completed = subprocess.run(
            [
                "git",
                f"--git-dir={mirror}",
                "cat-file",
                "--batch-all-objects",
                "--batch-check=%(objectname) %(objecttype) %(objectsize)",
            ],
            stdout=stream,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"git object inventory failed: {completed.stderr[-8000:].decode(errors='replace')}"
        )
    inventory_sha, object_count, object_bytes, object_types = _sorted_file_digest(
        unordered, ordered
    )

    run_checked(
        [
            "git",
            f"--git-dir={mirror}",
            "worktree",
            "add",
            "--detach",
            checkout,
            expected_commit,
        ]
    )
    checked_out = run_checked(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"]
    ).stdout.strip()
    if checked_out != expected_commit:
        raise ContractError("materialized worktree commit drifted")
    status = run_checked(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"]
    ).stdout
    if status:
        raise ContractError("fresh materialized worktree is dirty")
    gitlinks = run_checked(
        ["git", "-C", str(checkout), "ls-files", "--stage"]
    ).stdout.splitlines()
    gitlink_count = sum(1 for line in gitlinks if line.startswith("160000 "))
    return checkout, {
        "kind": "git_mirror",
        "remote_url": remote,
        "expected_commit": expected_commit,
        "resolved_commit": resolved_commit,
        "tree": tree,
        "head_ref": head_ref,
        "head_commit": head_commit,
        "refs": {
            "count": len(refs_lines),
            "sha256": hashlib.sha256(refs_payload).hexdigest(),
        },
        "objects": {
            "count": object_count,
            "logical_bytes": object_bytes,
            "types": dict(sorted(object_types.items())),
            "inventory_sha256": inventory_sha,
        },
        "gitlink_count": gitlink_count,
        "fsck": "ok",
    }


def _safe_archive_relative(name: str, strip_components: int) -> Path | None:
    pure = PurePosixPath(name)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ContractError(f"immutable source archive contains unsafe path: {name!r}")
    if len(pure.parts) <= strip_components:
        return None
    return Path(*pure.parts[strip_components:])


def extract_immutable_tar_zst(
    archive: Path, destination: Path, *, strip_components: int
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=False)
    zstd = subprocess.Popen(
        ["zstd", "-dc", "--", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert zstd.stdout is not None
    files = 0
    bytes_written = 0
    try:
        with tarfile.open(fileobj=zstd.stdout, mode="r|") as tar:
            for member in tar:
                relative = _safe_archive_relative(member.name, strip_components)
                if relative is None:
                    continue
                target = destination / relative
                try:
                    target.resolve().relative_to(destination.resolve())
                except ValueError as exc:
                    raise ContractError(
                        f"immutable source archive escaped extraction root: {member.name}"
                    ) from exc
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise ContractError(
                        "immutable source archive contains a link/device/special entry: "
                        f"{member.name}"
                    )
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ContractError(f"cannot read tar member: {member.name}")
                with source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
                if target.stat().st_size != member.size:
                    raise ContractError(f"tar member size mismatch: {member.name}")
                files += 1
                bytes_written += member.size
    finally:
        zstd.stdout.close()
    stderr = zstd.stderr.read() if zstd.stderr is not None else b""
    return_code = zstd.wait()
    if return_code != 0:
        raise RuntimeError(f"zstd extraction failed: {stderr[-8000:].decode(errors='replace')}")
    if files == 0:
        raise ContractError("immutable source archive contains no regular files")
    return {"file_count": files, "extracted_bytes": bytes_written}


def acquire_immutable_gcs_tar(
    source: Mapping[str, object], scratch: Path, store: ObjectStore
) -> tuple[Path, dict[str, object]]:
    uri = validate_gcs_uri(source["uri"], where="immutable source URI")
    generation = str(source["generation"])
    archive = scratch / "source.tar.zst"
    metadata = dict(store.download(uri, archive, generation=generation))
    if str(metadata.get("generation")) != generation:
        raise ContractError("immutable source object generation drifted")
    digest = sha256_file(archive)
    if digest != source["sha256"]:
        raise ContractError("immutable source object SHA-256 drifted")
    checkout = scratch / "checkout"
    extraction = extract_immutable_tar_zst(
        archive,
        checkout,
        strip_components=int(source["strip_components"]),
    )
    return checkout, {
        "kind": "immutable_gcs_tar",
        "object": {
            **metadata,
            "sha256": digest,
        },
        "archive_format": "tar.zst",
        "strip_components": int(source["strip_components"]),
        **extraction,
    }


def _canonical_sort_key(document: Mapping[str, object], payload_sha256: str) -> str:
    values = [str(document.get(field, "")) for field in _SORT_FIELDS]
    return json.dumps(values, ensure_ascii=True, separators=(",", ":")) + payload_sha256


def canonicalize_enriched_jsonl(
    source: Path,
    destination: Path,
    *,
    project_id: str,
    chunk_rows: int = 10_000,
) -> dict[str, object]:
    """Canonicalize JSON keys and externally sort documents with bounded RAM."""

    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunk_root = Path(tempfile.mkdtemp(prefix="candidate-sort-", dir=destination.parent))
    chunks: list[Path] = []
    rows: list[tuple[str, bytes]] = []
    documents = 0
    source_bytes = 0

    def flush() -> None:
        if not rows:
            return
        rows.sort(key=lambda item: (item[0], item[1]))
        path = chunk_root / f"chunk-{len(chunks):08d}.txt"
        with path.open("wb") as stream:
            for key, payload in rows:
                stream.write(key.encode("ascii"))
                stream.write(b"\t")
                stream.write(payload)
                stream.write(b"\n")
        chunks.append(path)
        rows.clear()

    try:
        with source.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                source_bytes += len(raw)
                if not raw.endswith(b"\n"):
                    raise ContractError(f"indexer JSONL line {line_number} is truncated")
                try:
                    document = json.loads(raw)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ContractError(
                        f"indexer JSONL line {line_number} is invalid"
                    ) from exc
                if not isinstance(document, dict):
                    raise ContractError(f"indexer JSONL line {line_number} is not an object")
                text = document.get("text")
                if not isinstance(text, str) or not text:
                    raise ContractError(
                        f"indexer JSONL line {line_number} has no non-empty text"
                    )
                row_repo = document.get("repo")
                if row_repo != project_id:
                    raise ContractError(
                        f"indexer JSONL line {line_number} repo drifted: {row_repo!r}"
                    )
                payload = canonical_json_bytes(document)
                payload_sha = hashlib.sha256(payload).hexdigest()
                rows.append((_canonical_sort_key(document, payload_sha), payload))
                documents += 1
                if len(rows) >= chunk_rows:
                    flush()
        flush()
        if documents == 0:
            raise ContractError("indexer emitted no pre-global-dedup documents")
        digest = hashlib.sha256()
        output_bytes = 0
        handles = [path.open("rb") for path in chunks]
        try:
            with destination.open("wb") as output:
                for encoded in heapq.merge(*handles):
                    _key, separator, payload = encoded.partition(b"\t")
                    if not separator or not payload.endswith(b"\n"):
                        raise ContractError("canonical sort spool is corrupt")
                    output.write(payload)
                    digest.update(payload)
                    output_bytes += len(payload)
        finally:
            for handle in handles:
                handle.close()
        return {
            "schema": PRE_GLOBAL_SCHEMA,
            "document_order": CANONICAL_DOCUMENT_ORDER,
            "documents": documents,
            "indexer_bytes": source_bytes,
            "canonical_bytes": output_bytes,
            "canonical_stream_sha256": digest.hexdigest(),
        }
    finally:
        shutil.rmtree(chunk_root, ignore_errors=True)


def compress_zstd(source: Path, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as output:
        completed = subprocess.run(
            ["zstd", "-19", "-T1", "--no-progress", "-c", "--", str(source)],
            stdout=output,
            stderr=subprocess.PIPE,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"zstd compression failed: {completed.stderr[-8000:].decode(errors='replace')}"
        )
    run_checked(["zstd", "-t", "--", str(destination)])
    version = run_checked(["zstd", "--version"]).stdout.strip()
    return {
        "compression": "zstd",
        "level": 19,
        "threads": 1,
        "zstd_version": version,
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _verify_pipeline_files(
    manifest: Mapping[str, object],
    *,
    repo_root: Path,
    indexer: Path,
    tokenizer: Path,
    quarantine_manifest: Path,
) -> None:
    pipeline = manifest["pipeline"]
    assert isinstance(pipeline, Mapping)
    for path, field in (
        (indexer, "indexer_sha256"),
        (tokenizer, "tokenizer_sha256"),
        (quarantine_manifest, "quarantine_manifest_sha256"),
    ):
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"pipeline input is not a regular file: {path}")
        if sha256_file(path) != pipeline[field]:
            raise ContractError(f"pipeline input hash drifted: {path}")
    revision = run_checked(["git", "-C", str(repo_root), "rev-parse", "HEAD"]).stdout.strip()
    if revision != manifest["code_revision"]:
        raise ContractError("worker checkout does not match manifest code_revision")
    if run_checked(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"]
    ).stdout:
        raise ContractError("worker code checkout has tracked changes")


def _run_indexer(
    *,
    python: Path,
    indexer: Path,
    source_root: Path,
    project_id: str,
    raw_output: Path,
    quarantine_manifest: Path,
    quarantine_receipt: Path,
    parse_workers: int,
    memory_limit_gb: float,
    max_tokens: int,
) -> dict[str, object]:
    # No --tokenizer-path and no --dedup-db: both per-repo and global claims are
    # intentionally disabled.  The central reducer owns the first primary-copy
    # decision in canonical manifest/document order.
    command = [
        str(python),
        str(indexer),
        "--project-dir",
        str(source_root),
        "--project-id",
        project_id,
        "--output",
        str(raw_output),
        "--enriched",
        "--max-tokens",
        str(max_tokens),
        "--exclude-dirs",
        "__pycache__,node_modules,build,.git",
        "--memory-limit-gb",
        str(memory_limit_gb),
        "--parse-workers",
        str(parse_workers),
        "--source-quarantine-manifest",
        str(quarantine_manifest),
        "--source-quarantine-receipt",
        str(quarantine_receipt),
    ]
    run_checked(command, capture_output=False)
    if not raw_output.is_file() or raw_output.stat().st_size == 0:
        raise ContractError("indexer did not produce a non-empty enriched JSONL")
    if not quarantine_receipt.is_file():
        raise ContractError("indexer did not publish a quarantine receipt")
    return {
        # Keep the receipt retry-stable: the actual argv contains random local
        # scratch paths, while every semantic switch is captured below and the
        # manifest separately binds the exact indexer bytes.
        "mode": "single_project_pre_global_enriched_v1",
        "project_id": project_id,
        "enriched": True,
        "max_tokens": max_tokens,
        "parse_workers": parse_workers,
        "memory_limit_gb": memory_limit_gb,
        "excluded_directories": ["__pycache__", "node_modules", "build", ".git"],
        "dedup_applied": False,
        "tokenizer_passed_to_indexer": False,
        "raw_output_sha256": sha256_file(raw_output),
        "quarantine_receipt_sha256": sha256_file(quarantine_receipt),
    }


def validate_quarantine_receipt_file(
    path: Path,
    *,
    project_id: str,
    manifest_sha256: str,
) -> dict[str, object]:
    """Validate the physical source-quarantine sidecar used by release audits."""

    raw, receipt = load_json_object(path, where="source quarantine receipt")
    require_exact_fields(
        receipt,
        {
            "schema",
            "project_id",
            "manifest_path",
            "manifest_sha256",
            "manifest_entry_count",
            "project_manifest_entry_count",
            "candidate_count_before_quarantine",
            "candidate_count_after_quarantine",
            "quarantined_count",
            "entries",
            "external_reference_omissions",
            "parse_recovery",
        },
        where="source quarantine receipt",
    )
    if (
        receipt["schema"] != "cppmega.source_quarantine_receipt_v1"
        or receipt["project_id"] != project_id
        or receipt["manifest_sha256"] != manifest_sha256
    ):
        raise ContractError("source quarantine receipt binding drifted")
    before = require_int(
        receipt["candidate_count_before_quarantine"],
        where="quarantine candidates before",
    )
    after = require_int(
        receipt["candidate_count_after_quarantine"],
        where="quarantine candidates after",
    )
    quarantined = require_int(
        receipt["quarantined_count"], where="quarantined count"
    )
    if after + quarantined != before:
        raise ContractError("source quarantine candidate counts do not close")
    entries = receipt["entries"]
    if not isinstance(entries, list) or len(entries) != quarantined:
        raise ContractError("source quarantine entry count does not close")
    require_int(receipt["manifest_entry_count"], where="quarantine manifest entries")
    require_int(
        receipt["project_manifest_entry_count"],
        where="project quarantine manifest entries",
    )
    for field, schema in (
        ("external_reference_omissions", "cppmega.external_reference_omissions_v1"),
        ("parse_recovery", "cppmega.source_parse_recovery_v1"),
    ):
        value = receipt[field]
        if (
            not isinstance(value, Mapping)
            or value.get("schema") != schema
            or not isinstance(value.get("status"), str)
        ):
            raise ContractError(f"source quarantine {field} receipt drifted")
    receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["receipt_size_bytes"] = len(raw)
    return receipt


def validate_worker_receipt(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    job: Mapping[str, object],
) -> dict[str, object]:
    value = dict(receipt)
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "assignment",
            "source_snapshot",
            "candidate",
            "artifact",
            "quarantine_artifact",
            "indexer",
            "training_ready",
        },
        where="source worker receipt",
    )
    if value["schema"] != SOURCE_WORKER_RECEIPT_SCHEMA or value["status"] != "complete":
        raise ContractError("source worker receipt schema/status is unsupported")
    if value["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ContractError("source worker receipt manifest binding drifted")
    require_sha256(value["manifest_file_sha256"], where="manifest_file_sha256")
    if value["training_ready"] is not False:
        raise ContractError("worker candidate must never claim training readiness")
    assignment = value["assignment"]
    if not isinstance(assignment, Mapping) or dict(assignment) != {
        key: job[key]
        for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
    }:
        raise ContractError("source worker assignment binding drifted")
    candidate = value["candidate"]
    if not isinstance(candidate, Mapping):
        raise ContractError("source worker candidate receipt is missing")
    if (
        candidate.get("schema") != PRE_GLOBAL_SCHEMA
        or candidate.get("document_order") != CANONICAL_DOCUMENT_ORDER
        or candidate.get("dedup_applied") is not False
    ):
        raise ContractError("source worker candidate is not pre-global-dedup")
    require_int(candidate.get("documents"), where="candidate.documents", minimum=1)
    require_sha256(
        candidate.get("canonical_stream_sha256"),
        where="candidate.canonical_stream_sha256",
    )
    artifact = value["artifact"]
    if not isinstance(artifact, Mapping):
        raise ContractError("source worker artifact receipt is missing")
    require_exact_fields(
        artifact,
        {
            "uri",
            "generation",
            "size_bytes",
            "crc32c",
            "md5_hash",
            "sha256",
            "compression",
        },
        where="source worker artifact",
    )
    artifact_uri = validate_gcs_uri(artifact.get("uri"), where="worker artifact URI")
    generation = str(artifact.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("source worker artifact generation is invalid")
    artifact_sha256 = require_sha256(
        artifact.get("sha256"), where="worker artifact sha256"
    )
    require_int(artifact.get("size_bytes"), where="worker artifact size", minimum=1)
    expected_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-candidates",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{artifact_sha256}.jsonl.zst",
    )
    if artifact_uri != expected_uri:
        raise ContractError("source worker artifact URI escaped its manifest namespace")
    compression = artifact.get("compression")
    if (
        not isinstance(compression, Mapping)
        or compression.get("compression") != "zstd"
        or compression.get("sha256") != artifact.get("sha256")
        or compression.get("size_bytes") != artifact.get("size_bytes")
    ):
        raise ContractError("source worker artifact compression binding drifted")
    quarantine_artifact = value["quarantine_artifact"]
    if not isinstance(quarantine_artifact, Mapping):
        raise ContractError("source worker quarantine artifact receipt is missing")
    require_exact_fields(
        quarantine_artifact,
        {"uri", "generation", "size_bytes", "crc32c", "md5_hash", "sha256"},
        where="source worker quarantine artifact",
    )
    quarantine_sha256 = require_sha256(
        quarantine_artifact.get("sha256"), where="quarantine artifact sha256"
    )
    quarantine_uri = validate_gcs_uri(
        quarantine_artifact.get("uri"), where="quarantine artifact URI"
    )
    expected_quarantine_uri = gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-quarantine-receipts",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{quarantine_sha256}.quarantine.json",
    )
    if quarantine_uri != expected_quarantine_uri:
        raise ContractError("quarantine artifact URI escaped its manifest namespace")
    quarantine_generation = str(quarantine_artifact.get("generation", ""))
    if not quarantine_generation.isdecimal() or int(quarantine_generation) < 1:
        raise ContractError("quarantine artifact generation is invalid")
    require_int(
        quarantine_artifact.get("size_bytes"),
        where="quarantine artifact size",
        minimum=1,
    )
    indexer = value["indexer"]
    if not isinstance(indexer, Mapping):
        raise ContractError("source worker indexer receipt is missing")
    if (
        indexer.get("mode") != "single_project_pre_global_enriched_v1"
        or indexer.get("project_id") != job["project_id"]
        or indexer.get("enriched") is not True
        or indexer.get("dedup_applied") is not False
        or indexer.get("tokenizer_passed_to_indexer") is not False
    ):
        raise ContractError("source worker indexer execution contract drifted")
    require_sha256(indexer.get("raw_output_sha256"), where="indexer raw output sha256")
    require_sha256(
        indexer.get("quarantine_receipt_sha256"),
        where="indexer quarantine receipt sha256",
    )
    if indexer.get("quarantine_receipt_sha256") != quarantine_sha256:
        raise ContractError("indexer quarantine receipt publication drifted")
    return value


def assignment_completion_uri(
    manifest: Mapping[str, object], job: Mapping[str, object]
) -> str:
    """Return the deterministic resume pointer for one manifest assignment."""

    assignment_sha256 = require_sha256(
        job.get("assignment_sha256"), where="assignment_sha256"
    )
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-assignment-completions",
        str(manifest["manifest_sha256"]),
        f"{assignment_sha256}.complete.json",
    )


def _source_receipt_uri(
    manifest: Mapping[str, object],
    job: Mapping[str, object],
    receipt: Mapping[str, object],
) -> str:
    artifact = receipt.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ContractError("source worker receipt has no artifact")
    compression = artifact.get("compression")
    if not isinstance(compression, Mapping):
        raise ContractError("source worker receipt has no compression metadata")
    compressed_sha256 = require_sha256(
        compression.get("sha256"), where="source receipt compressed sha256"
    )
    return gcs_join(
        str(manifest["gcs_output_prefix"]),
        "source-receipts",
        str(manifest["manifest_sha256"]),
        f"{int(job['ordinal']):05d}-{job['repo']}",
        f"{compressed_sha256}.receipt.json",
    )


def validate_assignment_completion_receipt(
    receipt: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
) -> dict[str, object]:
    """Validate the immutable pointer that makes one job safely resumable."""

    value = dict(receipt)
    require_exact_fields(
        value,
        {
            "schema",
            "status",
            "manifest_sha256",
            "manifest_file_sha256",
            "assignment",
            "source_receipt",
            "training_ready",
        },
        where="source assignment completion receipt",
    )
    if (
        value["schema"] != ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA
        or value["status"] != "complete"
        or value["manifest_sha256"] != manifest["manifest_sha256"]
        or value["manifest_file_sha256"] != manifest_file_sha256
        or value["training_ready"] is not False
    ):
        raise ContractError("source assignment completion receipt binding drifted")
    expected_assignment = {
        key: job[key]
        for key in ("ordinal", "repo", "project_id", "worker", "assignment_sha256")
    }
    assignment = value["assignment"]
    if not isinstance(assignment, Mapping) or dict(assignment) != expected_assignment:
        raise ContractError("source assignment completion assignment drifted")
    source_receipt = value["source_receipt"]
    if not isinstance(source_receipt, Mapping):
        raise ContractError("source assignment completion source receipt is missing")
    source = dict(source_receipt)
    require_exact_fields(
        source,
        {"uri", "generation", "size_bytes", "sha256"},
        where="source assignment completion source receipt",
    )
    validate_gcs_uri(source["uri"], where="source assignment completion receipt URI")
    generation = str(source["generation"])
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("source assignment completion receipt generation is invalid")
    require_int(
        source["size_bytes"],
        where="source assignment completion receipt size",
        minimum=1,
    )
    require_sha256(source["sha256"], where="source assignment completion receipt sha256")
    return value


def _load_completed_assignment(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    object_store: ObjectStore,
    scratch_root: Path,
) -> dict[str, object] | None:
    """Return a read-back source receipt for a confirmed assignment, if any."""

    pointer_uri = assignment_completion_uri(manifest, job)
    pointer_metadata = object_store.describe_if_present(pointer_uri)
    if pointer_metadata is None:
        return None
    pointer_generation = str(pointer_metadata.get("generation", ""))
    if not pointer_generation.isdecimal() or int(pointer_generation) < 1:
        raise ContractError(f"assignment completion pointer has invalid generation: {pointer_uri}")
    scratch_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="assignment-resume-", dir=scratch_root) as raw_tmp:
        temporary = Path(raw_tmp)
        pointer_path = temporary / "pointer.json"
        downloaded_pointer = object_store.download(
            pointer_uri, pointer_path, generation=pointer_generation
        )
        if (
            str(downloaded_pointer.get("uri")) != pointer_uri
            or str(downloaded_pointer.get("generation")) != pointer_generation
            or pointer_path.stat().st_size != int(pointer_metadata.get("size_bytes", -1))
        ):
            raise ContractError("assignment completion pointer readback drifted")
        _pointer_raw, pointer = load_json_object(
            pointer_path, where="source assignment completion receipt"
        )
        validated_pointer = validate_assignment_completion_receipt(
            pointer,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
        )
        source = validated_pointer["source_receipt"]
        assert isinstance(source, Mapping)
        source_uri = validate_gcs_uri(source["uri"], where="source receipt URI")
        source_generation = str(source["generation"])
        source_metadata = object_store.describe_if_present(
            source_uri, generation=source_generation
        )
        if source_metadata is None:
            raise ContractError("assignment pointer references a missing source receipt")
        source_path = temporary / "source-receipt.json"
        downloaded_source = object_store.download(
            source_uri, source_path, generation=source_generation
        )
        if (
            str(downloaded_source.get("uri")) != source_uri
            or str(downloaded_source.get("generation")) != source_generation
            or source_path.stat().st_size != int(source["size_bytes"])
            or sha256_file(source_path) != source["sha256"]
        ):
            raise ContractError("assignment completion source receipt readback drifted")
        _source_raw, source_receipt = load_json_object(
            source_path, where="source worker receipt"
        )
        validate_worker_receipt(source_receipt, manifest=manifest, job=job)
        if _source_receipt_uri(manifest, job, source_receipt) != source_uri:
            raise ContractError("assignment pointer source receipt URI drifted")
        return source_receipt


def _publish_assignment_completion(
    *,
    manifest: Mapping[str, object],
    manifest_file_sha256: str,
    job: Mapping[str, object],
    source_receipt: Mapping[str, object],
    source_receipt_path: Path,
    object_store: ObjectStore,
    scratch_root: Path,
) -> None:
    """Publish and read back a deterministic pointer after the source receipt."""

    receipt_uri = _source_receipt_uri(manifest, job, source_receipt)
    metadata = object_store.describe_if_present(receipt_uri)
    if metadata is None:
        raise ContractError("source receipt disappeared before assignment completion")
    generation = str(metadata.get("generation", ""))
    if not generation.isdecimal() or int(generation) < 1:
        raise ContractError("source receipt has invalid generation")
    source_sha256 = sha256_file(source_receipt_path)
    with tempfile.TemporaryDirectory(prefix="assignment-publish-", dir=scratch_root) as raw_tmp:
        temporary = Path(raw_tmp)
        verified_source = temporary / "source-receipt.json"
        downloaded = object_store.download(
            receipt_uri, verified_source, generation=generation
        )
        if (
            str(downloaded.get("uri")) != receipt_uri
            or str(downloaded.get("generation")) != generation
            or verified_source.stat().st_size != source_receipt_path.stat().st_size
            or sha256_file(verified_source) != source_sha256
        ):
            raise ContractError("published source receipt readback drifted")
        pointer: dict[str, object] = {
            "schema": ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA,
            "status": "complete",
            "manifest_sha256": manifest["manifest_sha256"],
            "manifest_file_sha256": manifest_file_sha256,
            "assignment": {
                key: job[key]
                for key in (
                    "ordinal",
                    "repo",
                    "project_id",
                    "worker",
                    "assignment_sha256",
                )
            },
            "source_receipt": {
                "uri": receipt_uri,
                "generation": generation,
                "size_bytes": source_receipt_path.stat().st_size,
                "sha256": source_sha256,
            },
            "training_ready": False,
        }
        validate_assignment_completion_receipt(
            pointer,
            manifest=manifest,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
        )
        pointer_path = temporary / "assignment-completion.json"
        atomic_write_json(pointer_path, pointer)
        pointer_uri = assignment_completion_uri(manifest, job)
        published = object_store.publish_if_absent(pointer_path, pointer_uri)
        pointer_generation = str(published.get("generation", ""))
        if str(published.get("uri")) != pointer_uri or not pointer_generation.isdecimal():
            raise ContractError("assignment completion pointer publication drifted")
        verified_pointer = temporary / "assignment-completion.verify.json"
        pointer_download = object_store.download(
            pointer_uri, verified_pointer, generation=pointer_generation
        )
        if (
            str(pointer_download.get("generation")) != pointer_generation
            or verified_pointer.stat().st_size != pointer_path.stat().st_size
            or sha256_file(verified_pointer) != sha256_file(pointer_path)
        ):
            raise ContractError("assignment completion pointer readback drifted")


def run_source_worker(
    manifest: Mapping[str, object],
    *,
    manifest_file_sha256: str,
    worker: str,
    scratch_root: Path,
    receipt_root: Path,
    repo_root: Path,
    python: Path,
    indexer: Path,
    tokenizer: Path,
    quarantine_manifest: Path,
    object_store: ObjectStore,
    parse_workers: int = 4,
    memory_limit_gb: float = 14.0,
    max_tokens: int | None = None,
) -> tuple[dict[str, object], ...]:
    """Run every assignment for one worker; publish data before each receipt."""

    plan = validate_source_manifest(manifest)
    require_sha256(manifest_file_sha256, where="manifest_file_sha256")
    pipeline = plan["pipeline"]
    assert isinstance(pipeline, Mapping)
    manifest_max_tokens = require_int(
        pipeline.get("index_max_tokens"),
        where="pipeline.index_max_tokens",
        minimum=1,
    )
    if max_tokens is None:
        max_tokens = manifest_max_tokens
    if max_tokens != manifest_max_tokens:
        raise ContractError("worker max_tokens drifted from the source manifest")
    if parse_workers < 1 or memory_limit_gb <= 0:
        raise ValueError("worker resource/index limits must be positive")
    _verify_pipeline_files(
        plan,
        repo_root=repo_root,
        indexer=indexer,
        tokenizer=tokenizer,
        quarantine_manifest=quarantine_manifest,
    )
    jobs = repositories_for_worker(plan, worker)
    scratch_root.mkdir(parents=True, exist_ok=True)
    receipt_root.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []
    for job in jobs:
        resumed = _load_completed_assignment(
            manifest=plan,
            manifest_file_sha256=manifest_file_sha256,
            job=job,
            object_store=object_store,
            scratch_root=scratch_root,
        )
        if resumed is not None:
            local_receipt = receipt_root / f"{int(job['ordinal']):05d}-{job['repo']}.json"
            atomic_write_json(local_receipt, resumed)
            receipts.append(resumed)
            continue
        with tempfile.TemporaryDirectory(
            prefix=f"source-{job['ordinal']:05d}-{job['repo']}-", dir=scratch_root
        ) as raw_scratch:
            scratch = Path(raw_scratch)
            source = job["source"]
            assert isinstance(source, Mapping)
            if source["kind"] == "git_mirror":
                checkout, source_snapshot = acquire_git_mirror(source, scratch)
            elif source["kind"] == "immutable_gcs_tar":
                checkout, source_snapshot = acquire_immutable_gcs_tar(
                    source, scratch, object_store
                )
            else:  # validate_source_manifest already rejects this.
                raise ContractError(f"unsupported source kind: {source['kind']}")

            raw_output = scratch / "pre-global.enriched.jsonl"
            quarantine_receipt = scratch / "source-quarantine-receipt.json"
            indexer_receipt = _run_indexer(
                python=python,
                indexer=indexer,
                source_root=checkout,
                project_id=str(job["project_id"]),
                raw_output=raw_output,
                quarantine_manifest=quarantine_manifest,
                quarantine_receipt=quarantine_receipt,
                parse_workers=parse_workers,
                memory_limit_gb=memory_limit_gb,
                max_tokens=max_tokens,
            )
            validated_quarantine = validate_quarantine_receipt_file(
                quarantine_receipt,
                project_id=str(job["project_id"]),
                manifest_sha256=str(pipeline["quarantine_manifest_sha256"]),
            )
            if (
                validated_quarantine["receipt_sha256"]
                != indexer_receipt["quarantine_receipt_sha256"]
            ):
                raise ContractError("indexer quarantine receipt digest drifted")
            canonical_output = scratch / "canonical.enriched.jsonl"
            candidate = canonicalize_enriched_jsonl(
                raw_output,
                canonical_output,
                project_id=str(job["project_id"]),
            )
            candidate["dedup_applied"] = False
            compressed = scratch / "canonical.enriched.jsonl.zst"
            compression = compress_zstd(canonical_output, compressed)
            artifact_uri = gcs_join(
                str(plan["gcs_output_prefix"]),
                "source-candidates",
                str(plan["manifest_sha256"]),
                f"{int(job['ordinal']):05d}-{job['repo']}",
                f"{compression['sha256']}.jsonl.zst",
            )
            published = dict(object_store.publish_if_absent(compressed, artifact_uri))
            if (
                int(published.get("size_bytes", -1)) != compressed.stat().st_size
                or str(published.get("uri")) != artifact_uri
            ):
                raise ContractError("published candidate object metadata drifted")
            published_generation = str(published.get("generation", ""))
            verified_download = scratch / "published-candidate.verify"
            verified_metadata = object_store.download(
                artifact_uri,
                verified_download,
                generation=published_generation,
            )
            if (
                str(verified_metadata.get("generation")) != published_generation
                or verified_download.stat().st_size != compressed.stat().st_size
                or sha256_file(verified_download) != compression["sha256"]
            ):
                raise ContractError("published candidate content verification failed")
            verified_download.unlink()
            artifact = {
                **published,
                "sha256": compression["sha256"],
                "compression": compression,
            }
            quarantine_sha256 = str(indexer_receipt["quarantine_receipt_sha256"])
            quarantine_uri = gcs_join(
                str(plan["gcs_output_prefix"]),
                "source-quarantine-receipts",
                str(plan["manifest_sha256"]),
                f"{int(job['ordinal']):05d}-{job['repo']}",
                f"{quarantine_sha256}.quarantine.json",
            )
            quarantine_artifact = dict(
                object_store.publish_if_absent(quarantine_receipt, quarantine_uri)
            )
            quarantine_generation = str(quarantine_artifact.get("generation", ""))
            quarantine_verified = scratch / "published-quarantine.verify"
            quarantine_metadata = object_store.download(
                quarantine_uri,
                quarantine_verified,
                generation=quarantine_generation,
            )
            if (
                str(quarantine_metadata.get("generation"))
                != quarantine_generation
                or quarantine_verified.stat().st_size
                != quarantine_receipt.stat().st_size
                or sha256_file(quarantine_verified) != quarantine_sha256
            ):
                raise ContractError(
                    "published quarantine receipt content verification failed"
                )
            quarantine_verified.unlink()
            quarantine_artifact["sha256"] = quarantine_sha256
            receipt: dict[str, object] = {
                "schema": SOURCE_WORKER_RECEIPT_SCHEMA,
                "status": "complete",
                "manifest_sha256": plan["manifest_sha256"],
                "manifest_file_sha256": manifest_file_sha256,
                "assignment": {
                    key: job[key]
                    for key in (
                        "ordinal",
                        "repo",
                        "project_id",
                        "worker",
                        "assignment_sha256",
                    )
                },
                "source_snapshot": source_snapshot,
                "candidate": candidate,
                "artifact": artifact,
                "quarantine_artifact": quarantine_artifact,
                "indexer": indexer_receipt,
                "training_ready": False,
            }
            validate_worker_receipt(receipt, manifest=plan, job=job)
            local_receipt = receipt_root / f"{int(job['ordinal']):05d}-{job['repo']}.json"
            atomic_write_json(local_receipt, receipt)
            receipt_uri = _source_receipt_uri(plan, job, receipt)
            # Publication order is intentional: an uploaded candidate without a
            # receipt is garbage-collectable; a receipt can never point to missing
            # data because it is uploaded last.
            object_store.publish_if_absent(local_receipt, receipt_uri)
            _publish_assignment_completion(
                manifest=plan,
                manifest_file_sha256=manifest_file_sha256,
                job=job,
                source_receipt=receipt,
                source_receipt_path=local_receipt,
                object_store=object_store,
                scratch_root=scratch,
            )
            receipts.append(receipt)
    return tuple(receipts)


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--worker", required=True)
    parser.add_argument("--scratch-root", required=True, type=Path)
    parser.add_argument("--receipt-root", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--indexer", type=Path, default=Path("tools/clang_indexer/index_project.py")
    )
    parser.add_argument(
        "--tokenizer", type=Path, default=Path("cppmega/tokenizer/tokenizer.json")
    )
    parser.add_argument(
        "--quarantine-manifest",
        type=Path,
        default=Path("configs/source_quarantine_manifest.json"),
    )
    parser.add_argument("--parse-workers", type=int, default=4)
    parser.add_argument("--memory-limit-gb", type=float, default=14.0)
    parser.add_argument("--max-tokens", type=int)
    args = parser.parse_args(argv)
    try:
        manifest, raw_sha256 = load_source_manifest(args.manifest)
        run_source_worker(
            manifest,
            manifest_file_sha256=raw_sha256,
            worker=args.worker,
            scratch_root=args.scratch_root,
            receipt_root=args.receipt_root,
            repo_root=args.repo_root.resolve(),
            python=args.python.resolve(),
            indexer=(args.repo_root / args.indexer).resolve()
            if not args.indexer.is_absolute()
            else args.indexer.resolve(),
            tokenizer=(args.repo_root / args.tokenizer).resolve()
            if not args.tokenizer.is_absolute()
            else args.tokenizer.resolve(),
            quarantine_manifest=(args.repo_root / args.quarantine_manifest).resolve()
            if not args.quarantine_manifest.is_absolute()
            else args.quarantine_manifest.resolve(),
            object_store=GcloudObjectStore(),
            parse_workers=args.parse_workers,
            memory_limit_gb=args.memory_limit_gb,
            max_tokens=args.max_tokens,
        )
    except (ContractError, RuntimeError, OSError, ValueError) as exc:
        parser.exit(2, f"distributed source worker failed: {exc}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(_main())


__all__ = [
    "ASSIGNMENT_COMPLETION_RECEIPT_SCHEMA",
    "CANONICAL_DOCUMENT_ORDER",
    "GcloudObjectStore",
    "LocalObjectStore",
    "ObjectStore",
    "SOURCE_WORKER_RECEIPT_SCHEMA",
    "acquire_git_mirror",
    "assignment_completion_uri",
    "canonicalize_enriched_jsonl",
    "compress_zstd",
    "extract_immutable_tar_zst",
    "run_source_worker",
    "validate_assignment_completion_receipt",
    "validate_worker_receipt",
    "validate_quarantine_receipt_file",
]
