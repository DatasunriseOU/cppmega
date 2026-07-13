#!/usr/bin/env python3
"""Restore and verify a committed cppmega Megatron archive transport."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Iterable

if __package__:
    from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
        DEFAULT_BUCKET,
        DEFAULT_ENDPOINT,
        DEFAULT_PREFIX,
        _load_env_file,
        _s3_env,
        _sha256,
        _validate_bundle,
    )
else:
    from publish_megatron_bundle_to_nebius_s3 import (  # type: ignore[no-redef]
        DEFAULT_BUCKET,
        DEFAULT_ENDPOINT,
        DEFAULT_PREFIX,
        _load_env_file,
        _s3_env,
        _sha256,
        _validate_bundle,
    )


def _aws_read(uri: str, *, endpoint: str, env: dict[str, str]) -> bytes:
    result = subprocess.run(
        ["aws", "s3", "cp", uri, "-", "--endpoint-url", endpoint, "--no-progress"],
        env=env,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to read {uri}: {result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def _aws_download(
    uri: str, destination: Path, *, endpoint: str, env: dict[str, str]
) -> None:
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            uri,
            str(destination),
            "--endpoint-url",
            endpoint,
            "--only-show-errors",
            "--no-progress",
        ],
        env=env,
        check=True,
    )


def _validate_transport(
    transport: dict, *, expected_bundle_id: str | None = None
) -> None:
    if transport.get("schema") != "cppmega_megatron_bundle_transport_v1":
        raise ValueError(f"unsupported transport schema: {transport.get('schema')!r}")
    bundle_id = str(transport.get("bundle_id", ""))
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}", bundle_id):
        raise ValueError(f"unsafe transport bundle ID: {bundle_id!r}")
    if expected_bundle_id and bundle_id != expected_bundle_id:
        raise ValueError(
            f"transport bundle mismatch: {bundle_id!r} != {expected_bundle_id!r}"
        )
    archive = transport.get("archive")
    if not isinstance(archive, dict) or archive.get("format") != "tar.zst":
        raise ValueError("transport has no supported tar.zst archive")
    if not str(archive.get("uri", "")).startswith("s3://"):
        raise ValueError("transport archive URI must use s3://")
    if int(archive.get("size", 0)) <= 0:
        raise ValueError("transport archive size must be positive")
    digest = str(archive.get("sha256", ""))
    if len(digest) != 64:
        raise ValueError("transport archive SHA-256 is invalid")
    logical_manifest = transport.get("logical_manifest")
    if not isinstance(logical_manifest, dict):
        raise ValueError("transport has no logical manifest descriptor")
    if not str(logical_manifest.get("uri", "")).startswith("s3://"):
        raise ValueError("transport logical manifest URI must use s3://")
    if int(logical_manifest.get("size", 0)) <= 0:
        raise ValueError("transport logical manifest size must be positive")
    if len(str(logical_manifest.get("sha256", ""))) != 64:
        raise ValueError("transport logical manifest SHA-256 is invalid")
    if transport.get("logical_manifest_sha256") != logical_manifest.get("sha256"):
        raise ValueError("transport logical manifest hashes disagree")
    if len(str(transport.get("artifact_set_sha256", ""))) != 64:
        raise ValueError("transport artifact-set SHA-256 is invalid")


def _extract_tar_zst(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    decoder = subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert decoder.stdout is not None
    extraction = subprocess.run(
        ["tar", "-xf", "-", "-C", str(destination)],
        stdin=decoder.stdout,
        capture_output=True,
        check=False,
    )
    decoder.stdout.close()
    decoder_stderr = decoder.communicate()[1]
    if decoder.returncode != 0 or extraction.returncode != 0:
        shutil.rmtree(destination, ignore_errors=True)
        raise RuntimeError(
            "archive extraction failed: "
            f"zstd={decoder.returncode} {decoder_stderr.decode(errors='replace').strip()} "
            f"tar={extraction.returncode} "
            f"{extraction.stderr.decode(errors='replace').strip()}"
        )


def _require_free_space(
    output_root: Path, *, artifact_bytes: int, archive_bytes: int, headroom_gb: int
) -> None:
    required = artifact_bytes + archive_bytes + headroom_gb * 1024**3
    free = shutil.disk_usage(output_root).free
    if free < required:
        raise RuntimeError(
            f"insufficient free space under {output_root}: "
            f"free={free} required={required}"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bundle-id")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--hash-jobs", type=int, default=4)
    parser.add_argument("--free-space-headroom-gb", type=int, default=10)
    parser.add_argument("--keep-archive", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    _load_env_file(args.env_file)
    env = _s3_env()
    prefix = args.prefix.strip("/")

    expected_transport_sha256: str | None = None
    if args.bundle_id:
        bundle_id = args.bundle_id
        transport_uri = (
            f"s3://{args.bucket}/{prefix}/transports/{bundle_id}/transport.json"
        )
    else:
        latest_uri = f"s3://{args.bucket}/{prefix}/latest_transport.json"
        latest_bytes = _aws_read(latest_uri, endpoint=args.endpoint_url, env=env)
        latest = json.loads(latest_bytes)
        if latest.get("schema") != "cppmega_megatron_latest_transport_v1":
            raise ValueError(f"unsupported latest schema: {latest.get('schema')!r}")
        bundle_id = str(latest["bundle_id"])
        transport_uri = str(latest["transport"])
        expected_transport_sha256 = str(latest["transport_sha256"])

    transport_bytes = _aws_read(
        transport_uri, endpoint=args.endpoint_url, env=env
    )
    if (
        expected_transport_sha256
        and hashlib.sha256(transport_bytes).hexdigest() != expected_transport_sha256
    ):
        raise ValueError("transport descriptor SHA-256 does not match latest pointer")
    transport = json.loads(transport_bytes)
    _validate_transport(transport, expected_bundle_id=bundle_id)

    logical_manifest_info = transport["logical_manifest"]
    logical_manifest_bytes = _aws_read(
        str(logical_manifest_info["uri"]), endpoint=args.endpoint_url, env=env
    )
    if len(logical_manifest_bytes) != int(logical_manifest_info["size"]):
        raise ValueError("remote logical manifest size does not match transport")
    if hashlib.sha256(logical_manifest_bytes).hexdigest() != logical_manifest_info["sha256"]:
        raise ValueError("remote logical manifest SHA-256 does not match transport")
    logical_manifest = json.loads(logical_manifest_bytes)
    if logical_manifest.get("bundle_id") != bundle_id:
        raise ValueError("remote logical manifest bundle ID does not match transport")
    if logical_manifest.get("artifact_set_sha256") != transport["artifact_set_sha256"]:
        raise ValueError("remote logical manifest artifact set does not match transport")
    if int(logical_manifest.get("artifact_count", -1)) != int(
        transport["artifact_count"]
    ) or int(logical_manifest.get("artifact_bytes", -1)) != int(
        transport["artifact_bytes"]
    ):
        raise ValueError("remote logical manifest counts do not match transport")

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / bundle_id
    if destination.exists():
        manifest, _artifacts = _validate_bundle(destination, args.hash_jobs)
        if manifest["bundle_id"] != bundle_id:
            raise ValueError("existing destination contains a different bundle")
        print(json.dumps({"bundle": str(destination), "status": "already_verified"}))
        return 0

    partial = output_root / f".{bundle_id}.partial"
    archive = output_root / f".{bundle_id}.tar.zst"
    if partial.exists() or archive.exists():
        raise RuntimeError(
            f"stale restore state exists; inspect and remove {partial} / {archive}"
        )

    archive_info = transport["archive"]
    _require_free_space(
        output_root,
        artifact_bytes=int(transport["artifact_bytes"]),
        archive_bytes=int(archive_info["size"]),
        headroom_gb=max(0, args.free_space_headroom_gb),
    )
    _aws_download(
        str(archive_info["uri"]), archive, endpoint=args.endpoint_url, env=env
    )
    if archive.stat().st_size != int(archive_info["size"]):
        raise ValueError("downloaded archive size does not match transport descriptor")
    if _sha256(archive) != archive_info["sha256"]:
        raise ValueError("downloaded archive SHA-256 does not match transport descriptor")

    _extract_tar_zst(archive, partial)
    manifest, _artifacts = _validate_bundle(partial, args.hash_jobs)
    if manifest["bundle_id"] != bundle_id:
        raise ValueError("restored logical bundle ID does not match transport")
    if manifest["artifact_set_sha256"] != transport["artifact_set_sha256"]:
        raise ValueError("restored artifact set does not match transport")
    if _sha256(partial / "manifest.json") != transport["logical_manifest_sha256"]:
        raise ValueError("restored logical manifest does not match transport")

    os.replace(partial, destination)
    if not args.keep_archive:
        archive.unlink()
    receipt = {
        "schema": "cppmega_megatron_restore_receipt_v1",
        "bundle_id": bundle_id,
        "transport": transport_uri,
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }
    (destination / "restore_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"bundle": str(destination), "status": "restored_verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
