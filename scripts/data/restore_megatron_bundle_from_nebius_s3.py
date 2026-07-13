#!/usr/bin/env python3
"""Restore and verify a committed cppmega Megatron archive transport."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import shutil
import subprocess
import tarfile
from typing import Iterable

if __package__:
    from scripts.data.publish_megatron_bundle_to_nebius_s3 import (
        DEFAULT_BUCKET,
        DEFAULT_ENDPOINT,
        DEFAULT_PREFIX,
        _load_env_file,
        _s3_env,
        _sha256,
        _validate_archive_member_names,
        _validate_bundle,
        _write_json_atomic,
    )
else:
    from publish_megatron_bundle_to_nebius_s3 import (  # type: ignore[no-redef]
        DEFAULT_BUCKET,
        DEFAULT_ENDPOINT,
        DEFAULT_PREFIX,
        _load_env_file,
        _s3_env,
        _sha256,
        _validate_archive_member_names,
        _validate_bundle,
        _write_json_atomic,
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


def _archive_receipt_path(archive: Path) -> Path:
    return archive.with_name(archive.name + ".receipt.json")


def _archive_matches(archive: Path, *, size: int, sha256: str) -> bool:
    return (
        archive.is_file()
        and archive.stat().st_size == size
        and _sha256(archive) == sha256
    )


def _acquire_archive(
    *,
    uri: str,
    archive: Path,
    endpoint: str,
    env: dict[str, str],
    expected_size: int,
    expected_sha256: str,
) -> dict[str, object]:
    download = archive.with_name(archive.name + ".download")
    receipt_path = _archive_receipt_path(archive)
    binding = {
        "schema": "cppmega_megatron_archive_download_receipt_v1",
        "uri": uri,
        "size": expected_size,
        "sha256": expected_sha256,
    }

    if archive.exists():
        if not _archive_matches(
            archive, size=expected_size, sha256=expected_sha256
        ):
            raise ValueError("existing archive does not match transport descriptor")
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if any(receipt.get(key) != value for key, value in binding.items()):
                raise ValueError("archive download receipt binding mismatch")
        else:
            _write_json_atomic(
                receipt_path,
                {
                    **binding,
                    "status": "recovered_verified",
                    "verified_at": datetime.now(timezone.utc).isoformat(),
                },
            )
        return {**binding, "status": "reused_verified", "receipt": str(receipt_path)}

    if receipt_path.exists():
        receipt_path.unlink()
    recovered_download = False
    if download.exists():
        if _archive_matches(
            download, size=expected_size, sha256=expected_sha256
        ):
            recovered_download = True
        else:
            download.unlink()
    if not download.exists():
        _aws_download(uri, download, endpoint=endpoint, env=env)
    if download.stat().st_size != expected_size:
        download.unlink(missing_ok=True)
        raise ValueError("downloaded archive size does not match transport descriptor")
    if _sha256(download) != expected_sha256:
        download.unlink(missing_ok=True)
        raise ValueError("downloaded archive SHA-256 does not match transport descriptor")
    os.replace(download, archive)
    status = "recovered_download_verified" if recovered_download else "downloaded_verified"
    receipt = {
        **binding,
        "status": status,
        "verified_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(receipt_path, receipt)
    return {**receipt, "receipt": str(receipt_path)}


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
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("transport archive SHA-256 is invalid")
    logical_manifest = transport.get("logical_manifest")
    if not isinstance(logical_manifest, dict):
        raise ValueError("transport has no logical manifest descriptor")
    if not str(logical_manifest.get("uri", "")).startswith("s3://"):
        raise ValueError("transport logical manifest URI must use s3://")
    if int(logical_manifest.get("size", 0)) <= 0:
        raise ValueError("transport logical manifest size must be positive")
    if not re.fullmatch(r"[0-9a-f]{64}", str(logical_manifest.get("sha256", ""))):
        raise ValueError("transport logical manifest SHA-256 is invalid")
    if transport.get("logical_manifest_sha256") != logical_manifest.get("sha256"):
        raise ValueError("transport logical manifest hashes disagree")
    if not re.fullmatch(r"[0-9a-f]{64}", str(transport.get("artifact_set_sha256", ""))):
        raise ValueError("transport artifact-set SHA-256 is invalid")
    if int(transport.get("artifact_count", 0)) <= 0:
        raise ValueError("transport artifact_count must be positive")
    if int(transport.get("artifact_bytes", 0)) <= 0:
        raise ValueError("transport artifact_bytes must be positive")


def _safe_archive_member_name(name: str) -> str:
    posix = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or posix.is_absolute()
        or any(part in ("", ".", "..") for part in posix.parts)
        or posix.as_posix() != name
    ):
        raise ValueError(f"unsafe archive member path: {name!r}")
    return name


def _validate_tar_member(member: tarfile.TarInfo) -> None:
    _safe_archive_member_name(member.name)
    if not member.isfile():
        raise ValueError(
            f"archive contains unsupported member type for {member.name!r}: "
            f"type={member.type!r}"
        )


def _stream_tar_zst(archive: Path) -> tuple[subprocess.Popen[bytes], tarfile.TarFile]:
    decoder = subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert decoder.stdout is not None
    return decoder, tarfile.open(fileobj=decoder.stdout, mode="r|")


def _finish_decoder(decoder: subprocess.Popen[bytes]) -> None:
    decoder_stderr = decoder.communicate()[1]
    if decoder.returncode != 0:
        raise RuntimeError(
            "zstd archive stream failed: "
            f"{decoder_stderr.decode(errors='replace').strip()}"
        )


def _validate_tar_zst_members(
    archive: Path,
    expected_member_names: set[str] | None = None,
    expected_member_records: dict[str, tuple[int, str]] | None = None,
) -> list[str]:
    if expected_member_records is not None:
        record_names = set(expected_member_records)
        if expected_member_names is not None and expected_member_names != record_names:
            raise ValueError("expected archive member names/records disagree")
        expected_member_names = record_names
    seen: list[str] = []
    seen_set: set[str] = set()
    decoder, tar = _stream_tar_zst(archive)
    try:
        with tar:
            for member in tar:
                _validate_tar_member(member)
                if member.name in seen_set:
                    raise ValueError(f"archive contains duplicate member: {member.name!r}")
                seen_set.add(member.name)
                seen.append(member.name)
                if expected_member_names is not None and member.name not in expected_member_names:
                    raise ValueError(f"archive contains unexpected member: {member.name!r}")
                if expected_member_records is not None:
                    expected_size, expected_sha256 = expected_member_records[member.name]
                    if member.size != expected_size:
                        raise ValueError(
                            f"archive member size mismatch for {member.name}: "
                            f"{member.size} != {expected_size}"
                        )
                    source = tar.extractfile(member)
                    if source is None:
                        raise ValueError(f"cannot read archive member: {member.name!r}")
                    digest = hashlib.sha256()
                    with source:
                        while chunk := source.read(8 * 1024 * 1024):
                            digest.update(chunk)
                    if digest.hexdigest() != expected_sha256:
                        raise ValueError(
                            f"archive member SHA-256 mismatch for {member.name}"
                        )
    except BaseException:
        decoder.kill()
        decoder.wait()
        raise
    finally:
        if decoder.stdout is not None:
            decoder.stdout.close()
    _finish_decoder(decoder)
    if expected_member_names is not None:
        _validate_archive_member_names(seen, expected_member_names)
    return seen


def _extract_tar_zst(
    archive: Path,
    destination: Path,
    expected_member_names: set[str] | None = None,
    expected_member_records: dict[str, tuple[int, str]] | None = None,
) -> None:
    # Validate all headers and the expected member set before creating files.
    _validate_tar_zst_members(
        archive,
        expected_member_names,
        expected_member_records,
    )

    destination.mkdir(parents=True, exist_ok=False)
    decoder, tar = _stream_tar_zst(archive)
    try:
        with tar:
            for member in tar:
                _validate_tar_member(member)
                target = destination / Path(*PurePosixPath(member.name).parts)
                root = destination.resolve()
                resolved_target = target.resolve()
                if root not in resolved_target.parents:
                    raise ValueError(f"archive member escapes restore root: {member.name!r}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                expected = (
                    expected_member_records.get(member.name)
                    if expected_member_records is not None
                    else None
                )
                if expected is not None and member.size != expected[0]:
                    raise ValueError(
                        f"archive member size mismatch for {member.name}: "
                        f"{member.size} != {expected[0]}"
                    )
                with source, target.open("xb") as out:
                    digest = hashlib.sha256()
                    while chunk := source.read(8 * 1024 * 1024):
                        out.write(chunk)
                        digest.update(chunk)
                if expected is not None and digest.hexdigest() != expected[1]:
                    raise ValueError(
                        f"archive member SHA-256 mismatch for {member.name}"
                    )
                os.chmod(target, member.mode & 0o777)
    except BaseException:
        decoder.kill()
        decoder.wait()
        shutil.rmtree(destination, ignore_errors=True)
        raise
    finally:
        if decoder.stdout is not None:
            decoder.stdout.close()
    try:
        _finish_decoder(decoder)
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def _expected_archive_members(
    logical_manifest: dict, logical_manifest_bytes: bytes
) -> dict[str, tuple[int, str]]:
    artifacts = logical_manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("remote logical manifest has no artifacts")
    records = {
        "manifest.json": (
            len(logical_manifest_bytes),
            hashlib.sha256(logical_manifest_bytes).hexdigest(),
        )
    }
    for record in artifacts:
        if not isinstance(record, dict):
            raise ValueError("remote logical manifest artifact records must be objects")
        name = _safe_archive_member_name(str(record.get("path", "")))
        size = record.get("size")
        digest = record.get("sha256")
        if (
            name in records
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(f"invalid remote logical manifest artifact: {name!r}")
        records[name] = (size, digest)
    if len(artifacts) != int(logical_manifest.get("artifact_count", -1)):
        raise ValueError("remote logical manifest artifact_count mismatch")
    if sum(size for name, (size, _digest) in records.items() if name != "manifest.json") != int(
        logical_manifest.get("artifact_bytes", -1)
    ):
        raise ValueError("remote logical manifest artifact_bytes mismatch")
    return records


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
        if manifest["artifact_set_sha256"] != transport["artifact_set_sha256"]:
            raise ValueError("existing destination artifact set does not match transport")
        if _sha256(destination / "manifest.json") != transport["logical_manifest_sha256"]:
            raise ValueError("existing destination logical manifest does not match transport")
        _write_json_atomic(
            destination / "restore_receipt.json",
            {
                "schema": "cppmega_megatron_restore_receipt_v1",
                "status": "already_verified",
                "bundle_id": bundle_id,
                "transport": transport_uri,
                "transport_sha256": hashlib.sha256(transport_bytes).hexdigest(),
                "logical_manifest_sha256": transport["logical_manifest_sha256"],
                "artifact_set_sha256": manifest["artifact_set_sha256"],
                "artifact_count": manifest["artifact_count"],
                "artifact_bytes": manifest["artifact_bytes"],
                "restored_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        print(json.dumps({"bundle": str(destination), "status": "already_verified"}))
        return 0

    partial = output_root / f".{bundle_id}.partial"
    archive = output_root / f".{bundle_id}.tar.zst"
    if partial.exists():
        shutil.rmtree(partial)

    archive_info = transport["archive"]
    _require_free_space(
        output_root,
        artifact_bytes=int(transport["artifact_bytes"]),
        archive_bytes=int(archive_info["size"]),
        headroom_gb=max(0, args.free_space_headroom_gb),
    )
    archive_download = _acquire_archive(
        uri=str(archive_info["uri"]),
        archive=archive,
        endpoint=args.endpoint_url,
        env=env,
        expected_size=int(archive_info["size"]),
        expected_sha256=str(archive_info["sha256"]),
    )

    try:
        expected_members = _expected_archive_members(
            logical_manifest, logical_manifest_bytes
        )
        _extract_tar_zst(
            archive,
            partial,
            expected_member_names=set(expected_members),
            expected_member_records=expected_members,
        )
        manifest, _artifacts = _validate_bundle(partial, args.hash_jobs)
        if manifest["bundle_id"] != bundle_id:
            raise ValueError("restored logical bundle ID does not match transport")
        if manifest["artifact_set_sha256"] != transport["artifact_set_sha256"]:
            raise ValueError("restored artifact set does not match transport")
        if _sha256(partial / "manifest.json") != transport["logical_manifest_sha256"]:
            raise ValueError("restored logical manifest does not match transport")
    except Exception:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    os.replace(partial, destination)
    if not args.keep_archive:
        archive.unlink()
        _archive_receipt_path(archive).unlink(missing_ok=True)
    receipt = {
        "schema": "cppmega_megatron_restore_receipt_v1",
        "status": "restored_verified",
        "bundle_id": bundle_id,
        "transport": transport_uri,
        "transport_sha256": hashlib.sha256(transport_bytes).hexdigest(),
        "logical_manifest_sha256": transport["logical_manifest_sha256"],
        "artifact_set_sha256": manifest["artifact_set_sha256"],
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
        "archive_download": archive_download,
        "restored_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(destination / "restore_receipt.json", receipt)
    print(json.dumps({"bundle": str(destination), "status": "restored_verified"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
