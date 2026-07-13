#!/usr/bin/env python3
"""Publish an immutable, hashed Megatron bundle to Nebius Object Storage.

Every artifact is uploaded under ``<prefix>/bundles/<bundle_id>/`` with its
SHA-256 in object metadata and verified by HEAD.  The bundle manifest is
uploaded only after all artifacts verify; ``latest.json`` is the final small
commit pointer.  With ``--archive``, the exact same logical bundle is published
as one validated tar.zst object under ``<prefix>/transports/<bundle_id>/`` and
``latest_transport.json`` is committed last.  Consumers must ignore any bundle
lacking its manifest or transport descriptor.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import tempfile
from typing import Iterable


DEFAULT_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
DEFAULT_BUCKET = "cppmega-sidecar-20260627"
DEFAULT_PREFIX = "cppmega-megatron/macro-routes"


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() and key.strip() not in os.environ:
            os.environ[key.strip()] = value.strip().strip('"').strip("'")


def _s3_env() -> dict[str, str]:
    env = os.environ.copy()
    access = env.get("NEBIUS_S3_ACCESS_KEY_ID") or env.get("AWS_ACCESS_KEY_ID")
    secret = env.get("NEBIUS_S3_SECRET_ACCESS_KEY") or env.get("AWS_SECRET_ACCESS_KEY")
    if not access or not secret:
        raise SystemExit("missing Nebius S3 access/secret credentials")
    env["AWS_ACCESS_KEY_ID"] = access
    env["AWS_SECRET_ACCESS_KEY"] = secret
    return env


def _safe_artifact_path(bundle: Path, relative: str) -> Path:
    posix = PurePosixPath(relative)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"unsafe artifact path in bundle manifest: {relative!r}")
    path = (bundle / Path(*posix.parts)).resolve()
    root = bundle.resolve()
    if path != root and root not in path.parents:
        raise ValueError(f"artifact path escapes bundle: {relative!r}")
    return path


def _load_bundle_manifest(bundle: Path) -> tuple[dict, list[dict]]:
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cppmega_megatron_bundle_v1":
        raise ValueError(f"unsupported bundle schema: {manifest.get('schema')!r}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("bundle manifest has no artifacts")
    if len(artifacts) != int(manifest.get("artifact_count", -1)):
        raise ValueError("bundle artifact_count does not match artifact list")
    seen: set[str] = set()
    for record in artifacts:
        relative = str(record.get("path"))
        if relative in seen:
            raise ValueError(f"duplicate artifact path: {relative}")
        seen.add(relative)
        _safe_artifact_path(bundle, relative)
    if sum(int(record["size"]) for record in artifacts) != int(manifest["artifact_bytes"]):
        raise ValueError("bundle artifact_bytes does not match artifact list")
    canonical = [
        {
            "path": str(record["path"]),
            "size": int(record["size"]),
            "sha256": str(record["sha256"]),
        }
        for record in sorted(artifacts, key=lambda item: str(item["path"]))
    ]
    artifact_set_sha256 = hashlib.sha256(
        json.dumps(canonical, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    if manifest.get("artifact_set_sha256") != artifact_set_sha256:
        raise ValueError("bundle artifact_set_sha256 does not match artifact list")
    if not str(manifest.get("bundle_id", "")).endswith(artifact_set_sha256[:16]):
        raise ValueError("bundle_id is not bound to artifact_set_sha256")
    return manifest, artifacts


def _validate_bundle(bundle: Path, hash_jobs: int) -> tuple[dict, list[dict]]:
    manifest, artifacts = _load_bundle_manifest(bundle)

    def validate(record: dict) -> dict:
        relative = str(record["path"])
        path = _safe_artifact_path(bundle, relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        if size != int(record["size"]):
            raise ValueError(f"artifact size mismatch for {relative}: {size} != {record['size']}")
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"artifact sha256 mismatch for {relative}")
        return {**record, "local_path": str(path)}

    with ThreadPoolExecutor(max_workers=max(1, hash_jobs)) as pool:
        validated = list(pool.map(validate, artifacts))
    return manifest, validated


def _validate_archive_member_names(
    member_names: list[str], expected_names: set[str]
) -> None:
    if len(member_names) != len(set(member_names)):
        raise ValueError("archive contains duplicate member names")
    actual = set(member_names)
    missing = sorted(expected_names - actual)
    extra = sorted(actual - expected_names)
    if missing or extra:
        raise ValueError(
            f"archive member set mismatch: missing={missing[:5]} extra={extra[:5]}"
        )


def _validate_archive(
    *, bundle: Path, archive: Path, manifest: dict
) -> tuple[int, str]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    local_manifest = (bundle / "manifest.json").read_bytes()
    expected = {
        str(record["path"]): (int(record["size"]), str(record["sha256"]))
        for record in manifest["artifacts"]
    }
    expected["manifest.json"] = (
        len(local_manifest),
        hashlib.sha256(local_manifest).hexdigest(),
    )
    seen: list[str] = []
    decoder = subprocess.Popen(
        ["zstd", "-dc", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    assert decoder.stdout is not None
    try:
        with tarfile.open(fileobj=decoder.stdout, mode="r|") as tar:
            for member in tar:
                if not member.isfile():
                    raise ValueError(f"archive contains non-file member: {member.name!r}")
                seen.append(member.name)
                expected_record = expected.get(member.name)
                if expected_record is None:
                    raise ValueError(f"archive contains unexpected member: {member.name!r}")
                expected_size, expected_sha256 = expected_record
                if member.size != expected_size:
                    raise ValueError(
                        f"archive member size mismatch for {member.name}: "
                        f"{member.size} != {expected_size}"
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise ValueError(f"cannot read archive member: {member.name!r}")
                digest = hashlib.sha256()
                with extracted:
                    while chunk := extracted.read(8 * 1024 * 1024):
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
        decoder.stdout.close()
    decoder_stderr = decoder.communicate()[1]
    if decoder.returncode != 0:
        raise RuntimeError(
            "zstd archive validation failed: "
            f"{decoder_stderr.decode(errors='replace').strip()}"
        )
    _validate_archive_member_names(seen, set(expected))
    return archive.stat().st_size, _sha256(archive)


def _head(
    *,
    endpoint: str,
    bucket: str,
    key: str,
    env: dict[str, str],
) -> dict | None:
    cmd = [
        "aws",
        "s3api",
        "head-object",
        "--bucket",
        bucket,
        "--key",
        key,
        "--endpoint-url",
        endpoint,
        "--output",
        "json",
    ]
    result = subprocess.run(
        cmd, env=env, text=True, capture_output=True, check=False
    )
    if result.returncode != 0:
        error = result.stderr.lower()
        if any(marker in error for marker in ("404", "not found", "nosuchkey")):
            return None
        raise RuntimeError(
            f"remote HEAD failed ({result.returncode}) for s3://{bucket}/{key}: "
            f"{result.stderr.strip()}"
        )
    return json.loads(result.stdout) if result.stdout.strip() else {}


def _head_matches(head: dict | None, *, size: int, sha256: str) -> bool:
    if not head or int(head.get("ContentLength", -1)) != size:
        return False
    metadata = {
        str(key).lower(): value for key, value in (head.get("Metadata") or {}).items()
    }
    return metadata.get("sha256") == sha256


def _upload_file(
    *,
    local: Path,
    endpoint: str,
    bucket: str,
    key: str,
    size: int,
    sha256: str,
    env: dict[str, str],
    dry_run: bool,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    if not dry_run:
        head = _head(endpoint=endpoint, bucket=bucket, key=key, env=env)
        if _head_matches(head, size=size, sha256=sha256):
            return {"key": key, "size": size, "sha256": sha256, "status": "already_verified"}
        if head is not None and not allow_overwrite:
            remote_metadata = {
                str(key).lower(): value
                for key, value in (head.get("Metadata") or {}).items()
            }
            raise RuntimeError(
                f"immutable remote object mismatch for s3://{bucket}/{key}: "
                f"size={head.get('ContentLength')} sha256={remote_metadata.get('sha256')}; "
                f"local size={size} sha256={sha256}"
            )
    if dry_run:
        return {"key": key, "size": size, "sha256": sha256, "status": "dry_run"}
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(local),
            f"s3://{bucket}/{key}",
            "--endpoint-url",
            endpoint,
            "--metadata",
            f"sha256={sha256}",
            "--only-show-errors",
            "--no-progress",
        ],
        env=env,
        check=True,
    )
    head = _head(endpoint=endpoint, bucket=bucket, key=key, env=env)
    if not _head_matches(head, size=size, sha256=sha256):
        raise RuntimeError(f"remote verification failed for s3://{bucket}/{key}")
    return {"key": key, "size": size, "sha256": sha256, "status": "uploaded_verified"}


def _publish_json(
    *,
    payload: dict,
    key: str,
    endpoint: str,
    bucket: str,
    env: dict[str, str],
    dry_run: bool,
    allow_overwrite: bool = False,
) -> dict[str, object]:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    digest = hashlib.sha256(data).hexdigest()
    if dry_run:
        return {"key": key, "size": len(data), "sha256": digest, "status": "dry_run"}
    with tempfile.NamedTemporaryFile(prefix="cppmega-s3-", suffix=".json") as fh:
        fh.write(data)
        fh.flush()
        return _upload_file(
            local=Path(fh.name),
            endpoint=endpoint,
            bucket=bucket,
            key=key,
            size=len(data),
            sha256=digest,
            env=env,
            dry_run=False,
            allow_overwrite=allow_overwrite,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--endpoint-url", default=DEFAULT_ENDPOINT)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--hash-jobs", type=int, default=4)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument(
        "--archive",
        type=Path,
        help="publish this exact manifest-bound tar.zst instead of loose objects",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    bundle = args.bundle.resolve()
    _load_env_file(args.env_file)
    env = os.environ.copy() if args.dry_run else _s3_env()
    if args.archive is None:
        manifest, artifacts = _validate_bundle(bundle, args.hash_jobs)
    else:
        manifest, artifacts = _load_bundle_manifest(bundle)
    bundle_id = str(manifest["bundle_id"])

    if args.archive is not None:
        archive = args.archive.resolve()
        archive_size, archive_sha256 = _validate_archive(
            bundle=bundle, archive=archive, manifest=manifest
        )
        transport_base = f"{args.prefix.strip('/')}/transports/{bundle_id}"
        archive_key = f"{transport_base}/bundle.tar.zst"
        archive_record = _upload_file(
            local=archive,
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            key=archive_key,
            size=archive_size,
            sha256=archive_sha256,
            env=env,
            dry_run=args.dry_run,
        )
        logical_manifest_path = bundle / "manifest.json"
        logical_manifest_sha256 = _sha256(logical_manifest_path)
        logical_manifest_key = f"{transport_base}/logical_manifest.json"
        logical_manifest_record = _upload_file(
            local=logical_manifest_path,
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            key=logical_manifest_key,
            size=logical_manifest_path.stat().st_size,
            sha256=logical_manifest_sha256,
            env=env,
            dry_run=args.dry_run,
        )
        transport = {
            "schema": "cppmega_megatron_bundle_transport_v1",
            "bundle_id": bundle_id,
            "logical_manifest_sha256": logical_manifest_sha256,
            "artifact_set_sha256": manifest["artifact_set_sha256"],
            "artifact_count": manifest["artifact_count"],
            "artifact_bytes": manifest["artifact_bytes"],
            "logical_manifest": {
                "uri": f"s3://{args.bucket}/{logical_manifest_key}",
                "size": logical_manifest_path.stat().st_size,
                "sha256": logical_manifest_sha256,
            },
            "archive": {
                "uri": f"s3://{args.bucket}/{archive_key}",
                "size": archive_size,
                "sha256": archive_sha256,
                "format": "tar.zst",
            },
        }
        transport_record = _publish_json(
            payload=transport,
            key=f"{transport_base}/transport.json",
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            env=env,
            dry_run=args.dry_run,
        )
        latest_transport = {
            "schema": "cppmega_megatron_latest_transport_v1",
            "bundle_id": bundle_id,
            "transport": f"s3://{args.bucket}/{transport_base}/transport.json",
            "transport_sha256": transport_record["sha256"],
            "archive": transport["archive"],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        latest_record = _publish_json(
            payload=latest_transport,
            key=f"{args.prefix.strip('/')}/latest_transport.json",
            endpoint=args.endpoint_url,
            bucket=args.bucket,
            env=env,
            dry_run=args.dry_run,
            allow_overwrite=True,
        )
        receipt_payload = {
            "schema": "cppmega_megatron_archive_publish_receipt_v1",
            "endpoint_url": args.endpoint_url,
            "bucket": args.bucket,
            "archive_validation": {
                "status": "verified",
                "member_count": len(artifacts) + 1,
                "artifact_set_sha256": manifest["artifact_set_sha256"],
                "logical_manifest_sha256": logical_manifest_sha256,
            },
            "archive": archive_record,
            "logical_manifest": logical_manifest_record,
            "transport": transport_record,
            "latest_transport": latest_record,
            "dry_run": args.dry_run,
        }
        receipt_path = bundle / "archive_publish_receipt.json"
        receipt_path.write_text(
            json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {"receipt": str(receipt_path), "latest_transport": latest_transport},
                indent=2,
            )
        )
        return 0

    base_key = f"{args.prefix.strip('/')}/bundles/{bundle_id}"

    receipts: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [
            pool.submit(
                _upload_file,
                local=Path(record["local_path"]),
                endpoint=args.endpoint_url,
                bucket=args.bucket,
                key=f"{base_key}/{record['path']}",
                size=int(record["size"]),
                sha256=str(record["sha256"]),
                env=env,
                dry_run=args.dry_run,
            )
            for record in artifacts
        ]
        for future in as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(json.dumps(receipt, sort_keys=True), flush=True)

    manifest_record = _publish_json(
        payload=manifest,
        key=f"{base_key}/manifest.json",
        endpoint=args.endpoint_url,
        bucket=args.bucket,
        env=env,
        dry_run=args.dry_run,
    )
    latest = {
        "schema": "cppmega_megatron_latest_v1",
        "bundle_id": bundle_id,
        "manifest": f"s3://{args.bucket}/{base_key}/manifest.json",
        "manifest_sha256": manifest_record["sha256"],
        "artifact_count": manifest["artifact_count"],
        "artifact_bytes": manifest["artifact_bytes"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    latest_record = _publish_json(
        payload=latest,
        key=f"{args.prefix.strip('/')}/latest.json",
        endpoint=args.endpoint_url,
        bucket=args.bucket,
        env=env,
        dry_run=args.dry_run,
        allow_overwrite=True,
    )
    receipt_payload = {
        "schema": "cppmega_megatron_publish_receipt_v1",
        "endpoint_url": args.endpoint_url,
        "bucket": args.bucket,
        "base_key": base_key,
        "artifacts": sorted(receipts, key=lambda item: str(item["key"])),
        "manifest": manifest_record,
        "latest": latest_record,
        "dry_run": args.dry_run,
    }
    receipt_path = bundle / "publish_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"receipt": str(receipt_path), "latest": latest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
