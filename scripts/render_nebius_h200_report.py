#!/usr/bin/env python3
from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reports" / "nebius_h200_100step_report.html"

RUNS = {
    "BF16": [
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782680290",
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782681292",
    ],
    "FP8 tensorwise + MLP recompute": [
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782682254",
    ],
    "FP8 tensorwise + MLP recompute + NVRTC-off": [
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782692003",
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782692760",
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782693401",
    ],
    "FP8 tensorwise": [
        ROOT / "outputs" / "nebius" / "cppmega-h200-megatron-1782683923",
    ],
}


@dataclass(frozen=True)
class Row:
    mode: str
    run: str
    log: Path
    seq: int
    bs: int
    status: str
    train_loss: float | None
    val_loss: float | None
    test_loss: float | None
    peak_mib: int | None
    cuda_alloc_gib: float | None
    cuda_reserved_gib: float | None
    iter_ms: float | None
    fp8_seen: bool
    sidecar_seen: bool
    backend: str | None

    @property
    def train_ppl(self) -> float | None:
        return _ppl(self.train_loss)

    @property
    def val_ppl(self) -> float | None:
        return _ppl(self.val_loss)

    @property
    def test_ppl(self) -> float | None:
        return _ppl(self.test_loss)

    @property
    def peak_gib(self) -> float | None:
        return None if self.peak_mib is None else self.peak_mib / 1024

    @property
    def tokens_per_sec(self) -> float | None:
        if self.iter_ms is None:
            return None
        return self.seq * self.bs / (self.iter_ms / 1000)

    @property
    def okish(self) -> bool:
        return self.status == "OK"


def _sci(mant: str, exp: str) -> float:
    return float(mant) * (10 ** int(exp))


def _ppl(loss: float | None) -> float | None:
    if loss is None or loss > 50:
        return None
    return math.exp(loss)


def _fmt(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _read_rows() -> list[Row]:
    rows: list[Row] = []
    for mode, dirs in RUNS.items():
        for run_dir in dirs:
            for path in sorted(run_dir.glob("seq_*_bs_*.log")):
                match = re.match(r"seq_(\d+)_bs_(\d+)\.log", path.name)
                if not match:
                    continue
                text = path.read_text(errors="replace")
                seq = int(match.group(1))
                bs = int(match.group(2))
                result = re.findall(
                    r"CPPMEGA_BATCH_RESULT\s+seq=\d+\s+batch=\d+\s+status=([^\s]+)",
                    text,
                )
                status = result[-1] if result else "FAIL"
                if status == "FAIL" and re.search(
                    r"out of memory|OutOfMemoryError|failed to CUDA calloc|CUDA calloc async",
                    text,
                    re.I,
                ):
                    status = "OOM"

                train_loss = None
                for mant, exp in re.findall(
                    r"iteration\s+100/\s+100 .*?lm loss:\s*([0-9.]+)E([+-]\d+)",
                    text,
                ):
                    train_loss = _sci(mant, exp)

                val_loss = None
                test_loss = None
                for split, mant, exp in re.findall(
                    r"validation loss at iteration 100 on (validation|test) set \| "
                    r"lm loss value:\s*([0-9.]+)E([+-]\d+)",
                    text,
                ):
                    if split == "validation":
                        val_loss = _sci(mant, exp)
                    else:
                        test_loss = _sci(mant, exp)

                peak_mib = None
                peak_match = re.findall(
                    r"CPPMEGA_NVIDIA_SMI_PEAK\s+seq=\d+\s+batch=\d+\s+peak_used_mib=(\d+)",
                    text,
                )
                if peak_match:
                    peak_mib = int(peak_match[-1])

                cuda_alloc = None
                cuda_reserved = None
                cuda_match = re.findall(
                    r"CPPMEGA_CUDA_PEAK\s+allocated_gib=([0-9.]+)\s+reserved_gib=([0-9.]+)",
                    text,
                )
                if cuda_match:
                    cuda_alloc, cuda_reserved = map(float, cuda_match[-1])

                iter_ms = None
                iter_match = re.findall(
                    r"iteration\s+100/\s+100 .*?elapsed time per iteration \(ms\):\s*([0-9.]+)",
                    text,
                )
                if iter_match:
                    iter_ms = float(iter_match[-1])

                backend = None
                backend_match = re.findall(r"Running with ([A-Za-z]+Attention backend[^\n]*)", text)
                if backend_match:
                    backend = backend_match[-1]

                rows.append(
                    Row(
                        mode=mode,
                        run=run_dir.name,
                        log=path,
                        seq=seq,
                        bs=bs,
                        status=status,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        test_loss=test_loss,
                        peak_mib=peak_mib,
                        cuda_alloc_gib=cuda_alloc,
                        cuda_reserved_gib=cuda_reserved,
                        iter_ms=iter_ms,
                        fp8_seen=(
                            "fp8_recipe tensorwise" in text
                            or "fp8_recipe='tensorwise'" in text
                            or "recipe_type=Float8CurrentScaling" in text
                        ),
                        sidecar_seen=all(
                            token in text
                            for token in ("token_structure_ids", "token_call_edges", "token_type_edges")
                        ),
                        backend=backend,
                    )
                )
    return sorted(rows, key=lambda item: (item.mode, item.seq, item.bs, item.run))


def _status_label(status: str) -> str:
    return {
        "OK": "OK",
        "OK_TE_CLEANUP_SIGSEGV": "100 steps + TE cleanup crash",
        "OOM": "OOM",
        "SKIP": "Skipped",
        "FAIL": "Fail",
    }.get(status, status)


def _status_class(status: str) -> str:
    if status == "OK":
        return "ok"
    if status == "OK_TE_CLEANUP_SIGSEGV":
        return "warn"
    if status == "OOM":
        return "oom"
    if status == "SKIP":
        return "skip"
    return "fail"


def _bar_chart(rows: list[Row], title: str, value_fn, unit: str, max_value: float | None = None) -> str:
    data = [(row, value_fn(row)) for row in rows if value_fn(row) is not None]
    if not data:
        return ""
    max_v = max_value or max(float(value) for _, value in data)
    bars = []
    for row, value in data:
        pct = max(3.0, min(100.0, float(value) / max_v * 100))
        cls = _status_class(row.status)
        label = f"{row.mode} {row.seq}/{row.bs}"
        bars.append(
            f"""
            <div class="bar-row">
              <div class="bar-label">{html.escape(label)}</div>
              <div class="bar-track"><div class="bar-fill {cls}" style="width:{pct:.2f}%"></div></div>
              <div class="bar-value">{_fmt(float(value), 1)} {unit}</div>
            </div>
            """
        )
    return f"""
    <section class="panel">
      <h2>{html.escape(title)}</h2>
      <div class="bar-chart">{''.join(bars)}</div>
    </section>
    """


def _line_chart(rows: list[Row]) -> str:
    success = [row for row in rows if row.okish and row.train_loss is not None]
    if not success:
        return ""
    width = 920
    height = 260
    pad_l = 58
    pad_r = 24
    pad_t = 26
    pad_b = 48
    min_loss = min(row.train_loss for row in success if row.train_loss is not None)
    max_loss = max(row.train_loss for row in success if row.train_loss is not None)
    span = max(max_loss - min_loss, 0.1)
    groups = {}
    for mode in sorted({row.mode for row in success}):
        points = []
        series = [row for row in success if row.mode == mode]
        for idx, row in enumerate(series):
            x = pad_l + idx * ((width - pad_l - pad_r) / max(1, len(series) - 1))
            y = pad_t + (max_loss - row.train_loss) / span * (height - pad_t - pad_b)
            points.append((x, y, row))
        groups[mode] = points

    colors = {
        "BF16": "#176b87",
        "FP8 tensorwise": "#b45f06",
        "FP8 tensorwise + MLP recompute": "#b42318",
        "FP8 tensorwise + MLP recompute + NVRTC-off": "#2c7a47",
    }
    lines = []
    circles = []
    for mode, points in groups.items():
        if len(points) > 1:
            lines.append(
                f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y, _ in points)}" '
                f'fill="none" stroke="{colors.get(mode, "#333")}" stroke-width="3" stroke-linecap="round" />'
            )
        for x, y, row in points:
            circles.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{colors.get(mode, "#333")}">'
                f'<title>{html.escape(mode)} seq={row.seq} bs={row.bs}: loss={row.train_loss:.4f}</title>'
                f"</circle>"
            )

    x_labels = []
    for idx, row in enumerate(success):
        x = pad_l + idx * ((width - pad_l - pad_r) / max(1, len(success) - 1))
        x_labels.append(
            f'<text x="{x:.1f}" y="{height - 13}" text-anchor="middle" class="svg-label">'
            f'{row.seq}/{row.bs}</text>'
        )

    return f"""
    <section class="panel wide">
      <div class="panel-title-row">
        <h2>Training loss after 100 steps</h2>
        <div class="legend"><span><i class="bf16"></i>BF16</span><span><i class="fp8"></i>FP8 tensorwise</span><span><i class="fixed"></i>FP8 + recompute fixed</span></div>
      </div>
      <svg viewBox="0 0 {width} {height}" role="img" aria-label="Training loss line chart">
        <line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" class="axis" />
        <line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" class="axis" />
        <text x="10" y="{pad_t + 4}" class="svg-label">{max_loss:.2f}</text>
        <text x="10" y="{height-pad_b}" class="svg-label">{min_loss:.2f}</text>
        {''.join(lines)}
        {''.join(circles)}
        {''.join(x_labels)}
      </svg>
    </section>
    """


def _capacity_grid(rows: list[Row]) -> str:
    modes = sorted({row.mode for row in rows})
    seqs = sorted({row.seq for row in rows})
    batches = sorted({row.bs for row in rows})
    cells = []
    for mode in modes:
        cells.append(f"<h3>{html.escape(mode)}</h3>")
        cells.append('<table class="matrix"><thead><tr><th>seq \\ BS</th>')
        cells.extend(f"<th>{batch}</th>" for batch in batches)
        cells.append("</tr></thead><tbody>")
        for seq in seqs:
            cells.append(f"<tr><th>{seq}</th>")
            for batch in batches:
                match = next((row for row in rows if row.mode == mode and row.seq == seq and row.bs == batch), None)
                if match is None:
                    cells.append('<td class="na">-</td>')
                    continue
                cls = _status_class(match.status)
                value = _status_label(match.status)
                peak = "" if match.peak_gib is None else f"<small>{match.peak_gib:.1f} GiB</small>"
                cells.append(f'<td class="{cls}">{html.escape(value)}{peak}</td>')
            cells.append("</tr>")
        cells.append("</tbody></table>")
    return f"""
    <section class="panel wide">
      <h2>Capacity matrix</h2>
      <div class="matrix-wrap">{''.join(cells)}</div>
    </section>
    """


def _rows_table(rows: list[Row]) -> str:
    body = []
    for row in rows:
        rel_log = os.path.relpath(row.log, OUT.parent)
        body.append(
            f"""
            <tr>
              <td>{html.escape(row.mode)}</td>
              <td>{row.seq}</td>
              <td>{row.bs}</td>
              <td><span class="pill {_status_class(row.status)}">{html.escape(_status_label(row.status))}</span></td>
              <td>{_fmt(row.train_loss, 4)} / {_fmt(row.train_ppl, 2)}</td>
              <td>{_fmt(row.val_loss, 4)} / {_fmt(row.val_ppl, 2)}</td>
              <td>{_fmt(row.test_loss, 4)} / {_fmt(row.test_ppl, 2)}</td>
              <td>{_fmt(row.peak_gib, 1)} GiB</td>
              <td>{_fmt(row.cuda_alloc_gib, 1)} / {_fmt(row.cuda_reserved_gib, 1)} GiB</td>
              <td>{_fmt(row.iter_ms, 1)}</td>
              <td>{_fmt(row.tokens_per_sec, 0)}</td>
              <td>{'yes' if row.sidecar_seen else 'no'}</td>
              <td><a href="{html.escape(str(rel_log))}">log</a></td>
            </tr>
            """
        )
    return f"""
    <section class="panel wide">
      <h2>Run table</h2>
      <table class="runs">
        <thead>
          <tr>
            <th>Mode</th><th>Seq</th><th>BS</th><th>Status</th>
            <th>Train loss / PPL</th><th>Val loss / PPL</th><th>Test loss / PPL</th>
            <th>Peak</th><th>CUDA alloc/res</th><th>Iter ms</th><th>Tok/s</th><th>Sidecar</th><th>Source</th>
          </tr>
        </thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </section>
    """


def _summary_cards(rows: list[Row]) -> str:
    ok_rows = [row for row in rows if row.okish]
    best_bf16 = max((row for row in ok_rows if row.mode == "BF16"), key=lambda row: (row.seq, row.bs), default=None)
    best_fp8 = max(
        (row for row in ok_rows if row.mode == "FP8 tensorwise + MLP recompute + NVRTC-off"),
        key=lambda row: (row.seq, row.bs),
        default=None,
    )
    sidecar = sum(1 for row in rows if row.sidecar_seen)
    fp8 = sum(1 for row in rows if row.fp8_seen)
    cards = [
        ("BF16 max passing", "-" if best_bf16 is None else f"{best_bf16.seq} / BS {best_bf16.bs}"),
        ("Fixed FP8+recompute max passing", "-" if best_fp8 is None else f"{best_fp8.seq} / BS {best_fp8.bs}"),
        ("Graph sidecar evidence", f"{sidecar}/{len(rows)} log rows"),
        ("FP8 evidence", f"{fp8} logs with TE FP8"),
    ]
    return "".join(
        f'<div class="metric"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong></div>'
        for label, value in cards
    )


def render() -> str:
    rows = _read_rows()
    if not rows:
        raise SystemExit("No H200 run logs found")

    memory_rows = [row for row in rows if row.peak_gib is not None and row.status != "SKIP"]
    success_rows = [row for row in rows if row.okish]
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cppmega H200 100-step report</title>
  <style>
    :root {{
      --bg: #f4f6f6;
      --ink: #121514;
      --muted: #68706d;
      --line: #d8ddda;
      --panel: #ffffff;
      --ok: #2c7a47;
      --warn: #b45f06;
      --oom: #b42318;
      --skip: #6d7370;
      --accent: #176b87;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    a {{ color: var(--accent); text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    .page {{ max-width: 1480px; margin: 0 auto; padding: 36px 28px 56px; }}
    header {{
      display: grid;
      grid-template-columns: 1.6fr 1fr;
      gap: 24px;
      align-items: end;
      padding-bottom: 24px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{ margin: 0; font-size: 44px; line-height: 1.02; letter-spacing: 0; }}
    h2 {{ margin: 0 0 16px; font-size: 18px; letter-spacing: 0; }}
    h3 {{ margin: 18px 0 8px; font-size: 14px; color: var(--muted); }}
    .lede {{ margin: 12px 0 0; max-width: 82ch; color: var(--muted); font-size: 16px; }}
    .stamp {{ text-align: right; color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    .metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 22px 0; }}
    .metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px; }}
    .metric span {{ display: block; color: var(--muted); font-size: 12px; }}
    .metric strong {{ display: block; margin-top: 5px; font-size: 22px; line-height: 1.1; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; overflow: hidden; }}
    .wide {{ grid-column: 1 / -1; }}
    .panel-title-row {{ display: flex; justify-content: space-between; gap: 18px; align-items: center; }}
    .legend {{ display: flex; gap: 16px; color: var(--muted); font-size: 12px; }}
    .legend i {{ display: inline-block; width: 12px; height: 12px; margin-right: 6px; border-radius: 3px; vertical-align: -2px; }}
    .legend .bf16 {{ background: #176b87; }}
    .legend .fp8 {{ background: #b45f06; }}
    .legend .fixed {{ background: #2c7a47; }}
    .bar-chart {{ display: grid; gap: 9px; }}
    .bar-row {{ display: grid; grid-template-columns: 168px 1fr 110px; gap: 12px; align-items: center; }}
    .bar-label {{ color: var(--muted); white-space: nowrap; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }}
    .bar-track {{ height: 18px; background: #eef1ef; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: var(--accent); }}
    .bar-fill.ok {{ background: var(--ok); }}
    .bar-fill.warn {{ background: var(--warn); }}
    .bar-fill.oom {{ background: var(--oom); }}
    .bar-fill.fail {{ background: var(--oom); }}
    .bar-value {{ text-align: right; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    svg {{ width: 100%; height: auto; display: block; }}
    .axis {{ stroke: var(--line); stroke-width: 1; }}
    .svg-label {{ fill: var(--muted); font: 11px ui-monospace, SFMono-Regular, Menlo, monospace; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 9px 10px; text-align: left; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 650; }}
    td {{ font-variant-numeric: tabular-nums; }}
    .runs td:nth-child(2), .runs td:nth-child(3), .runs td:nth-child(8), .runs td:nth-child(10), .runs td:nth-child(11) {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; font-size: 12px; border: 1px solid currentColor; white-space: nowrap; }}
    .pill.ok, td.ok {{ color: var(--ok); }}
    .pill.warn, td.warn {{ color: var(--warn); }}
    .pill.oom, .pill.fail, td.oom, td.fail {{ color: var(--oom); }}
    .pill.skip, td.skip, td.na {{ color: var(--skip); }}
    .matrix-wrap {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .matrix td, .matrix th {{ text-align: center; }}
    .matrix td {{ border: 1px solid var(--line); border-radius: 4px; font-size: 12px; }}
    .matrix small {{ display: block; margin-top: 3px; color: var(--muted); }}
    .note {{
      margin-top: 16px;
      padding: 14px 16px;
      border-left: 4px solid var(--warn);
      background: #fff8ee;
      color: #5e3708;
      border-radius: 6px;
    }}
    footer {{ margin-top: 18px; color: var(--muted); font-size: 12px; }}
    @media (max-width: 960px) {{
      header, .grid, .metrics, .matrix-wrap {{ grid-template-columns: 1fr; }}
      .stamp {{ text-align: left; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 4px; }}
      .bar-value {{ text-align: left; }}
      .runs {{ display: block; overflow-x: auto; white-space: nowrap; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <header>
      <div>
        <h1>cppmega H200 100-step report</h1>
        <p class="lede">Real Nebius H200 smoke/capacity sweep for the cppmega Megatron world model using verified code+commit graph sidecar data. No synthetic dataset and no token-only path.</p>
      </div>
      <div class="stamp">
        generated {html.escape(timestamp)}<br>
        source: outputs/nebius/*
      </div>
    </header>

    <section class="metrics">{_summary_cards(rows)}</section>

    <div class="grid">
      {_bar_chart(memory_rows, "Peak H200 memory", lambda row: row.peak_gib, "GiB", 143771 / 1024)}
      {_bar_chart(success_rows, "Throughput at step 100", lambda row: row.tokens_per_sec, "tok/s")}
      {_line_chart(rows)}
      {_capacity_grid(rows)}
      {_rows_table(rows)}
    </div>

    <section class="panel wide">
      <h2>Interpretation</h2>
      <p>The fixed FP8 tensorwise + MLP recompute profile now passes cleanly on a single Nebius H200 for 1024-token context at BS64, BS128 and BS192. The fix has two parts: Megatron TE checkpoint kwargs are bound into the forwarded callable, and FP8 tensorwise sweeps default to <code>NVTE_DISABLE_NVRTC=1</code> to avoid the Transformer Engine RTC teardown crash seen with TE 2.16.0.</p>
      <p>BS192 is the best current 1024-token smoke profile: 100 steps pass with no OOM, no skipped/nan iterations, no <code>Kernel::~Kernel</code> cleanup crash, about 122.9k tok/s over the last 20 iterations, and about 110.9GB nvidia-smi peak. Prior BS256 runs OOM near the H200 memory limit, so BS192 is the stable recommendation for the 10000-step trial.</p>
      <p>For maximum MFU, the remaining performance branch is to rebuild the image with a newer Transformer Engine and retest <code>--enable-nvrtc</code>. On the current image, keeping NVRTC enabled is around 4% faster at BS64 but crashes during TE RTC cleanup after metrics are emitted.</p>
      <div class="note">Do not treat <code>OK_TE_CLEANUP_SIGSEGV</code> as a clean production status. It means 100 training steps and eval completed, then TE crashed during process teardown.</div>
    </section>

    <footer>
      Logs are linked relative to this report. Rebuild with <code>python3 scripts/render_nebius_h200_report.py</code>.
    </footer>
  </main>
</body>
</html>
"""
    return html_doc


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(render())
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
