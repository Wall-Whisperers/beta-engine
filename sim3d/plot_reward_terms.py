#!/usr/bin/env python3
"""Reward-term decomposition viewer for a sim3d training run.

Reads ``episode_stats.csv`` (written by ``sim3d.train``) and shows how the
episode return splits across reward terms over the course of training. This is
the diagnostic dashboard for the clean-restart reward: the whole point is that
``r_height`` should dominate and stay dominant. If any other column starts
climbing, that's the next exploit surfacing — catch it here before it eats a
training run.

Works headless: it always prints a rolling-mean text table. If ``matplotlib``
is importable and ``--png PATH`` is given, it also writes a stacked plot.

Usage:
    python -m sim3d.plot_reward_terms data/runs/sim3d/<run_id>/episode_stats.csv
    python -m sim3d.plot_reward_terms <csv> --window 50 --png terms.png
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

# The per-term columns written by _EpisodeStatsCallback, in display order.
TERMS = ["r_height", "r_match", "r_reach", "r_finish", "r_fall",
         "r_slip", "r_intersect", "r_energy", "r_other"]


def _read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{csv_path} has no episodes yet.")
    if "r_height" not in rows[0]:
        raise SystemExit(
            f"{csv_path} has no r_* columns — it predates the per-term reward "
            "decomposition. Re-run training to get the diagnostic breakdown."
        )
    return rows


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default) or default)
    except (TypeError, ValueError):
        return default


def summarize(rows: list[dict], window: int, buckets: int) -> None:
    n = len(rows)
    step = max(1, n // max(1, buckets))
    hdr = (["step", "len", "total", "com_z", "succ%"]
           + [t[2:] for t in TERMS])   # strip the "r_" prefix for width
    widths = [9, 6, 8, 6, 6] + [8] * len(TERMS)
    print("  ".join(h.rjust(w) for h, w in zip(hdr, widths)))
    print("-" * (sum(widths) + 2 * len(widths)))

    for end in list(range(step, n, step)) + [n]:
        lo = max(0, end - window)
        win = rows[lo:end]
        if not win:
            continue
        total_steps = int(_f(win[-1], "total_steps"))
        avg_len = sum(_f(r, "length") for r in win) / len(win)
        avg_total = sum(_f(r, "reward") for r in win) / len(win)
        avg_comz = sum(_f(r, "final_com_z") for r in win) / len(win)
        succ = 100.0 * sum(
            1 for r in win if r.get("outcome") == "completed"
        ) / len(win)
        cells = [f"{total_steps:,}", f"{avg_len:.0f}",
                 f"{avg_total:+.1f}", f"{avg_comz:.2f}", f"{succ:.0f}"]
        for t in TERMS:
            cells.append(f"{sum(_f(r, t) for r in win) / len(win):+.1f}")
        print("  ".join(c.rjust(w) for c, w in zip(cells, widths)))

    # Farming check. Only POSITIVE shaping terms can be farmed for free reward;
    # physics gates (slip/intersect/energy) are penalties (≤0) and the terminals
    # (fall/finish) are not shaping — exclude all of them. The farm signature is
    # a non-height term paying out positive while the body stays low.
    tail = rows[-window:]
    shaping = {t: sum(_f(r, t) for r in tail) / len(tail)
               for t in ("r_match", "r_other")}
    worst = max(shaping, key=lambda k: shaping[k])
    h_mean = sum(_f(r, "r_height") for r in tail) / len(tail)
    comz = sum(_f(r, "final_com_z") for r in tail) / len(tail)
    print()
    print(f"last {len(tail)} eps:  r_height={h_mean:+.2f}  "
          f"r_match={shaping['r_match']:+.2f}  r_other={shaping['r_other']:+.2f}  "
          f"com_z={comz:.2f}m")
    if shaping[worst] > max(2.0, abs(h_mean)) and comz < 1.0:
        print(f"  ⚠ {worst}={shaping[worst]:+.2f} pays positive while com_z stays "
              f"low — possible farming, inspect.")
    else:
        print("  ✓ height progress is the dominant shaping signal (no farm).")


def save_png(rows: list[dict], png_path: Path, window: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed — skipping PNG, text table above.)")
        return

    def rolling(key: str) -> list[float]:
        vals = [_f(r, key) for r in rows]
        out, acc = [], 0.0
        from collections import deque
        dq: deque[float] = deque(maxlen=window)
        for v in vals:
            dq.append(v)
            out.append(sum(dq) / len(dq))
        return out

    xs = [int(_f(r, "total_steps")) for r in rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    for t in TERMS:
        ax.plot(xs, rolling(t), label=t, linewidth=1.4)
    ax.plot(xs, rolling("reward"), label="total", color="black",
            linewidth=2.0, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.6)
    ax.set_xlabel("env steps")
    ax.set_ylabel(f"reward (rolling mean, window={window} eps)")
    ax.set_title(f"Reward-term decomposition — {png_path.parent.name}")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"saved {png_path}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim3d.plot_reward_terms")
    p.add_argument("csv", help="path to a run's episode_stats.csv")
    p.add_argument("--window", type=int, default=50,
                   help="rolling-mean window in episodes (default 50)")
    p.add_argument("--buckets", type=int, default=20,
                   help="rows in the text table (default 20)")
    p.add_argument("--png", default=None, help="also save a stacked PNG here")
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f"no such file: {csv_path}")
    rows = _read_rows(csv_path)
    summarize(rows, window=args.window, buckets=args.buckets)
    if args.png:
        save_png(rows, Path(args.png), window=args.window)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
