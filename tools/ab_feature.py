"""Paired ABBA A/B for any feature flag — the protocol behind docs/FEATURES.md.

Why this shape (each guard earned the hard way, see FEATURES.md):
  * PAIRED: both variants are benchmarked back-to-back inside ONE process,
    so clock/power drift is common-mode and cancels in the ratio.
  * ABBA: the within-round order alternates, so any residual ordering bias
    (first-measured runs cooler) cancels across rounds.
  * BURN-IN: the first two rounds are discarded — clocks settle at the power
    wall, and their ratios are visibly noisier.
  * MEDIAN + IQR over rounds, not a single delta: at this margin size a
    single pair is indistinguishable from noise (real wins here are 1-3%,
    round-to-round spread is ~0.5%).

Usage:
  python tools/ab_feature.py bwd cluster_n 2 4            # bwd flag, 2 vs 4
  python tools/ab_feature.py fwd rescale_skip_threshold 0 8
  python tools/ab_feature.py bwd nsplit 1 4 --kind cross --rounds 12
"""
import argparse
import ast
import pathlib
import statistics
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from bench.bench import make_qkv  # noqa: E402
from wan_flash import features  # noqa: E402
from wan_flash.interface import wan_flash_bwd, wan_flash_fwd  # noqa: E402
from wan_flash.shapes import ALL_SHAPES  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pass_", choices=["fwd", "bwd"])
    p.add_argument("flag")
    p.add_argument("value_a")
    p.add_argument("value_b")
    p.add_argument("--kind", default="self", choices=["self", "cross", "all"])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--burn-in", type=int, default=2)
    args = p.parse_args()

    lit = lambda s: ast.literal_eval(s)  # noqa: E731
    a, b = lit(args.value_a), lit(args.value_b)
    setter = features.set_overrides if args.pass_ == "fwd" else features.set_bwd_overrides
    shapes = [s for s in ALL_SHAPES if args.kind == "all" or s.kind == args.kind]
    ratios = {s.name: [] for s in shapes}

    for shape in shapes:
        q, k, v = make_qkv(shape)
        o, lse = wan_flash_fwd(q, k, v)
        do = torch.randn_like(o)
        run = (lambda: wan_flash_fwd(q, k, v)) if args.pass_ == "fwd" else (
            lambda: wan_flash_bwd(q, k, v, o, lse, do)
        )
        for val in (a, b):  # warm both compile-cache entries
            setter(**{args.flag: val})
            run()
        torch.cuda.synchronize()

        for r in range(args.rounds):
            order = (a, b) if r % 2 == 0 else (b, a)
            ms = {}
            for val in order:
                setter(**{args.flag: val})
                ms[val] = do_bench(run, warmup=200, rep=500)
            if r >= args.burn_in:
                ratios[shape.name].append(ms[a] / ms[b])
            tag = " (burn-in)" if r < args.burn_in else ""
            print(f"r{r} {shape.name:28s} {args.flag}={a!r} {ms[a]:8.3f}  "
                  f"={b!r} {ms[b]:8.3f}  ratio {ms[a] / ms[b]:.4f}x{tag}",
                  flush=True)

    print(f"\n== paired ratio {a!r}/{b!r} (>1 = {b!r} faster) ==")
    for name, rs in ratios.items():
        if not rs:
            continue
        qs = statistics.quantiles(rs, n=4) if len(rs) >= 4 else [min(rs), 0, max(rs)]
        print(f"{name:28s} median {statistics.median(rs):.4f}x  "
              f"IQR [{qs[0]:.4f}, {qs[2]:.4f}]  n={len(rs)}")
    features.reset()


if __name__ == "__main__":
    main()
