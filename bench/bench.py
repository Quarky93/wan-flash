"""Benchmark wan-flash kernels vs FA3/FA4 baselines at Wan shapes, and run the
feature A/B matrix.

Usage:
  python -m bench.bench --impl fa3 fa4 wan --shapes primary --modes fwd bwd
  python -m bench.bench --matrix               # feature-flag A/B sweep for ours
"""

import argparse
import json
import pathlib
import time

import torch
from triton.testing import do_bench

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wan_flash.shapes import ALL_SHAPES, PRIMARY, WanShape

WARMUP_MS = 200
REP_MS = 1000


def make_qkv(shape: WanShape, requires_grad=False, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(shape.b, shape.sq, shape.h, shape.d, device="cuda",
                    dtype=torch.bfloat16, requires_grad=requires_grad)
    k = torch.randn(shape.b, shape.skv, shape.h, shape.d, device="cuda",
                    dtype=torch.bfloat16, requires_grad=requires_grad)
    v = torch.randn_like(k, requires_grad=requires_grad)
    return q, k, v


def get_impl(name, feature_overrides=None):
    """Returns fwd_fn(q, k, v) -> (o, lse) in (b, s, h, d) layout."""
    if name == "fa3":
        import flash_attn_interface as fa3

        return lambda q, k, v: fa3._flash_attn_forward(q, k, v)[:2]
    if name == "fa4":
        from flash_attn.cute import interface as fa4i

        return lambda q, k, v: fa4i._flash_attn_fwd(q, k, v, return_lse=True)[:2]
    if name == "wan":
        from wan_flash.interface import wan_flash_fwd

        if feature_overrides:
            from wan_flash import features

            features.set_overrides(**feature_overrides)
        return lambda q, k, v: wan_flash_fwd(q, k, v)
    raise ValueError(name)


def tflops(shape, mode, ms):
    fl = 4.0 * shape.b * shape.h * shape.sq * shape.skv * shape.d
    if mode == "bwd":
        fl *= 2.5
    return fl / (ms * 1e-3) / 1e12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", nargs="*", default=["fa3", "fa4", "wan"])
    ap.add_argument("--shapes", default="primary", choices=["primary", "all", "self12"])
    ap.add_argument("--modes", nargs="*", default=["fwd"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    shapes = {"primary": PRIMARY, "all": ALL_SHAPES,
              "self12": [s for s in PRIMARY if s.kind == "self"]}[args.shapes]
    rows = []
    for shape in shapes:
        base_ms = None
        for name in args.impl:
            try:
                fn = get_impl(name)
                q, k, v = make_qkv(shape)
                fn(q, k, v)  # warmup/JIT
                torch.cuda.synchronize()
                ms = do_bench(lambda: fn(q, k, v), warmup=WARMUP_MS, rep=REP_MS)
                if name == "fa3":
                    base_ms = ms
                sp = f" ({base_ms / ms:5.3f}x)" if base_ms and name != "fa3" else ""
                print(f"{name:6s} {shape.name:26s} fwd {ms:9.3f} ms "
                      f"{tflops(shape, 'fwd', ms):6.1f} TFLOP/s{sp}")
                rows.append({"impl": name, "shape": shape.name, "mode": "fwd", "ms": ms})
            except Exception as e:
                print(f"{name:6s} {shape.name:26s} fwd FAILED: {type(e).__name__}: {str(e)[:120]}")
        print()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)
    (out / f"bench_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
