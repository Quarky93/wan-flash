"""Feature A/B sweep at true Wan self-attn shapes, vs FA3/FA4 baselines."""
import pathlib
import sys

import torch
from triton.testing import do_bench

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from wan_flash import features
from wan_flash.shapes import WanShape

WARMUP_MS = 200
REP_MS = 1000


def bench_one(shape, feat=None, impl="wan"):
    torch.manual_seed(0)
    q = torch.randn(shape.b, shape.sq, shape.h, shape.d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape.b, shape.skv, shape.h, shape.d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if impl == "wan":
        features.reset()
        if feat:
            features.set_overrides(**feat)
        from wan_flash.interface import wan_flash_fwd as fn
    elif impl == "fa3":
        import flash_attn_interface as fa3
        fn = lambda a, b_, c: fa3._flash_attn_forward(a, b_, c)[:2]
    elif impl == "fa4":
        from flash_attn.cute import interface as fa4i
        fn = lambda a, b_, c: fa4i._flash_attn_fwd(a, b_, c, return_lse=True)[:2]
    fn(q, k, v)
    torch.cuda.synchronize()
    ms = do_bench(lambda: fn(q, k, v), warmup=WARMUP_MS, rep=REP_MS)
    fl = 4.0 * shape.b * shape.h * shape.sq * shape.skv * shape.d
    print(f"{impl:4s} {str(feat):58s} {shape.name:26s} {ms:9.3f} ms {fl/(ms*1e-3)/1e12:6.1f} TFLOP/s", flush=True)
    return ms


if __name__ == "__main__":
    shapes = [WanShape("self", 1, 12, 32760, 32760), WanShape("self", 1, 12, 75600, 75600)]
    configs = sys.argv[1] if len(sys.argv) > 1 else "all"
    for shape in shapes:
        bench_one(shape, None, impl="fa3")
        bench_one(shape, None, impl="fa4")
        for feat in [
            {},
            {"tile_n": 120},
            {"rescale_skip_threshold": 8.0},
            {"tile_n": 120, "rescale_skip_threshold": 8.0},
            {"num_stages": 3},
        ]:
            bench_one(shape, feat)
        print(flush=True)
