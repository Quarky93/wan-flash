"""Inner-loop dev harness: build + run the kernel on a small shape vs oracle."""
import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests.oracle import error_metrics, oracle_fwd


def run(b=1, h=2, sq=1024, skv=None, seed=0, quiet=False, **feat):
    from wan_flash import features
    features.reset()
    if feat:
        features.set_overrides(**feat)
    from wan_flash.interface import wan_flash_fwd

    skv = skv if skv is not None else sq
    torch.manual_seed(seed)
    q = torch.randn(b, sq, h, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, skv, h, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    o, lse = wan_flash_fwd(q, k, v)
    torch.cuda.synchronize()

    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    ref_o, ref_lse = oracle_fwd(qh, kh, vh)
    e = error_metrics(o.transpose(1, 2), ref_o)
    lse_err = (lse - ref_lse).abs().max().item()
    if not quiet:
        print(f"shape b{b} h{h} sq{sq} skv{skv} feat={feat}")
        print(f"  o rel_l2={e['rel_l2']:.3e} max_abs={e['max_abs']:.3e} finite={e['finite']}")
        print(f"  lse max err={lse_err:.3e}")
    ok = e["finite"] and e["rel_l2"] < 6e-3 and lse_err < 1e-3
    print(("PASS" if ok else "FAIL") + f" b{b} h{h} sq{sq} skv{skv} {feat}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq", type=int, default=1024)
    ap.add_argument("--skv", type=int, default=None)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--tile_n", type=int, default=None)
    ap.add_argument("--rescale_skip", type=float, default=None)
    args = ap.parse_args()
    feat = {}
    if args.tile_n is not None:
        feat["tile_n"] = args.tile_n
    if args.rescale_skip is not None:
        feat["rescale_skip_threshold"] = args.rescale_skip
    # (feature dict keys must match wan_flash.features.FwdFeatures fields)
    run(h=args.h, sq=args.sq, skv=args.skv, **feat)
