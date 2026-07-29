"""Inner-loop dev harness for the backward: build + run vs oracle_bwd."""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests.oracle import error_metrics, oracle_bwd, oracle_fwd


def run(b=1, h=2, sq=1024, skv=None, seed=0, quiet=False, use_fa3_fwd=False, **feat):
    from wan_flash import features
    features.reset()
    if feat:
        features.set_bwd_overrides(**feat)
    from wan_flash.interface import wan_flash_bwd, wan_flash_fwd

    skv = skv if skv is not None else sq
    torch.manual_seed(seed)
    q = torch.randn(b, sq, h, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(b, skv, h, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    do = torch.randn_like(q)

    if use_fa3_fwd:
        import flash_attn_interface as fa3
        o, lse = fa3._flash_attn_forward(q, k, v)[:2]
    else:
        o, lse = wan_flash_fwd(q, k, v)
    dq, dk, dv = wan_flash_bwd(q, k, v, o, lse, do)
    torch.cuda.synchronize()

    qh, kh, vh, oh, doh = (t.transpose(1, 2) for t in (q, k, v, o, do))
    ref_o, ref_lse = oracle_fwd(qh, kh, vh)
    ref_dq, ref_dk, ref_dv = oracle_bwd(qh, kh, vh, ref_o, ref_lse, doh)
    es = {}
    for name, x, ref in (
        ("dq", dq, ref_dq), ("dk", dk, ref_dk), ("dv", dv, ref_dv),
    ):
        es[name] = error_metrics(x.transpose(1, 2), ref)
        if not quiet:
            e = es[name]
            print(f"  {name} rel_l2={e['rel_l2']:.3e} max_abs={e['max_abs']:.3e} "
                  f"finite={e['finite']}")
    ok = all(e["finite"] and e["rel_l2"] < 1.2e-2 for e in es.values())
    print(("PASS" if ok else "FAIL") + f" b{b} h{h} sq{sq} skv{skv} {feat}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--sq", type=int, default=1024)
    ap.add_argument("--skv", type=int, default=None)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--tile_m", type=int, default=None)
    ap.add_argument("--num_stages", type=int, default=None)
    ap.add_argument("--fa3-fwd", action="store_true")
    args = ap.parse_args()
    feat = {}
    if args.tile_m is not None:
        feat["tile_m"] = args.tile_m
    if args.num_stages is not None:
        feat["num_stages"] = args.num_stages
    run(b=args.b, h=args.h, sq=args.sq, skv=args.skv, use_fa3_fwd=args.fa3_fwd, **feat)
