"""Forward correctness: wan-flash vs the chunked-fp32 oracle, gated at <= 2x
FA3's error on identical inputs (the FA test-suite convention), plus LSE
contract checks (natural log, (b, h, s), fp32 — SAC-compatible)."""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from tests.oracle import error_metrics, oracle_fwd
from wan_flash.shapes import WanShape

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# small shapes for the fast battery; true shapes behind WAN_FLASH_SLOW_TESTS
FAST_SHAPES = [
    WanShape("self", 1, 2, 1024, 1024),
    WanShape("self", 1, 2, 960, 960),      # not a multiple of 128 -> tail path
    WanShape("self", 1, 4, 3000, 3000),    # ragged both dims
    WanShape("cross", 1, 2, 2048, 512),
    WanShape("cross", 1, 2, 2048, 257),    # I2V image branch length
]


def _run(shape, seed=0):
    from wan_flash.interface import wan_flash_fwd
    import flash_attn_interface as fa3

    torch.manual_seed(seed)
    q = torch.randn(shape.b, shape.sq, shape.h, shape.d, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(shape.b, shape.skv, shape.h, shape.d, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)

    o, lse = wan_flash_fwd(q, k, v)
    o3, lse3 = fa3._flash_attn_forward(q, k, v)[:2]

    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    ref_o, ref_lse = oracle_fwd(qh, kh, vh)

    e_ours = error_metrics(o.transpose(1, 2), ref_o)
    e_fa3 = error_metrics(o3.transpose(1, 2), ref_o)
    assert e_ours["finite"], f"non-finite O at {shape.name}"
    assert e_ours["rel_l2"] <= 2.0 * e_fa3["rel_l2"] + 1e-9, (
        f"{shape.name}: ours {e_ours['rel_l2']:.3e} > 2x FA3 {e_fa3['rel_l2']:.3e}"
    )
    # LSE contract
    assert lse.shape == (shape.b, shape.h, shape.sq) and lse.dtype == torch.float32
    assert (lse - ref_lse).abs().max().item() < 1e-3, "LSE must be natural-log, oracle-tight"


@requires_gpu
@pytest.mark.parametrize("shape", FAST_SHAPES, ids=lambda s: s.name)
def test_fwd_fast(shape):
    _run(shape)


@requires_gpu
@pytest.mark.skipif(__import__("os").environ.get("WAN_FLASH_SLOW_TESTS") != "1",
                    reason="WAN_FLASH_SLOW_TESTS=1 for true shapes")
@pytest.mark.parametrize("shape", [WanShape("self", 1, 12, 32760, 32760),
                                   WanShape("self", 1, 12, 75600, 75600)],
                         ids=lambda s: s.name)
def test_fwd_true_shapes(shape):
    _run(shape)
