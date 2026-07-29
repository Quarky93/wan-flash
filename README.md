# wan-flash

Greenfield flash attention for **Wan2.1 video-diffusion training** on H100 (sm90),
written in **CuTeDSL** (NVIDIA cutlass-dsl). Not a wrapper: the kernels in
`wan_flash/` are new source, specialized to Wan's exact workload:

- bf16, head_dim 128, MHA (no GQA), non-causal, no mask/dropout/softcap
- self-attention: S_q = S_kv ∈ {32,760 (480p/81f), 75,600 (720p/81f)}, batch 1,
  heads ∈ {12 (1.3B), 40 (14B)}
- cross-attention: S_q × 512 (dense, unmasked umT5 context; I2V adds S_q × 257)
- forward AND backward (training), fp32 accumulation, natural-log LSE
  `(b, h, s)` fp32 output contract (SAC-compatible, drop-in for wan-attn's
  `wan_attention` op)

## Design rule

Every technique choice is a **measured decision**, not an inheritance:
FlashAttention-4's new tricks (CuTeDSL authoring, softmax-rescale skipping,
scheduler design) and FlashAttention-3's Hopper design (warp specialization,
TMA pipelines, pingpong) are implemented behind feature flags in
`wan_flash/features.py` and A/B-tested at Wan shapes by `bench/bench.py`.
Where the FA3-style variant measures faster, the FA3 variant is the default —
`docs/FEATURES.md` records every verdict with numbers.

Baselines benchmarked against: flash-attn-3 3.0.0 (C++ hopper) and
flash-attn-4 4.0.0b23 (CuTeDSL) as installed packages — used for comparison
only; no kernel code is imported from them.

## Layout

```
wan_flash/
  fwd_sm90.py       forward kernel (CuTeDSL, warp-specialized TMA pipeline)
  bwd_sm90.py       backward kernels (preprocess -> main -> postprocess)
  features.py       the A/B feature-flag registry
  interface.py      torch autograd op: wan_flash_attn(q, k, v) -> (o, lse)
  shapes.py         the Wan shape registry + tile arithmetic
tests/              chunked-fp32 oracle + correctness batteries (pytest)
bench/              baseline comparison + feature-matrix runner + perf gate
docs/DESIGN.md      kernel architecture (from FA3/FA4 source study)
docs/FEATURES.md    per-feature A/B verdicts with measurements
```

## Status

**Forward: done and green.** `tests/test_fwd.py` passes (fast battery + true
shapes behind `WAN_FLASH_SLOW_TESTS=1`); error vs the chunked-fp32 oracle sits
at the bf16 floor (rel_l2 ≈ 2.2e-3, same as FA3; LSE ≤ 1.5e-6).

H100 SXM, forward, defaults (persistent scheduler + rescale-skip 8.0 +
intra-WG overlap, 128×128, 2 stages):

| shape | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|
| self h12 S=32760 | 662.5 TFLOP/s | 0.972x | 1.009x |
| self h12 S=75600 | 666.9 TFLOP/s | 0.971x | 1.004x |
| cross h12 S=75600×512 | 536.3 TFLOP/s | 1.003x | 1.175x |
| cross h40 S=75600×512 | 529.8 TFLOP/s | 1.011x | 1.170x |

**Backward: done and green.** `tests/test_bwd.py` passes (fast battery, FA3-fwd
hybrid, autograd chain, true shapes behind `WAN_FLASH_SLOW_TESTS=1`); dq/dk/dv
vs the chunked-fp32 oracle sit at the bf16 floor (rel_l2 ≈ 2.3e-3, gated at
≤ 2x FA3's raw backward on identical inputs). Three-kernel chain
(preprocess → dK/dV-stationary warp-specialized main → dQ convert), FA4's
hd128 non-causal config (80×128, SdP/dQ swapAB, register-resident dK/dV),
packed bf16x2 converts, PDL, and a split-M schedule that fixes the
small-KV cross-attention occupancy pathology FA3/FA4 both have.

H100 SXM, raw backward (identical o/lse/do fed to all three):

| shape | wan-flash | vs FA3 | vs FA4 |
|---|---|---|---|
| self h12 S=32760 | 617.7 TFLOP/s | 1.031x | 0.998x |
| self h40 S=75600 | 614.0 TFLOP/s | 1.030x | 0.981x |
| cross h12 S=75600×512 | 373.8 TFLOP/s | **1.698x** | **1.650x** |
| cross h40 S=75600×512 | 381.4 TFLOP/s | **1.261x** | **1.248x** |

Full 8-shape tables (fwd + bwd) and per-feature verdicts: `docs/FEATURES.md`.

## License / provenance

BSD-3-Clause. Techniques informed by the FlashAttention-2/3/4 papers and by
reading the BSD-licensed FlashAttention CuTeDSL sources; all kernel code here
is written fresh against nvidia-cutlass-dsl 4.6.0.
