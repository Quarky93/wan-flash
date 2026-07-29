# CUTEDSL_COOKBOOK

> Source study of FA4 CuTeDSL (installed flash-attn-4 4.0.0b23) against
> nvidia-cutlass-dsl 4.6.0, 2026-07-29. File:line refs are into the installed tree.

## Summary

CuTeDSL 4.6.0 API cookbook for writing an sm90 warp-specialized attention kernel, verified against the actual install at `/workspace/wan-attn/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/` (note: `cutlass` is NOT in site-packages directly — it is injected onto `sys.path` by `nvidia_cutlass_dsl.pth`). Every API below is quoted from installed source with file:line. Covers: host `@cute.jit` driver + `cute.compile` caching + torch→cute conversion (two distinct patterns FA4 uses: `from_dlpack(...).mark_layout_dynamic()` and `cute.runtime.make_fake_tensor` + `sym_int64`); `cute.struct`/`SmemAllocator` smem with swizzled WGMMA layouts; TMA atom creation + `tma_partition` copy-closures + `PipelineTmaAsync`; `make_trivial_tiled_mma` / `partition_fragment_ABC` / `cute.gemm` with the `Field.ACCUMULATE` idiom; warp reductions, `cute.math.exp2(fastmath=True)`, shuffles, predication; `cute.arch` named barriers and `setmaxregister_*`; compile-time-vs-runtime pitfalls; and an ~110-line annotated skeleton built only from verified APIs.

## Details

# CuTeDSL 4.6.0 Cookbook — sm90 Attention Kernel

**Install roots** (all paths absolute):
- CuTeDSL: `/workspace/wan-attn/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl/python_packages/cutlass/`
  (reached via `/workspace/wan-attn/.venv/lib/python3.12/site-packages/nvidia_cutlass_dsl.pth`, which does
  `sys.path.insert(0, os.path.join(nvidia_cutlass_dsl.__path__[0], 'python_packages'))`. There is **no**
  `site-packages/cutlass/` directory — don't go looking for it.)
- FA4: `/workspace/wan-attn/.venv/lib/python3.12/site-packages/flash_attn/cute/`
- quack: `/workspace/wan-attn/.venv/lib/python3.12/site-packages/quack/`

Below I abbreviate `$C = .../nvidia_cutlass_dsl/python_packages/cutlass`, `$F = .../flash_attn/cute`, `$Q = .../quack`.

---

## 0. Import map — what FA4 actually imports

`$F/flash_fwd_sm90.py:8-49` (verbatim head of the sm90 forward):

```python
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from quack import copy_utils
from quack import layout_utils
from quack import sm90_utils
from cutlass.cute import FastDivmodDivisor
```

Subpackage inventory:
- `$C/cute/`: `core.py` (6454 lines), `atom.py`, `tensor.py`, `algorithm.py`, `math.py`, `typing.py`, `runtime.py`, `arch/`, `nvgpu/{cpasync,warpgroup,warp,tcgen05}`
- `$C/pipeline/`: `helpers.py`, `sm90.py`, `sm100.py`, `profiling.py`; exports listed at `$C/pipeline/__init__.py:12-50`
- `$C/utils/`: `smem_allocator.py`, `hopper_helpers.py`, `layout.py`, `hardware_info.py`, tile schedulers; exports at `$C/utils/__init__.py:12-130`
- `$C/cutlass_dsl/cutlass.py` (3261 lines) — `KernelLauncher`, `min`/`max`/`if_generate`

---

## 1. Kernel definition, launch, host driver, compilation & caching

### 1.1 The two decorators

`$C/cute/__init__.py:221-225`:
```python
jit: Callable[..., Any] = _dsl.CuTeDSL.jit
kernel: Callable[..., Any] = _dsl.CuTeDSL.kernel
compile = _dsl.CompileCallable()
compile_to = compile.compile_to
```
`$C/base_dsl/dsl.py:863-881`:
```python
@classmethod
def jit(cls, *dargs, **dkwargs):     # "Decorator to mark a function for JIT compilation for Host code."
@classmethod
def kernel(cls, *dargs, **dkwargs):  # "Decorator to mark a function for JIT compilation for GPU."
```

`@cute.jit` = host code **and** device helper functions (FA4 uses it for both: `__call__` at
`$F/flash_fwd_sm90.py:157` is host; `load`/`mma`/`mma_one_n_block` at `:638,:936,:1348` are device-side
`@cute.jit` inlined into the kernel). `@cute.kernel` marks exactly one function — the GPU entry point
(`$F/flash_fwd_sm90.py:401`).

### 1.2 Launch

`@cute.kernel`-decorated calls return a `KernelLauncher` (`$C/cutlass_dsl/cutlass.py:1957-2050`). Docstring at
`:1963-1971`:
```python
kernel(arg1, arg2, ...).launch(grid=[1, 1, 1], block=[1, 1, 1], ...)
# or
kernel(arg1, arg2, ...)(grid=[1, 1, 1], block=[1, 1, 1], ...)
```
FA4's actual launch, `$F/flash_fwd_sm90.py:394-399`:
```python
).launch(
    grid=grid_dim,
    block=[self.num_threads, 1, 1],
    stream=stream,
    min_blocks_per_mp=1,
)
```
Full `LaunchConfig` field list — `$C/base_dsl/dsl.py:1325-1346`:
```python
@dataclass
class LaunchConfig:
    cluster: list | None = None
    fallback_cluster: list | None = None
    grid: list = [1,1,1]
    block: list = [1,1,1]
    max_number_threads: list = [0,0,0]
    smem: int | None = None
    async_deps: list = []
    has_cluster: bool = False
    has_fallback_cluster: bool = False
    min_blocks_per_mp: int = 0
    use_pdl: bool = False
    cooperative: bool = False
    launch_completion_event / ..._flags / programmatic_event / ... 
    smem_merge_branch_allocs: bool = False
    preferred_smem_carveout: int | None = None
```
**You do not pass `smem=`** — `SmemAllocator` computes it. `$C/utils/smem_allocator.py:60-63`:
> "SmemAllocator will automatically calculate the usage upon kernel launch. There is no need to explicitly specify shared memory size in kernel launch."

`stream` is a plain kwarg forwarded; FA4 declares it as the *last* host param with a comment
(`$F/flash_fwd_sm90.py:176-177`):
```python
# Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
stream: cuda.CUstream = None,
```

Grid shape is computed host-side by the scheduler. `$F/tile_scheduler.py:228-246`:
```python
@staticmethod
def get_grid_shape(params: Params, *, loc=None, ip=None) -> Tuple[Int32, Int32, Int32]:
    ...
    return (grid_x, params.num_head * params.num_splits, params.num_batch)
```
and consumed inside the kernel via `cute.arch.block_idx()` (`$F/tile_scheduler.py:220`:
`blk_coord = cute.arch.block_idx()`), which returns `Tuple[Int32,Int32,Int32]`
(`$C/cute/arch/nvvm_wrappers.py:378-389`).

For your fixed-shape Wan case you can skip the scheduler entirely: `grid=[ceil_div(S_q,tile_m), nheads, 1]`
and read `m_block, head_idx, _ = cute.arch.block_idx()`.

### 1.3 torch → cute: **two** patterns, both live in this install

**Pattern A — real DLPack tensor as compile "prototype"** (`$F/cute_dsl_utils.py:62-84`):
```python
from cutlass.cute.runtime import from_dlpack           # $F/cute_dsl_utils.py:16

def to_cute_tensor(t, assumed_align=16, leading_dim=-1, fully_dynamic=False, enable_tvm_ffi=True):
    """Convert torch tensor to cute tensor for TVM FFI. leading_dim=-1 defaults to t.ndim-1."""
    if t is None: return None
    ...
    tensor = from_dlpack(t.detach(), assumed_align=assumed_align, enable_tvm_ffi=enable_tvm_ffi)
    if fully_dynamic:
        return tensor.mark_layout_dynamic()
    if leading_dim == -1:
        leading_dim = t.ndim - 1
    return tensor.mark_layout_dynamic(leading_dim=leading_dim)
```
`from_dlpack` signature (`$C/cute/runtime.py:804-811`):
```python
def from_dlpack(
    tensor_dlpack: object,
    assumed_align: Optional[int] = None,
    use_32bit_stride: bool = False,
    *,
    enable_tvm_ffi: bool = False,
    force_tf32: bool = False,
) -> Tensor:
```
Variants with per-mode dynamic shape (`$F/utils.py:216-224`):
```python
def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )
```
and `convert_from_dlpack_leading_static` (`$F/utils.py:254-263`) which marks *every* mode except the leading
one dynamic.

**Pattern B — fake tensors, no GPU allocation needed** (this is the cleaner one for a greenfield kernel;
FA4 uses it for bwd preprocess/postprocess). `$F/interface.py:1189-1214`:
```python
mCuSeqlensQ = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cuseqlens_q else None
...
return cute.compile(
    fa_bwd_pre, mO, mdO, mPdPsum, mLSE, mLSElog2, mdQaccum, mCuSeqlensQ, mSequsedQ, mdLSE,
    mRowMax, mScaleP, softmax_scale,
    cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
    options="--enable-tvm-ffi",
)
```
where `fake_tensor` is `quack.compile_utils.make_fake_tensor` (`$Q/compile_utils.py:8-19`):
```python
def make_fake_tensor(dtype, shape, divisibility=1, leading_dim=-1) -> Optional[cute.Tensor]:
    if leading_dim < 0: leading_dim = len(shape) + leading_dim
    if dtype is None: return None
    stride = tuple(
        cute.sym_int64(divisibility=divisibility) if i != leading_dim else 1
        for i in range(len(shape))
    )
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=stride, assumed_align=divisibility * dtype.width // 8
    )
```
Underlying builders (`$C/cute/runtime.py:688-700`, `:587-600`):
```python
def make_fake_tensor(dtype, shape, stride, *, memspace=AddressSpace.gmem,
                     assumed_align: int | None = None) -> _FakeTensor: ...
def make_fake_compact_tensor(dtype, shape, *, stride_order=None, memspace=AddressSpace.gmem,
                             assumed_align=None, use_32bit_stride=False) -> _FakeTensor: ...
def make_fake_stream(*, use_tvm_ffi_env_stream: bool = False) -> _FakeStream:   # :789
```
Symbolic ints: `cute.sym_int(...)`, `cute.sym_int32(divisibility=1, symbol=None)`,
`cute.sym_int64(divisibility=1, symbol=None)` — `$C/cute/typing.py:264,282,297`.
Doc note at `$C/cute/runtime.py:733-736`: *"If the same runtime symbolic quantity appears in multiple
positions, reuse the same SymInt object at every occurrence. Different SymInt objects are treated as distinct
runtime parameters even if they share the same symbol string."*

**Since Wan2.1 shapes are fixed** (S_q ∈ {32760, 75600}, d=128, h ∈ {12,40}, b=1) you should pass *static*
Python ints in `make_fake_compact_tensor` shape tuples and get a fully-static kernel — no dynamic strides, no
`FastDivmodDivisor`, no `mark_layout_dynamic` at all. That is strictly better codegen than what FA4's generic
path produces.

### 1.4 `cute.compile` + host-side cache

`CompileCallable.__call__` (`$C/base_dsl/compiler.py:958-1023`):
```python
def __call__(self, *args, **kwargs) -> Any:
    """Compile ``func`` for the signature described by ``args``.
    :param args: ``func`` followed by representative compile-time arguments...
    :param kwargs: Optional compile controls such as ``options=...``"""
    return self._compile(*args, **kwargs)
```
FA4's host driver, `$F/interface.py:1007-1032`:
```python
compile_args = [
    fa_fwd, q_tensor, k_tensor, v_tensor, o_tensor, lse_tensor, softmax_scale,
    cu_seqlens_q_tensor, ..., learnable_sink_tensor,
]
compile_args.extend([sparse_tensors, AuxData(cute_aux_tensors, aux_scalars)])
compile_args.append(current_stream)
_flash_attn_fwd.compile_cache[compile_key] = cute.compile(
    *compile_args, options="--enable-tvm-ffi"
)
```
Note `fa_fwd` is a **callable instance** whose `__call__` is `@cute.jit` — `cute.compile` accepts
"a regular function, bound method, or callable instance" (`$C/base_dsl/compiler.py:1044-1046`). This is
exactly how FA4 keeps compile-time config (tile sizes, causal flags) as plain Python attributes on `self`.

The cache is a plain dict keyed by a big tuple of static config (`$F/interface.py:782-1118`); dispatch to
`get_jit_cache("fwd")` (`$F/interface.py:1118`, factory at `$F/cache_utils.py:264-281`) which returns either
an in-memory `JITCache` or a disk-backed `JITPersistentCache` when
`FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1`.

Invocation with **raw torch tensors** once `--enable-tvm-ffi` is on (`$F/interface.py:1043-1060`):
```python
_flash_attn_fwd.compile_cache[compile_key](
    q_call, k_call, v_call, out.detach(), lse, softmax_scale, ...
)
```

Minimal greenfield host driver:
```python
import cutlass, cutlass.cute as cute
from cutlass import Float32, Int32, BFloat16

_cache = {}
def wan_attn_fwd(q, k, v, out, lse, softmax_scale):
    key = (q.shape, k.shape, q.dtype, bool(lse is not None))
    if key not in _cache:
        b, s_q, h, d = q.shape
        _, s_kv, _, _ = k.shape
        mk = lambda shp: cute.runtime.make_fake_compact_tensor(BFloat16, shp, stride_order=(3,1,2,0))
        _cache[key] = cute.compile(
            WanAttnFwdSm90(head_dim=d, tile_m=128, tile_n=128, num_stages=2),
            mk((b, s_q, h, d)), mk((b, s_kv, h, d)), mk((b, s_kv, h, d)), mk((b, s_q, h, d)),
            cute.runtime.make_fake_compact_tensor(Float32, (b, h, s_q)),
            Float32(softmax_scale),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
    _cache[key](q, k, v, out, lse, softmax_scale)
```

---

## 2. SMEM allocation

### 2.1 Swizzled layout atoms for WGMMA operands

Pick the atom by majorness + contiguous-mode size — `$C/utils/hopper_helpers.py:174-215`:
```python
@dsl_user_op
def get_smem_layout_atom(layout: LayoutEnum, element_type: Type[Numeric],
                         major_mode_size: int, *, loc=None, ip=None) -> Any:
    assert major_mode_size % 8 == 0
    sw128_num_contiguous_bits = 1024; sw64 = 512; sw32 = 256
    major_mode_size_bits = major_mode_size * element_type.width
    if layout.sm90_mma_major_mode() == OperandMajorMode.MN:
        if major_mode_size_bits % 1024 == 0: return SmemLayoutAtomKind.MN_SW128
        ...
    if major_mode_size_bits % 1024 == 0: return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW128
    if major_mode_size_bits % 512 == 0:  return ...K_SW64
    if major_mode_size_bits % 256 == 0:  return ...K_SW32
    return cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER
```
Enum `SmemLayoutAtomKind` = `{MN_INTER, MN_SW32, MN_SW64, MN_SW128, K_INTER, K_SW32, K_SW64, K_SW128}`
(`$C/cute/nvgpu/warpgroup/mma.py:702-721`).
For **bf16, head_dim=128**: `128*16 = 2048 bits`, `2048 % 1024 == 0` → **`K_SW128`**.

Materialize the atom — `$C/cute/nvgpu/warpgroup/helpers.py:25-91`:
```python
def make_smem_layout_atom(kind: SmemLayoutAtomKind, element_type: Type[Numeric],
                          *, loc=None, ip=None) -> ComposedLayout:
    ...
    elif kind in (SmemLayoutAtomKind.MN_SW128, SmemLayoutAtomKind.K_SW128):
        num_contiguous_bits = 1024
        sw = core.make_swizzle(3, 4, 3)
    ...
    else:  # K-major layout
        return core.make_composed_layout(
            sw, 0, core.make_layout((8, num_contiguous_elems), stride=(num_contiguous_elems, 1)))
```
Returns a `ComposedLayout` with `.outer` (the plain layout) and `.inner` (the `Swizzle`) — you use both when
making the tensor (§2.3).

### 2.2 One-call staged layout builder (quack)

`$Q/sm90_utils.py:14-38`:
```python
@dsl_user_op
def make_smem_layout(dtype: Type[Numeric], layout: LayoutEnum, tile: cute.Tile,
                     stage: Optional[int] = None, major_mode_size: Optional[int] = None,
                     *, loc=None, ip=None) -> Union[cute.Layout, cute.ComposedLayout]:
    shape = cute.product_each(cute.shape(tile, loc=loc, ip=ip), loc=loc, ip=ip)
    if const_expr(major_mode_size is None):
        major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(layout, dtype, major_mode_size), dtype)
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2])
    return smem_layout_staged
```
FA4's call site — `$F/flash_fwd_sm90.py:235-243`:
```python
self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
    sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
    for mX, shape, stage in [
        (mQ, (self.tile_m, self.tile_hdim), None),
        (mK, (self.tile_n, self.tile_hdim), self.num_stages),
        (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
        (mO, (self.tile_m, self.tile_hdimv), None),
    ]
]
```
`LayoutEnum` = `{ROW_MAJOR, COL_MAJOR}` with helpers `is_m_major_c/is_n_major_c/sm90_mma_major_mode`
(`$C/utils/layout.py:18-68`).

**Backward-specific gotcha** worth stealing: when the same buffer feeds both `X` and `Xᵀ` operands, you must
pin `major_mode_size` down to the gcd so the swizzle works for both views — `$F/flash_bwd_sm90.py:205-241`:
```python
# Need to set major_mode_size (mms) to accommodate Q and Q.T
((self.tile_m, self.tile_hdim), self.Q_stage, self.tile_hdim // wg_d_dKV),
...
major_mode_size=math.gcd(self.tile_n // wg_n_SdP, self.tile_n // wg_n_dKV),   # for sPdS
```

### 2.3 `cute.struct` storage + `SmemAllocator`

Struct declaration — `$F/flash_fwd_sm90.py:120-155`:
```python
def _get_shared_storage_cls(self):
    sQ_struct, sK_struct, sV_struct = [
        cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(layout)], self.buffer_align_bytes
        ]
        for layout in (self.sQ_layout, self.sK_layout, self.sV_layout)
    ]
    mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
    mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
    mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

    @cute.struct
    class SharedStorageQKV:
        mbar_ptr_Q: mbar_ptr_Q_struct
        mbar_ptr_K: mbar_ptr_K_struct
        mbar_ptr_V: mbar_ptr_V_struct
        sV: sV_struct
        sQ: sQ_struct
        sK: sK_struct
        sP: sP_struct

    return SharedStorageQKV
```
`self.buffer_align_bytes = 1024` (`$F/flash_fwd_sm90.py:64`). `2*num_stages` Int64 slots per pipeline =
full + empty mbarrier arrays.

`cute.struct` API (`$C/cute/core.py:5440-5490` docstring): supports scalars, `MemRange[T, n]`, nested structs,
`Align[member, bytes]`; `Storage.__sizeof__()` / `__alignof__()` are static.

Allocation & tensor materialization — `$F/flash_fwd_sm90.py:448-449, 452, 521-535`:
```python
smem = cutlass.utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
...
mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()
...
sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
# reuse sQ's data iterator for the epilogue O buffer
sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
```
Signatures:
- `SmemAllocator.__init__(self, *, loc=None, ip=None)` — `$C/utils/smem_allocator.py:122-135` (takes nothing)
- `allocate(self, size_or_type: Any, byte_alignment: int = 1, *, loc=None, ip=None) -> cute.Pointer`
  — `$C/utils/smem_allocator.py:215-222`; for a struct type it returns the initialized struct instance
- `MemRange.data_ptr(self, *, loc=None, ip=None) -> Pointer` — `$C/cute/core.py:5582-5596`
- `MemRange.get_tensor(self, layout, swizzle=None, dtype=None, *, loc=None, ip=None) -> Tensor`
  — `$C/cute/core.py:5598-5628`. Note: *"raises TypeError: If the layout is incompatible with the swizzle"* —
  you must pass `.outer` when you pass `swizzle=.inner`, never the `ComposedLayout` itself with a swizzle.
- `allocate_tensor(element_type, layout, byte_alignment=1, swizzle=None, ...)` — `$C/utils/smem_allocator.py:331-340`
- `SmemAllocator.capacity_in_bytes(compute_capability=None)` — `:102-120`; also
  `cutlass.utils.get_smem_capacity_in_bytes("sm_90")` (used in `$F/flash_fwd.py:169`)

**Transpose view** of V for the PV WGMMA (no data movement, just a `composition`) — `$Q/layout_utils.py:10-15`:
```python
def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))
```
used at `$F/flash_fwd_sm90.py:530`: `sVt = layout_utils.transpose_view(sV)`.

---

## 3. TMA: atoms, mbarriers, pipelines

### 3.1 Copy ops and atom creation (host side)

`$F/flash_fwd_sm90.py:261-263`:
```python
gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()   # Might multicast
gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
```
(Class defs: `$C/cute/nvgpu/cpasync/copy.py:291` G2S tile, `:461` G2S multicast, `:815` S2G tile,
`:1146` `CopyBulkG2SOp`, `:71` `CopyG2SOp` for cp.async.)

`$C/cute/nvgpu/cpasync/helpers.py:419-430`:
```python
def make_tiled_tma_atom(
    op: TMAOp,
    gmem_tensor: Tensor,
    smem_layout_: Union[Layout, ComposedLayout],
    cta_tiler: Tiler,
    num_multicast: int = 1,
    *,
    internal_type: Optional[Type[Numeric]] = None,
    loc=None, ip=None,
) -> TmaInfo:
```
Returns `(copy_atom, tma_tensor)` (it's a `TmaInfo` namedtuple-like, unpacked as a 2-tuple by callers).
Docstring at `:470-473`: *"smem_layout must be non-staged (rank == rank(cta_tiler)) or staged
(rank == rank(cta_tiler)+1)"*.

FA4's calls, `$F/flash_fwd_sm90.py:278-314`:
```python
tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_Q, mQ, self.sQ_layout, (self.tile_m, self.tile_hdim),   # No mcast
)
tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_KV, mK,
    cute.select(self.sK_layout, mode=[0, 1]),      # strip the stage mode
    (self.tile_n, self.tile_hdim),
    1,                                             # No mcast for now
)
tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_KV, mV, cute.select(self.sV_layout, mode=[0, 1]),
    (self.tile_n, self.tile_hdimv), 1,
)
tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
    gmem_tiled_copy_O, mO_tma, self.sO_layout, (self.tile_m, self.tile_hdimv),
)
```
Transaction-byte counts (needed by the pipeline) — `$F/flash_fwd_sm90.py:264-271`:
```python
self.tma_copy_bytes = {
    name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
    for name, mX, layout in [("Q", mQ, self.sQ_layout), ("K", mK, self.sK_layout),
                             ("V", mV, self.sV_layout)]
}
```

**Descriptor prefetch** (first thing in the kernel) — `$F/flash_fwd_sm90.py:441-446`:
```python
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
if warp_idx == 0:
    for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
        if const_expr(tma_atom is not None):
            cpasync.prefetch_descriptor(tma_atom)
```
(`$C/cute/nvgpu/cpasync/helpers.py:729-739`: `def prefetch_descriptor(tma_atom: atom.CopyAtom, *, loc=None, ip=None) -> None`.)

### 3.2 `tma_partition` and the copy-closure idiom

`$C/cute/nvgpu/cpasync/helpers.py:596-605`:
```python
def tma_partition(
    atom: atom.CopyAtom,
    cta_coord: Coord,
    cta_layout: Layout,
    smem_tensor: Tensor,
    gmem_tensor: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    *, loc=None, ip=None,
) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, Tensor]]:
```
quack wraps it into a closure — `$Q/copy_utils.py:868-914`:
```python
@dsl_user_op
def tma_get_copy_fn(atom, cta_coord, cta_layout, src_tensor, dst_tensor,
                    filter_zeros=False, single_stage=False, *, loc=None, ip=None, **kwargs) -> Callable:
    src_is_smem = const_expr(isinstance(src_tensor.iterator, cute.Pointer)
                             and src_tensor.memspace == cute.AddressSpace.smem)
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    group_rank_smem = const_expr(cute.rank(smem_tensor) - (1 if not single_stage else 0))
    group_rank_gmem = const_expr(cute.rank(gmem_tensor) - (1 if not single_stage else 0))
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom, cta_coord, cta_layout,
        cute.group_modes(smem_tensor, 0, group_rank_smem),
        cute.group_modes(gmem_tensor, 0, group_rank_gmem), loc=loc, ip=ip)
    ...
    @dsl_user_op
    def copy_tma(src_idx, dst_idx, *, loc=None, ip=None, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs, loc=loc, ip=ip)

    @dsl_user_op
    def copy_tma_single_stage(*, loc=None, ip=None, **new_kwargs):
        cute.copy(atom, src, dst, **new_kwargs, **kwargs, loc=loc, ip=ip)

    return (copy_tma if const_expr(not single_stage) else copy_tma_single_stage), s, g
```
and binds it to a pipeline — `$Q/copy_utils.py:1071-1081`:
```python
def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(src_idx=src_idx, dst_idx=producer_state.index,
             tma_bar_ptr=pipeline.producer_get_barrier(producer_state), **new_kwargs)
    return copy_fn
```
FA4 producer side — `$F/flash_fwd_sm90.py:686-721`:
```python
gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
load_Q, _, _ = copy_utils.tma_get_copy_fn(
    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True)
...
gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))   # None => keep the block mode
tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(tma_atom_K, 0, cute.make_layout(1), gK, sK)
tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
```
`cta_coord=0`, `cta_layout=cute.make_layout(1)` is the no-cluster / no-multicast spelling (what you want:
`cluster_shape_mn = (1, 1)` at `$F/flash_fwd_sm90.py:69`).

`cute.copy` signature — `$C/cute/algorithm.py:486-496`:
```python
def copy(atom: CopyAtom, src, dst, *, pred: Optional[Tensor] = None,
         unroll_factor: Optional[int] = None, loc=None, ip=None, **kwargs) -> None:
```
`tma_bar_ptr=` flows through `**kwargs` into the copy op.

**Critical warning** (`$C/cute/arch/elect.py:117-120`): *"Do NOT use `elect_one()` for … TMA copy operations
(`cute.copy` with TMA atoms) — TMA partitioning ensures only one thread within a warp issues the operation
automatically. Wrapping in `elect_one()` can cause GPU deadlock."*

### 3.3 mbarrier primitives (raw)

`$C/cute/arch/mbar.py`:
```python
def mbarrier_init(mbar_ptr: Pointer, cnt: Int, *, loc=None, ip=None) -> None          # :35
def mbarrier_init_fence(*, loc=None, ip=None) -> None                                 # :78
def mbarrier_arrive_and_expect_tx(mbar_ptr, bytes, peer_cta_rank_in_cluster=None,
                                  relaxed=False, scope=MemScopeKind.CTA, ...)          # :90
def mbarrier_expect_tx(...)                                                            # :160
def mbarrier_wait(...)      def mbarrier_try_wait(...)                                 # :223 / :254
def mbarrier_test_wait(...) def mbarrier_conditional_try_wait(...)                     # :285 / :316
def mbarrier_arrive(...)                                                               # :348
```
`mbarrier_init` and `mbarrier_expect_tx` **must** be inside `with cute.arch.elect_one():`
(`$C/cute/arch/mbar.py:44-53`). You almost never call these directly — the pipeline classes do.

### 3.4 Pipeline classes (what FA4 actually uses)

Exports — `$C/pipeline/__init__.py:12-50`: `Agent, CooperativeGroup, PipelineOp, MbarrierLayout, SyncObject,
MbarrierArray, NamedBarrier, TmaStoreFence, PipelineUserType, PipelineState, make_pipeline_state,
pipeline_init_arrive, pipeline_init_wait, agent_sync, arrive, wait, arrive_and_wait, sync,
PipelineAsync, PipelineCpAsync, PipelineTmaAsync, PipelineTmaStore, PipelineOrder, PipelineProducer,
PipelineConsumer, PipelineTmaUmma, ...`

`PipelineTmaAsync.create` — `$C/pipeline/sm90.py:529-546` (**keyword-only**):
```python
@staticmethod
def create(  # type: ignore[override]
    *,
    num_stages: int,
    producer_group: CooperativeGroup,
    consumer_group: CooperativeGroup,
    tx_count: int,
    barrier_storage: Optional[cute.Pointer] = None,
    cta_layout_vmnk: Optional[cute.Layout] = None,
    tidx: Optional[Int32] = None,
    mcast_mode_mn: tuple[int, int] = (1, 1),
    enable_multicast_signaling: bool = False,
    defer_sync: bool = False,
    name: str = "",
) -> "PipelineTmaAsync":
```
Internally it splits `barrier_storage` into full/empty arrays (`$C/pipeline/sm90.py:634-648`):
```python
sync_object_full  = PipelineAsync._make_sync_object(barrier_storage.align(min_align=8), num_stages, producer, tx_count, ..., phase="full")
sync_object_empty = PipelineAsync._make_sync_object(barrier_storage.align(min_align=8) + num_stages, num_stages, consumer, ..., phase="empty")
```
→ that's why the storage struct reserves `num_stages * 2` Int64 (§2.3).

`defer_sync=True` skips the built-in `mbarrier_init_fence` + block sync (`:571-573`,
*"Bool specifying whether or not to skip the built-in mbarrier fence and sync for performance"*) — you then do
one combined fence for all pipelines via `pipeline_init_arrive`/`pipeline_init_wait`.

Instance methods (`PipelineAsync` base, `$C/pipeline/sm90.py`):
```python
producer_acquire(state, try_acquire_token=None)   # :238   (TmaAsync override :684)
producer_try_acquire(state, ...)                  # :256
producer_commit(state)                            # :266   (TmaAsync: no-op, :706-712 — "TMA instruction itself updates the transaction count")
consumer_wait(state, try_wait_token=None)         # :276
consumer_try_wait(state)                          # :294
consumer_release(state)                           # :304   (TmaAsync override :719)
producer_tail(state)                              # :334
producer_get_barrier(state)                       # used by $Q/copy_utils.py:1077
```

Groups — `$C/pipeline/helpers.py:31-43, 51-77`:
```python
class Agent(enum.Enum): Thread; Warp; ThreadBlock; ThreadBlockCluster
class CooperativeGroup:
    def __init__(self, agent: Agent, size: Union[int, Int32] = 1, alignment: Optional[int] = None)
```
FA4's construction — `$F/flash_fwd_sm90.py:454-494`:
```python
ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
tma_warp     = ThreadCooperativeGroup(1)
load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
mma_warps    = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)

pipeline_k = pipeline_custom.PipelineTmaAsync.create(
    barrier_storage=storage.mbar_ptr_K.data_ptr(),
    num_stages=self.num_stages,
    producer_group=tma_warp,
    consumer_group=mma_warps,
    tx_count=self.tma_copy_bytes["K"],
    defer_sync=True,
)
```
Note the consumer group counts **warps** (`num_mma_threads // 32`), because `consumer_release` is issued once
per warp — see `$C/pipeline/sm90.py:719-736` gating on `is_signaling_thread`.

Init handshake — `$F/flash_fwd_sm90.py:516` and `:578`:
```python
pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
...
pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
```
signatures `$C/pipeline/helpers.py:933-939, 958-964`:
```python
def pipeline_init_arrive(cluster_shape_mn: Optional[cute.Layout] = None, is_relaxed: bool = False, *, loc=None, ip=None) -> None
def pipeline_init_wait(cluster_shape_mn: Optional[cute.Layout] = None, *, loc=None, ip=None) -> None
```
Bodies: `pipeline_init_arrive` does `cute.arch.mbarrier_init_fence()` then (cluster>1 only) `cluster_arrive*`;
`pipeline_init_wait` does `agent_sync(Agent.ThreadBlock)` when cluster size is 1 (`:947-974`). With
`cluster_shape_mn=(1,1)` these are exactly "fence, then `__syncthreads()`".

### 3.5 Pipeline state

Standard: `$C/pipeline/helpers.py` `make_pipeline_state(type, stages)` + `PipelineState` with
`.index`, `.phase`, `.advance()`, `.clone()`. FA4 usage `$F/flash_fwd_sm90.py:671-673, 997-999`:
```python
kv_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_stages)
kv_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
```
FA4 also ships a cheaper single-register state, `$F/pipeline.py:38-96`:
```python
class PipelineStateSimple:
    """Use a single Int32 to store both the index and phase bit, then we use divmod...
       If stages is a power of 2, divmod turns into bit twiddling."""
    @property
    def index(self):  return Int32(0) if self._stages == 1 else self._phase_index % self._stages
    @property
    def phase(self):  return self._phase_index if self._stages == 1 else self._phase_index // self._stages
    def advance(self): self._phase_index ^= 1 if self._stages == 1 else ...  # += 1
    def __extract_mlir_values__(self) / __new_from_mlir_values__(self, values)   # :78-83
```
The `__extract_mlir_values__`/`__new_from_mlir_values__` pair is the protocol that lets a Python object be
carried across `cutlass.range` loop boundaries as loop-carried IR values — copy this if you write your own
state objects.

FA4's index/phase-explicit mixin for single-stage Q (avoids a whole `PipelineState`) — `$F/pipeline.py:118-157`:
```python
class _PipelineIndexPhaseMixin:
    @dsl_user_op
    def producer_acquire_w_index_phase(self, index: Int32, phase: Int32, try_acquire_token=None, *, loc=None, ip=None)
    @dsl_user_op
    def producer_commit_w_index(self, index: Int32, *, loc=None, ip=None)
    @dsl_user_op
    def consumer_wait_w_index_phase(self, index: Int32, phase: Int32, try_wait_token=None, *, loc=None, ip=None)
    @dsl_user_op
    def consumer_release_w_index(self, index: Int32, *, loc=None, ip=None)
```
used at `$F/flash_fwd_sm90.py:792-794, 1099, 1182`:
```python
pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))
q_producer_phase ^= 1
...
pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
...
pipeline_q.consumer_release_w_index(0)
```

### 3.6 Producer / consumer call sequence (verbatim)

Producer, `$F/flash_fwd_sm90.py:786-789, 820-832, 913-914`:
```python
pipeline_k.producer_acquire(kv_producer_state)
load_K(block=n_block, producer_state=kv_producer_state, page_idx=page_idx)   # -> pipeline_k.producer_commit(state) inside load_KV
...
pipeline_v.producer_acquire(kv_producer_state)
load_V(...)
kv_producer_state.advance()
...
pipeline_v.producer_tail(kv_producer_state)   # only needed on the last-loaded buffer
```
`load_KV` body (`$F/flash_fwd_sm90.py:928-934`):
```python
if const_expr(self.use_tma_KV):
    tma_load_fn(src_idx=src_idx, producer_state=producer_state)
else:
    paged_kv_manager.load_KV(...)
    cute.arch.cp_async_commit_group()
pipeline_kv.producer_commit(producer_state)
```
Consumer, `$F/flash_fwd_sm90.py:1368-1373, 1402-1407`:
```python
pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
self.warp_scheduler_barrier_arrive()
warpgroup.wait_group(0)
pipeline_k.consumer_release(smem_pipe_read)
...
pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
pipeline_v.consumer_release(smem_pipe_read)
smem_pipe_read.advance()
```
The `consumer_wait(state, consumer_try_wait(state))` two-call form issues the `try_wait` early so the blocking
wait can be predicated away (`$C/pipeline/sm90.py:276-303`).

---

## 4. WGMMA

### 4.1 Tiled MMA construction

`$C/utils/hopper_helpers.py:92-170`:
```python
def make_trivial_tiled_mma(
    a_dtype: Type[Numeric],
    b_dtype: Type[Numeric],
    a_leading_mode: OperandMajorMode,
    b_leading_mode: OperandMajorMode,
    acc_dtype: Type[Numeric],
    atom_layout_mnk: Tuple[int, int, int],
    tiler_mn: Tuple[int, int],
    a_source: OperandSource = OperandSource.SMEM,
    *, loc=None, ip=None,
) -> cute.TiledMma:
    ...
    if a_dtype in {Float16, BFloat16}:
        mma_op = MmaF16BF16Op(a_dtype, acc_dtype, (*tiler_mn, 16), a_source, a_leading_mode, b_leading_mode)
    ...
    return cute.make_tiled_mma(cute.make_mma_atom(mma_op), atom_layout_mnk)
```
So for bf16 the K-tile is hard-wired to **16** and `tiler_mn[0]` must be 64 (one warpgroup tile).

FA4's two MMAs — `$F/flash_fwd_sm90.py:96-118`:
```python
tiled_mma_qk = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype, self.dtype,
    warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K, Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_n),
)
tiled_mma_pv = sm90_utils_basic.make_trivial_tiled_mma(
    self.dtype, self.dtype,
    warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN, Float32,
    atom_layout_mnk=(self.tile_m // 64, 1, 1),
    tiler_mn=(64, self.tile_hdimv),
    a_source=warpgroup.OperandSource.RMEM if self.mma_pv_is_rs else warpgroup.OperandSource.SMEM,
)
```
For Wan (`tile_m=128`) → `atom_layout_mnk=(2,1,1)` → `tiled_mma_qk.size == 256` → 2 MMA warpgroups.
Enums: `OperandMajorMode.{MN,K}` (`$C/cute/nvgpu/warpgroup/mma.py:65-103`),
`OperandSource.{RMEM,SMEM}` (`:107-123`), `Field.ACCUMULATE` (`:125-141`).
`MmaF16BF16Op` at `:308`.

A thinner wrapper (string-keyed) is available at `$Q/sm90_utils.py:45-74`:
```python
def make_tiled_mma(a_dtype, a_major: Literal["K","MN"], b_major: Literal["K","MN"], tiler_n: int,
                   source: Literal["SS","RS"] = "SS", atom_layout_mnk=(1,1,1), swap_AB=False,
                   b_dtype=None, acc_dtype=Float32) -> cute.TiledMma
```

### 4.2 Operand fragments

`$Q/sm90_utils.py:165-193`:
```python
def partition_fragment_ABC(thr_mma: cute.ThrMma, shape_mnk: cute.Shape,
                           sA: Optional[cute.Tensor], sB: Optional[cute.Tensor], swap_AB: bool = False):
    is_rs = thr_mma.op.a_src == warpgroup.OperandSource.RMEM
    if const_expr(not swap_AB):
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C(shape_mnk[:2]), Float32)
        if const_expr(not is_rs):
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_A(sA))
        else:
            tCrA = thr_mma.make_fragment_A(thr_mma.partition_shape_A((shape_mnk[0], shape_mnk[2])))
        tCrB = thr_mma.make_fragment_B(thr_mma.partition_B(sB))
    ...
    return acc, tCrA, tCrB
```
FA4 call sites — `$F/flash_fwd_sm90.py:966-982`:
```python
warp_group_idx = cute.arch.make_warp_uniform(tidx // self.num_threads_per_warp_group)
warp_group_thread_layout = cute.make_layout(self.num_wg_mma, stride=self.num_threads_per_warp_group)
thr_mma_qk = tiled_mma_qk.get_slice(tidx)                                   # for C-partitioning / masks
wg_mma_qk  = tiled_mma_qk.get_slice(warp_group_thread_layout(warp_group_idx))  # for A/B fragments
wg_mma_pv  = tiled_mma_pv.get_slice(warp_group_thread_layout(warp_group_idx))

_, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
    wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK)
mma_qk_fn = partial(sm90_utils.gemm_zero_init, tiled_mma_qk, (self.tile_m, self.tile_n), tSrQ, tSrK)

acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
    wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt)
mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
```
**Two different slices of the same TiledMma** is the key trick: `get_slice(tidx)` for accumulator/coordinate
partitioning, `get_slice(warp_group_thread_layout(wg_idx))` for the smem-descriptor operands.

### 4.3 The gemm call + `Field.ACCUMULATE`

`$Q/sm90_utils.py:97-121` — read the comment, it's a real trap:
```python
@cute.jit
def gemm(tiled_mma, acc, tCrA, tCrB, zero_init=False, wg_wait=0, swap_AB=False) -> None:
    ...
    warpgroup.fence()
    # We make a new mma_atom since we'll be modifying its attribute (accumulate).
    # Otherwise the compiler complains "operand #0 does not dominate this use"
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if const_expr(wg_wait >= 0):
        warpgroup.wait_group(wg_wait)
```
Wrappers `$Q/sm90_utils.py:124-143` (`gemm_zero_init` — allocates the accumulator and returns it) and
`:146-162` (`gemm_w_idx` — accumulates into an existing `acc`, selects the stage with
`tCrB[None, None, None, B_idx]`).

`cute.gemm` — `$C/cute/algorithm.py:69-79`:
```python
def gemm(atom: MmaAtom, d: Tensor, a, b, c: Tensor, *, loc=None, ip=None, **kwargs) -> None:
    """Computes ``D <- A * B + C`` where ``C`` and ``D`` can alias. Note that some MMA Atoms
       (e.g. warpgroup-wide or tcgen05 MMAs) require manually setting an "accumulate" boolean field."""
```
Warpgroup sync ops — `$C/cute/nvgpu/warpgroup/helpers.py:94-125`:
```python
def fence(*, loc=None, ip=None) -> None          # wgmma.fence.sync.aligned
def commit_group(*, loc=None, ip=None) -> None   # wgmma.commit_group.sync.aligned
def wait_group(group: Any, *, loc=None, ip=None) -> None
```
`wg_wait=-1` means "don't wait here"; the caller then does `warpgroup.wait_group(1)` /
`wait_group(0)` manually to overlap QK and PV (`$F/flash_fwd_sm90.py:1442, 1453`).

### 4.4 Accumulator → next-GEMM-A relayout

`acc_S` (fp32, layout `((2,2,V), MMA_M, MMA_N)`) must become an RMEM A-operand for PV.
`$Q/layout_utils.py:208-252`:
```python
@cute.jit
def convert_layout_acc_frgA(acc_layout: cute.Layout) -> cute.Layout:
    # For Sm90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N)
    #   to ((2, 2, 2), MMA_M, (N / 16, MMA_N)).
    # If N / 8 is odd, we'll convert to ((2, 2, 1), MMA_M, N / 8, MMA_N).
def reshape_acc_to_frgA(acc): return cute.make_tensor(acc.iterator, convert_layout_acc_frgA(acc.layout))
```
and the MN view for softmax (`:168-206`):
```python
def convert_layout_acc_mn(acc_layout, transpose=False) -> cute.Layout:
    """For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...)."""
def reshape_acc_to_mn(acc, transpose=False): ...
```
Use in the mainloop — `$F/flash_fwd_sm90.py:1383-1396`:
```python
tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
tOrP_cur = tOrP if const_expr(self.mma_pv_is_rs) else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
# tOrP.store(tOrP_acc.load().to(self.dtype))
# the "to(self.dtype)" conversion fails to vectorize for block sizes other than 128 x 128, i.e. it calls
# convert on 1 fp32 element at a time instead of 2 elements. So we just call ptx directly.
utils.cvt_f16(tOrP_acc, tOrP_cur)
```
(`utils.cvt_f16` at `$F/utils.py:638-660`, built on `cvt_f16x2_f32` at `:620`.) **Note the comment: at
tile 128×128 the plain `.to(dtype)` does vectorize.** Wan is exactly 128×128, so you may use the simple form
first and only drop to PTX if the SASS shows scalar `cvt`.

If you use SMEM-source PV (`mma_pv_is_rs=False`) you must fence after storing P to smem —
`$F/flash_fwd_sm90.py:1394-1401`:
```python
tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
...
cute.arch.fence_view_async_shared()
cute.arch.sync_warp()  # Only need syncwarp since each warp is using its own P values for MmaPV
```
The smem store atom is `stmatrix` for 16-bit on sm90 — `$F/utils.py:302-316`:
```python
def get_smem_store_atom(arch, element_type, transpose=False) -> cute.CopyAtom:
    if const_expr(arch < 90 or element_type.width != 16):
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), element_type,
                                   num_bits_per_copy=2 * element_type.width)
    else:
        return cute.make_copy_atom(cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=transpose, num_matrices=4),
                                   element_type)
```
paired with `cute.make_tiled_copy_C(smem_copy_atom, tiled_mma).get_slice(tidx)`
(`$F/flash_fwd_sm90.py:990`, `$C/cute/atom.py:1265`).

---

## 5. Reductions, exp2, shuffles, predication

### 5.1 Warp / quad reductions

Built-in — `$C/cute/arch/nvvm_wrappers.py:640-677`:
```python
@dsl_user_op
def warp_reduction(val: Numeric, op: Callable, *, threads_in_group: int = WARP_SIZE, loc=None, ip=None) -> Numeric:
    """...The threads_in_group is the number of threads reduction group in a warp.
       E.g. WARP_SIZE (32) means the whole warp reduced in one group. 8 means the warp is divided into
       4 thread groups, each group has 8 threads in reduction."""
    offset = threads_in_group // 2
    while offset > 0:
        val = op(val, shuffle_sync_bfly(val, offset=offset, mask=-1, mask_and_clamp=31, loc=loc, ip=ip))
        offset = offset // 2
    return val

warp_reduction_max = partial(warp_reduction, op=lambda x, y: fmax(x, y) if isinstance(x, Float32) else cutlass_dsl.max(x, y))
warp_reduction_sum = partial(warp_reduction, op=lambda x, y: x + y)
```
Softmax uses the 4-lane (quad) form because a WGMMA accumulator row lives across 4 lanes —
`$F/softmax.py:159`:
```python
row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
```
FA4's own vectorized variant (works on `TensorSSA`) — `$F/utils.py:318-333`:
```python
@cute.jit
def warp_reduce(val: cute.TensorSSA | cute.Numeric, op: Callable,
                width: cutlass.Constexpr[int] = cute.arch.WARP_SIZE) -> cute.TensorSSA | cute.Numeric:
    if const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype); res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
    return val
```
used at `$F/softmax.py:204`: `row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))`.

Intra-thread tree reductions (`$F/utils.py:367-417` `fmax_reduce`, `:418-460` `fadd_reduce`) hand-unroll a
4-wide max tree on sm90 and fall back to `x.reduce(cute.ReductionOp.ADD, init_val, 0)` for the sum
(`$F/utils.py:421-423`). Three-input `fmax(a,b,c)` (`$F/utils.py:351-364`) and packed
`cute.arch.add_packed_f32x2` are sm100-only paths (`arch < 100` guard at `:369`, `:420`).

### 5.2 exp2

`cute.arch.exp2` is **deprecated** in 4.6.0 — `$C/cute/arch/nvvm_wrappers.py:1352-1358`:
```python
@dsl_user_op
@deprecated("cute.arch.exp2 is deprecated, use cute.math.exp2 with `fastmath=True` instead")
def exp2(a, *, loc=None, ip=None) -> Float32:
```
Use `cute.math.exp2` — `$C/_mlir_helpers/math.py:649-681` (re-exported by `$C/cute/math.py:44`):
```python
def exp2(x: MathOperand, fastmath: bool = False, approx: bool = False, ftz: bool = False,
         *, loc=None, ip=None) -> MathOperand:
    _validate_fastmath_exclusive("exp2", fastmath, approx=approx, ftz=ftz)
    _validate_ftz_requires_approx("exp2", approx, ftz)
    if approx: return _call_nvvm_unary(x, "ex2", approx=True, ftz=ftz, loc=loc, ip=ip)
    return _unary_math_op(x, math_dialect.exp2, None, fastmath, "exp2", loc=loc, ip=ip)
```
FA4 online-softmax kernel, `$F/softmax.py:167-188`:
```python
row_max_cur_scaled = row_max_cur * scale_log2
acc_S_row_exp = cute.math.exp2(acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True)
acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=None, arch=arch)
row_scale[r] = 1.0
# ... non-first block:
row_scale[r] = cute.math.exp2((row_max_prev - row_max_cur) * scale_log2, fastmath=True)
acc_S_row_sum = utils.fadd_reduce(acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch)
row_sum[r] = acc_S_row_sum
acc_S_mn[r, None].store(acc_S_row_exp)
```
Finalize (`$F/softmax.py:207-227`): guards `row_sum == 0 or NaN`, uses
`cute.arch.rcp_approx(row_sum[r])` and `cute.math.log2(row_sum_cur, fastmath=True)`, writes
`lse = (row_max*scale_log2 + log2(row_sum)) * LN2`.
`cute.math` exports include `exp, exp2, log, log2, rsqrt, sqrt, tanh, erf, ...`
(`$C/cute/math.py:44-66, 87-109`).

### 5.3 Shuffles

`$C/cute/arch/nvvm_wrappers.py:492-506` + `:299-306`:
```python
def shuffle_sync_op(value, offset: Int, mask: Int = FULL_MASK, mask_and_clamp: Int = WARP_SIZE - 1,
                    kind: nvvm.ShflKind = nvvm.ShflKind.idx, *, loc=None, ip=None)
shuffle_sync      = partial(shuffle_sync_op, kind=nvvm.ShflKind.idx)
shuffle_sync_up   = partial(shuffle_sync_op, kind=nvvm.ShflKind.up)
shuffle_sync_down = partial(shuffle_sync_op, kind=nvvm.ShflKind.down)
shuffle_sync_bfly = partial(shuffle_sync_op, kind=nvvm.ShflKind.bfly)
```
Supports `TensorSSA` when total bit width is exactly 32 (`:527-541`), scalars up to 64 bits.

### 5.4 Predication (this is what you need for `S_kv = 32760`)

32760 = 128·255 + 120 → the last KV tile is partial, so you *do* need column masking even non-causal.
The identity-tensor pattern — `$F/mask.py:193-222`:
```python
acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S, transpose=self.swap_AB)
cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
tScS_mn  = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cS), transpose=self.swap_AB)
# We use t0ScS as these indices are known at compile time. We then must subtract the
# column limit by the thread column offset.
t0ScS_mn = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cS), transpose=self.swap_AB)
thr_col_offset = tScS_mn[0][COL]
seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset
if const_expr(not mask_causal and not mask_local and mask_mod is None):
    if const_expr(mask_seqlen):
        for c in cutlass.range(cute.size(tScS_mn.shape[1]), unroll_full=True):
            oob = t0ScS_mn[0, c][COL] >= seqlenk_col_limit
            for r in cutlass.range(cute.size(tScS_mn.shape[0]), unroll_full=True):
                acc_S_mn[r, c] = -Float32.inf if oob else acc_S_mn[r, c]
```
A faster sm90 path uses R2P bitmasks (`$F/mask.py:220-222`, helpers `r2p_bitmask_below` at `:53-61`,
`mask_r2p_lambda` at `:74`) — worth trying but the loop above is the correct baseline.

For copy predication (head-dim OOB, not needed at d=128 but shown for completeness) — `$F/utils.py:484-497`:
```python
@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_rmem_tensor(cute.make_layout(
        (cute.size(tAcA, mode=[0,1]), cute.size(tAcA, mode=[1]), cute.size(tAcA, mode=[2])),
        stride=(cute.size(tAcA, mode=[2]), 0, 1)), cutlass.Boolean)
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA
```
then `cute.copy(..., pred=tOpO[None, rest_m, None])` (`$F/flash_fwd.py:441-448`).

---

## 6. `cute.arch` API surface (verified `__all__` at `$C/cute/arch/__init__.py:24-186`)

| Call | Signature / file:line |
|---|---|
| `cute.arch.thread_idx()` | `-> Tuple[Int32,Int32,Int32]`, `nvvm_wrappers.py:350` |
| `cute.arch.block_idx()` / `grid_dim()` / `block_dim()` | `-> Tuple[Int32×3]`, `:378`, `:392`, `:365` |
| `cute.arch.warp_idx()` / `lane_idx()` | `:322` / `:312` |
| `cute.arch.make_warp_uniform(value: Int) -> Int32` | `elect.py:24-45` — "compiler hint indicating that the specified value is invariant across all threads in the warp" |
| `with cute.arch.elect_one():` | `elect.py:79-149` — returns an `IfOpRegion` context manager |
| `cute.arch.barrier(*, barrier_id=None, number_of_threads=None, loc, ip)` | `nvvm_wrappers.py:683-707` — **keyword-only** |
| `cute.arch.barrier_arrive(*, barrier_id=None, number_of_threads, aligned=True, loc, ip)` | `nvvm_wrappers.py:710-763` — `number_of_threads` is **required**; `aligned=False` emits `barrier.cta.arrive` (use when only a subset of warps arrives) |
| `cute.arch.sync_threads()` | `:766-774` (`barrier 0`) |
| `cute.arch.sync_warp(mask=FULL_MASK)` | `:776-787` |
| `cute.arch.setmaxregister_increase(reg_count: int)` | `:1186-1196` |
| `cute.arch.setmaxregister_decrease(reg_count: int)` | `:1198-1209` |
| `cute.arch.warpgroup_reg_alloc/dealloc` | `:1211-1234` — **`@deprecated("use setmaxregister_increase instead")`**; FA4 uses the new names |
| `cute.arch.fence_view_async_shared()` | `:1166-1183` — "only available on sm_90 or higher … required to synchronize the shared memory load/store and let the pipeline release or commit the buffer" |
| `cute.arch.fence_proxy(kind=..., space=...)` | `:925` |
| `cute.arch.cp_async_commit_group()` / `cp_async_wait_group(n)` | `:837` / `:849` |
| `cute.arch.cp_async_bulk_commit_group()` / `cp_async_bulk_wait_group(n, read=True)` | in `__all__`; used at `$F/flash_fwd.py:417-418` |
| `cute.arch.fmax(a,b)` / `fmin` / `rcp_approx(a)` | `:1307` / `:1325` / `:1343` |
| `cute.arch.shuffle_sync{,_up,_down,_bfly}` | `:299-306` |
| `cute.arch.warp_reduction{,_max,_sum}` | `:640`, `:675`, `:679` |
| `cute.arch.mbarrier_*` | `mbar.py:35,78,90,160,223,254,285,316,348` |
| `cute.arch.WARP_SIZE=32`, `WARPS_PER_WARPGROUP=4`, `THREADS_PER_WARPGROUP=128` | `arch/constants.py:16-18` |
| `cute.arch.dynamic_smem_size()` | `smem.py`; replaces the deprecated `SmemAllocator._allocated_bytes` (`smem_allocator.py:141-144`) |

### Named barriers
Barrier **0 is reserved** for `sync_threads()` — `$F/named_barrier.py:6-12`:
```python
class NamedBarrierFwd(enum.IntEnum):
    Epilogue = enum.auto()  # starts from 1 as barrier 0 is reserved for sync_threads()
    WarpSchedulerWG1 = enum.auto()
    WarpSchedulerWG2 = enum.auto()
    WarpSchedulerWG3 = enum.auto()
    PFull = enum.auto()
    PEmpty = enum.auto()
```
The WG ping-pong scheduler (this is the FA3 "warp scheduler barrier") — `$F/flash_fwd_sm90.py:1479-1545`:
```python
@cute.jit
def mma_init(self):
    warp_group_idx = utils.canonical_warp_group_idx(sync=False)
    if const_expr(self.use_scheduler_barrier):
        if warp_group_idx == 1:
            cute.arch.barrier_arrive(barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                                     number_of_threads=2 * self.num_threads_per_warp_group)

def warp_scheduler_barrier_sync(self):
    if const_expr(self.use_scheduler_barrier):
        cute.arch.barrier(barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) - 1
                                     + utils.canonical_warp_group_idx(sync=False),
                          number_of_threads=2 * self.num_threads_per_warp_group)

def warp_scheduler_barrier_arrive(self):
    if const_expr(self.use_scheduler_barrier):
        cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
        next_wg = 1 - cur_wg if const_expr(self.num_wg_mma == 2) else (cur_wg + 1) % self.num_wg_mma
        cute.arch.barrier_arrive(barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                                 number_of_threads=2 * self.num_threads_per_warp_group)
```
with `utils.canonical_warp_group_idx` — `$F/utils.py:499-504`:
```python
def canonical_warp_group_idx(sync: bool = True) -> cutlass.Int32:
    warp_group_idx = cute.arch.thread_idx()[0] // 128
    if const_expr(sync): warp_group_idx = cute.arch.make_warp_uniform(warp_group_idx)
    return warp_group_idx
```
Enabled when `num_wg_mma >= 2 and tile_hdim <= 128` (`$F/flash_fwd_sm90.py:220-224`) — i.e. **on** for Wan.

### Register budget
`$F/flash_fwd_sm90.py:215-231`:
```python
self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[self.num_wg_mma]
...
if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
    self.num_mma_regs, self.num_producer_regs = 224, 40
```
applied at `$F/flash_fwd_sm90.py:580-604`:
```python
if warp_idx < 4:   # Producer
    cute.arch.setmaxregister_decrease(self.num_producer_regs)
    self.load(...)
else:              # Consumer
    cute.arch.setmaxregister_increase(self.num_mma_regs)
    tidx, _, _ = cute.arch.thread_idx()
    tidx = tidx - 128
    self.mma(...)
```
**For Wan (2 MMA WGs, all-TMA): `(240, 24)`.**

---

## 7. Pitfalls (from in-tree comments)

1. **`const_expr` vs `Constexpr` vs `range` vs `range_constexpr`.**
   - `cutlass.const_expr(x)` (`$C/base_dsl/ast_helpers.py:400-420`): *"check if the expression is a python
     value… If the expression is a dynamic expression, raise an error."* Use it to guard structural
     (shape/branch-elimination) decisions. It **raises** at trace time if you feed it an `Int32`.
   - `cutlass.Constexpr[T]` as a *type annotation* forces a parameter to be baked in
     (`$F/flash_fwd_sm90.py:436-437`: `TileScheduler: cutlass.Constexpr[Callable]`,
     `SharedStorage: cutlass.Constexpr[Callable]`). quack even patches the TVM-FFI arg converter so
     `Constexpr`-annotated fields become `spec.ConstNone` and are *not* runtime args
     (`$Q/cute_dsl_utils.py:31-53`).
   - `cutlass.range(stop, unroll=0, unroll_full=False, prefetch_stages=None, vectorize=None,
     at_least_once=False)` (`$C/base_dsl/ast_helpers.py:332-380`) → **dynamic trip count**, becomes an
     `scf.for`. FA4 always passes `unroll=1` for the KV mainloop
     (`$F/flash_fwd_sm90.py:811, 834, 1143, 1160`) and `unroll_full=True` for register-resident row loops
     (`$F/softmax.py:150, 207, 239`).
   - `cutlass.range_constexpr(...)` (`:391-393`) → fully unrolled Python loop, trip count must be a Python int.
   - `cutlass.min/max` (`$C/cutlass_dsl/cutlass.py:2369, 2432`) for dynamic values; `cutlass.if_generate`
     (`:2832`) to build a predicated region from a lambda.

2. **Why `mark_layout_dynamic`.** `from_dlpack` bakes the concrete shape/stride in as *static* constants, so
   each new shape would recompile. `mark_layout_dynamic(leading_dim=k)` keeps mode `k` at stride 1 (needed for
   vectorized 128-bit access + TMA) and makes the rest runtime values. Warning in
   `$F/cute_dsl_utils.py:116-123`:
   > "This is useful for compile keys since CuTe's `mark_layout_dynamic()` keeps stride=0 as static, meaning
   > kernels compiled with different broadcast patterns are not interchangeable."
   **For fixed Wan shapes, skip this entirely and go fully static.**

3. **Alignment must be *asserted*, it isn't inferred.** `$F/cute_dsl_utils.py:44-59`:
   ```python
   def assume_strides_aligned(t):
       """Assume all strides except the last are divisible by 128 bits."""
       divby = 128 // t.element_type.width
       strides = tuple(s if isinstance(s, int) else cute.assume(s, divby=divby) for s in t.stride[:-1])
       return (*strides, t.stride[-1])
   def assume_tensor_aligned(t):
       if t is None: return None
       return cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=assume_strides_aligned(t)))
   ```
   applied first thing in `__call__` (`$F/flash_fwd_sm90.py:194`). Without it the compiler can't prove
   128-bit vectorization is legal on dynamic strides.

4. **Mutating a shared MMA atom breaks SSA.** `$Q/sm90_utils.py:112-114`:
   > "We make a new mma_atom since we'll be modifying its attribute (accumulate). Otherwise the compiler
   > complains `operand #0 does not dominate this use`."

5. **fp32→bf16 conversion may not vectorize.** `$F/flash_fwd_sm90.py:1390-1392` (quoted in §4.3).

6. **`elect_one` around TMA copies deadlocks.** `$C/cute/arch/elect.py:117-120`.

7. **`barrier_arrive(aligned=True)` (default) must be outside divergent control flow.**
   `$C/cute/arch/nvvm_wrappers.py:723-732`.

8. **Pipeline `defer_sync`.** If you create N pipelines, set `defer_sync=True` on all but the last (or all,
   then call `pipeline_init_arrive`/`pipeline_init_wait` yourself) — otherwise you pay N block syncs.
   Compare `$F/flash_fwd_sm90.py:465,485,491` (all `defer_sync=True` + explicit
   `pipeline_init_arrive/wait`) vs `$F/flash_bwd_sm90.py:698` (`defer_sync=False` on the *last* pipeline only).

9. **Frozen dataclass pipelines.** `PipelineAsync` is `@dataclass(frozen=True)` (`$C/pipeline/sm90.py:42`),
   which is why FA4 subclasses with `object.__setattr__(obj, "__class__", child_cls)` —
   `$F/pipeline.py:20-30`:
   ```python
   def _override_create(parent_cls, child_cls):
       @staticmethod
       def create(*args, **kwargs):
           obj = parent_cls.create(*args, **kwargs)
           # Can't assign to __class__ directly since the dataclass is frozen
           object.__setattr__(obj, "__class__", child_cls)
           return obj
       return create
   ```

10. **`get_tensor(layout, swizzle=...)`** raises `TypeError` if you pass a `ComposedLayout` *and* a swizzle
    (`$C/cute/core.py:5623-5624`) — always `layout.outer` + `swizzle=layout.inner`.

11. **`phase` doesn't need modulo-2** despite the PTX docs — `$F/pipeline.py:65-67`:
    > "PTX docs say that the phase parity needs to be 0 or 1, so by right we need to take modulo 2. But in
    > practice just passing the phase in without modulo works fine."

12. **`cute.arch.exp2` and `warpgroup_reg_alloc/dealloc` are deprecated** in 4.6.0 (§5.2, §6) — new code
    should use `cute.math.exp2(..., fastmath=True)` and `setmaxregister_increase/decrease`.

### Reference configs for head_dim=128 bf16 (from FA4's tuned tables)
Forward, `$F/interface.py:150-151`:
```python
elif head_dim <= 128:
    return FwdConfig(128, 128, True, True)   # tile_m=128, tile_n=128, mma_pv_is_rs=True, intra_wg_overlap=True
```
with `num_stages=2` for sm90 (`$F/interface.py:866-867`, hardcoded).
Backward, `$F/interface.py:199-212`:
```python
elif head_dim <= 128:
    m_block_size = 64 if (causal or local) else 80        # non-causal → 80
    return BwdConfig(m_block_size=m_block_size, n_block_size=128,
                     num_stages_Q=2, num_stages_dO=2, num_stages_PdS=2,
                     SdP_swapAB=True, dKV_swapAB=False,
                     dQ_swapAB=m_block_size % 64 != 0,     # True for 80
                     AtomLayoutMSdP=1, AtomLayoutNdKV=2, AtomLayoutMdQ=1)
```

---

## 8. Minimal-viable sm90 warp-specialized skeleton

Every call below is verified present in this install. Assumes bf16, d=128, MHA, non-causal, b=1,
`tile_m=tile_n=128`, `num_stages=2`, 2 MMA warpgroups + 1 producer warpgroup → `block=[384,1,1]`.

```python
import math
from functools import partial
import cuda.bindings.driver as cuda
import cutlass, cutlass.cute as cute
from cutlass import Float32, Int32, BFloat16, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from quack import sm90_utils, copy_utils, layout_utils

class WanAttnFwdSm90:
    def __init__(self, tile_m=128, tile_n=128, hdim=128, num_stages=2):
        self.tile_m, self.tile_n, self.hdim, self.num_stages = tile_m, tile_n, hdim, num_stages
        self.dtype = BFloat16
        self.num_wg_mma = tile_m // 64                      # 2   ($F/flash_fwd_sm90.py:207-211)
        self.num_mma_threads = 128 * self.num_wg_mma        # 256
        self.num_threads = 128 * (self.num_wg_mma + 1)      # 384
        self.num_mma_regs, self.num_producer_regs = 240, 24 # ($F/flash_fwd_sm90.py:215-217)

    # ---------------- HOST ----------------
    @cute.jit
    def __call__(self, mQ, mK, mV, mO, mLSE, softmax_scale: Float32, stream: cuda.CUstream = None):
        # (b,s,h,d) -> (s,d,h,b) so mode 0 = tiled seq, mode 1 = contiguous hdim  ($F/..._sm90.py:195-198)
        mQ, mO = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mQ, mO)]
        mK, mV = [layout_utils.select(t, [1, 3, 2, 0]) for t in (mK, mV)]
        mLSE   = layout_utils.select(mLSE, [2, 1, 0])

        # 1) swizzled smem layouts: K_SW128 for bf16 d=128        ($Q/sm90_utils.py:14-38)
        sQ_layout, sK_layout, sV_layout, sO_layout = [
            sm90_utils.make_smem_layout(self.dtype, LayoutEnum.ROW_MAJOR, shape, stage)
            for shape, stage in [((self.tile_m, self.hdim), None),
                                 ((self.tile_n, self.hdim), self.num_stages),
                                 ((self.tile_n, self.hdim), self.num_stages),
                                 ((self.tile_m, self.hdim), None)]]

        # 2) WGMMAs: S=Q@Kt (K-major,K-major); O+=P@V (A from RMEM, B MN-major = Vt)
        tiled_mma_qk = cutlass.utils.sm90_make_trivial_tiled_mma(
            self.dtype, self.dtype, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.K,
            Float32, atom_layout_mnk=(self.num_wg_mma, 1, 1), tiler_mn=(64, self.tile_n))
        tiled_mma_pv = cutlass.utils.sm90_make_trivial_tiled_mma(
            self.dtype, self.dtype, warpgroup.OperandMajorMode.K, warpgroup.OperandMajorMode.MN,
            Float32, atom_layout_mnk=(self.num_wg_mma, 1, 1), tiler_mn=(64, self.hdim),
            a_source=warpgroup.OperandSource.RMEM)

        # 3) TMA atoms + tx byte counts                          ($F/..._sm90.py:260-314)
        self.tx = {n: cute.size_in_bytes(self.dtype, cute.select(l, mode=[0, 1]))
                   for n, l in [("Q", sQ_layout), ("K", sK_layout), ("V", sV_layout)]}
        tma_atom_Q, tQ = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mQ, sQ_layout, (self.tile_m, self.hdim))
        tma_atom_K, tK = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mK, cute.select(sK_layout, mode=[0,1]),
            (self.tile_n, self.hdim), 1)
        tma_atom_V, tV = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), mV, cute.select(sV_layout, mode=[0,1]),
            (self.tile_n, self.hdim), 1)
        tma_atom_O, tO = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), mO, sO_layout, (self.tile_m, self.hdim))

        # 4) smem storage struct                                 ($F/..._sm90.py:120-155)
        A = lambda l: cute.struct.Align[cute.struct.MemRange[self.dtype, cute.cosize(l)], 1024]
        @cute.struct
        class Storage:
            mbar_Q: cute.struct.MemRange[cutlass.Int64, 1 * 2]
            mbar_K: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            mbar_V: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            sV: A(sV_layout); sQ: A(sQ_layout); sK: A(sK_layout)

        grid = (cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m), cute.size(mQ.shape[2]), 1)
        self.kernel(tQ, tK, tV, tO, mLSE, tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O,
                    Float32(softmax_scale * math.log2(math.e)), sQ_layout, sK_layout, sV_layout,
                    sO_layout, tiled_mma_qk, tiled_mma_pv, Storage
        ).launch(grid=grid, block=[self.num_threads, 1, 1], stream=stream, min_blocks_per_mp=1)

    # ---------------- DEVICE ----------------
    @cute.kernel
    def kernel(self, mQ, mK, mV, mO, mLSE, ta_Q, ta_K, ta_V, ta_O, scale_log2,
               sQl, sKl, sVl, sOl, mma_qk, mma_pv, Storage: cutlass.Constexpr):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:                                        # ($F/..._sm90.py:443-446)
            for a in (ta_Q, ta_K, ta_V, ta_O): cpasync.prefetch_descriptor(a)

        st = cutlass.utils.SmemAllocator().allocate(Storage)      # (:448-449)
        G  = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        mk = lambda ptr, ns, tx: pipeline.PipelineTmaAsync.create(
            barrier_storage=ptr, num_stages=ns, producer_group=G(1),
            consumer_group=G(self.num_mma_threads // cute.arch.WARP_SIZE),
            tx_count=tx, defer_sync=True)                        # (:479-486)
        pQ = mk(st.mbar_Q.data_ptr(), 1, self.tx["Q"])
        pK = mk(st.mbar_K.data_ptr(), self.num_stages, self.tx["K"])
        pV = mk(st.mbar_V.data_ptr(), self.num_stages, self.tx["V"])
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)          # (:516)

        sQ = st.sQ.get_tensor(sQl.outer, swizzle=sQl.inner)                     # (:521-528)
        sK = st.sK.get_tensor(sKl.outer, swizzle=sKl.inner)
        sV = st.sV.get_tensor(sVl.outer, swizzle=sVl.inner)
        sVt = layout_utils.transpose_view(sV)                                   # (:530)
        sO = st.sQ.get_tensor(sOl.outer, swizzle=sOl.inner, dtype=self.dtype)   # (:535, aliases sQ)

        m_block, head, _ = cute.arch.block_idx()
        n_blocks = cute.ceil_div(cute.size(mK.shape[0]), self.tile_n)
        pipeline_init_wait(cluster_shape_mn=(1, 1))                             # (:578)

        # ============ PRODUCER (warps 0-3, 1 issuing warp) ============
        if warp_idx < 4:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)           # (:581)
            if warp_idx == 0:
                gQ = cute.local_tile(mQ[None,None,head,0], (self.tile_m, self.hdim), (m_block, 0))
                gK = cute.local_tile(mK[None,None,head,0], (self.tile_n, self.hdim), (None, 0))
                gV = cute.local_tile(mV[None,None,head,0], (self.tile_n, self.hdim), (None, 0))
                ldQ, _, _ = copy_utils.tma_get_copy_fn(ta_Q, 0, cute.make_layout(1), gQ, sQ,
                                                       single_stage=True)       # (:687-690)
                ldK, _, _ = copy_utils.tma_get_copy_fn(ta_K, 0, cute.make_layout(1), gK, sK)
                ldV, _, _ = copy_utils.tma_get_copy_fn(ta_V, 0, cute.make_layout(1), gV, sV)
                ldK = copy_utils.tma_producer_copy_fn(ldK, pK)                  # (:717)
                ldV = copy_utils.tma_producer_copy_fn(ldV, pV)
                pQ.producer_acquire_w_index_phase(0, Int32(1))                  # (:792-794)
                ldQ(tma_bar_ptr=pQ.sync_object_full.get_barrier(0))
                ps = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer,
                                                  self.num_stages)              # (:671-673)
                for i in cutlass.range(n_blocks, unroll=1):                     # (:811)
                    pK.producer_acquire(ps); ldK(src_idx=i, producer_state=ps); pK.producer_commit(ps)
                    pV.producer_acquire(ps); ldV(src_idx=i, producer_state=ps); pV.producer_commit(ps)
                    ps.advance()
                pV.producer_tail(ps)                                            # (:914)

        # ============ CONSUMER (warps 4-11 = 2 WGs) ============
        else:
            cute.arch.setmaxregister_increase(self.num_mma_regs)                # (:604)
            tidx, _, _ = cute.arch.thread_idx(); tidx = tidx - 128              # (:608-609)
            wg = cute.arch.make_warp_uniform(tidx // 128)
            wg_layout = cute.make_layout(self.num_wg_mma, stride=128)           # (:967-969)
            thr_qk = mma_qk.get_slice(tidx)
            _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
                mma_qk.get_slice(wg_layout(wg)), (self.tile_m, self.tile_n, self.hdim), sQ, sK)
            acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
                mma_pv.get_slice(wg_layout(wg)), (self.tile_m, self.hdim, self.tile_n), None, sVt)
            row_max = cute.make_rmem_tensor(acc_O.shape[0][0] * acc_O.shape[1], Float32)
            row_sum = cute.make_rmem_tensor(acc_O.shape[0][0] * acc_O.shape[1], Float32)

            pQ.consumer_wait_w_index_phase(0, Int32(0))                         # (:1099)
            cs = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_stages)
            for i in cutlass.range(n_blocks, unroll=1):
                pK.consumer_wait(cs, pK.consumer_try_wait(cs))                  # (:1368)
                acc_S = sm90_utils.gemm_zero_init(mma_qk, (self.tile_m, self.tile_n),
                                                  tSrQ, tSrK, B_idx=cs.index, wg_wait=-1)
                warpgroup.wait_group(0)                                         # (:1372)
                pK.consumer_release(cs)
                # --- seqlen mask on the last partial KV tile (S_kv % 128 != 0) ---
                acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)                # ($F/mask.py:193-219)
                # --- online softmax: fmax_reduce -> quad reduce -> exp2 -> rescale acc_O ---
                #     (see $F/softmax.py:150-190 for the exact 12-line body)
                tOrP.store(layout_utils.reshape_acc_to_frgA(acc_S).load().to(self.dtype))  # (:1303-1309)
                pV.consumer_wait(cs, pV.consumer_try_wait(cs))                  # (:1402)
                sm90_utils.gemm_w_idx(mma_pv, acc_O, tOrP, tOrVt,
                                      zero_init=(i == 0), B_idx=cs.index, wg_wait=0)
                pV.consumer_release(cs); cs.advance()
            pQ.consumer_release_w_index(0)                                      # (:1182)

            # ---------------- EPILOGUE ($F/flash_fwd.py:347-420) ----------------
            rO = cute.make_fragment_like(acc_O, self.dtype); rO.store(acc_O.load().to(self.dtype))
            cute.arch.barrier(barrier_id=1, number_of_threads=self.num_mma_threads)  # 1 = Epilogue
            st_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(transpose=False, num_matrices=4), self.dtype)
            thr_st = cute.make_tiled_copy_C(st_atom, mma_pv).get_slice(tidx)
            cute.copy(st_atom, thr_st.retile(rO), thr_st.partition_D(sO))
            cute.arch.fence_view_async_shared()                                 # ($F/flash_fwd.py:409)
            cute.arch.barrier_arrive(barrier_id=1,
                                     number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE)
            gO = cute.local_tile(mO[None,None,head,0], (self.tile_m, self.hdim), (m_block, 0))
            stO, _, _ = copy_utils.tma_get_copy_fn(ta_O, 0, cute.make_layout(1), sO, gO,
                                                   single_stage=True)
            if cute.arch.make_warp_uniform(cute.arch.warp_idx()) == 4:
                cute.arch.barrier(barrier_id=1,
                                  number_of_threads=self.num_mma_threads + cute.arch.WARP_SIZE)
                stO()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
```

**Refinements to add once this compiles and is numerically correct** (all present in FA4, cited above):
1. Warp-scheduler ping-pong barriers around each GEMM (`$F/flash_fwd_sm90.py:1524-1545`) — required to keep
   the two MMA WGs from colliding on the tensor cores; ~15-20% on 2-WG configs.
2. Intra-warpgroup overlap: issue QK for block `i+1` before PV for block `i`, then
   `warpgroup.wait_group(1)` (`$F/flash_fwd_sm90.py:1429-1477`).
3. `utils.cvt_f16` PTX path if the SASS shows scalar fp32→bf16 (`$F/flash_fwd_sm90.py:1390-1393`).
4. `PipelineStateSimple` (`$F/pipeline.py:38-96`) to shave the pipeline state to one register.
5. R2P bitmask masking on the final KV tile (`$F/mask.py:213-222`).
