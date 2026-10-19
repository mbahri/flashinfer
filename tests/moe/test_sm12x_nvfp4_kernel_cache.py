"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

On-disk CuTe-DSL cache of the SM12x canonical NVFP4 MoE kernels
(``moe_nvfp4_w4a4`` / ``moe_nvfp4_w4a16``, unified-API backends
``sm12x_nvfp4`` / ``sm12x_nvfp4_bf16``).

Naming contract, as docs/design_docs/cute_dsl_kernel_cache.md asks of every
adopter: the kernel name is the sole per-kernel cache key, so

1. signature coverage: every codegen input of a kernel is a parameter of its
   name function;
2. per-argument perturbation: changing any single input changes the name --
   including the SM-count-derived grid of the persistent W4A16 kernel, which
   differs between GPUs of one arch (e.g. RTX PRO 6000 vs 4500, both sm_120);
3. names are symbol-safe and stable.

The GPU tests check the round trip: a kernel compiled and persisted once is
reloaded without compiling and computes the same result.
"""

import inspect
import re
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

import torch  # noqa: E402

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (  # noqa: E402
    moe_nvfp4_w4a4 as w4a4,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import (  # noqa: E402
    moe_nvfp4_w4a16 as w4a16,
)
from flashinfer.jit import cute_dsl_core  # noqa: E402

# Gemma-4 26B A4B MoE geometry.
H, I, E, TOPK = 2816, 704, 128, 8
N13 = 2 * I


def _args(T: int = 8, hidden: int = H, dtype=torch.bfloat16):
    return (
        torch.empty(T, hidden, dtype=dtype),
        torch.empty(T * TOPK, dtype=torch.int32),
        torch.empty(E, N13, H // 2, dtype=torch.uint8),
    )


W4A4_BASELINE = dict(T=8, H=H, I=I, N13=N13, E=E, TOPK=TOPK, act="silu")
W4A4_PERTURBED = dict(T=16, H=2048, I=768, N13=I, E=64, TOPK=4, act="geglu_tanh")


def test_w4a4_disk_name_covers_entry_key():
    """Every parameter of the in-process key (``_entry``) reaches the name."""
    entry = set(inspect.signature(w4a4._entry).parameters)
    name = set(inspect.signature(w4a4._disk_name).parameters)
    # ``dev`` selects the device, not the code (the cache module directory is
    # per compile arch); ``activation`` is resolved to ``act``.
    assert entry - {"dev", "activation"} | {"act", "args"} == name


@pytest.mark.parametrize("param", sorted(W4A4_BASELINE))
def test_w4a4_disk_name_varies_with_every_argument(param):
    base = w4a4._disk_name(**W4A4_BASELINE, args=_args())
    perturbed = w4a4._disk_name(
        **dict(W4A4_BASELINE, **{param: W4A4_PERTURBED[param]}), args=_args()
    )
    assert base != perturbed, f"w4a4 disk name ignores {param!r}"


@pytest.mark.parametrize(
    "args",
    [_args(T=16), _args(dtype=torch.float16), (_args()[0].t(),) + _args()[1:]],
    ids=["shape", "dtype", "stride"],
)
def test_disk_names_vary_with_the_compiled_signature(args):
    base = _args()
    assert w4a4._disk_name(**W4A4_BASELINE, args=base) != w4a4._disk_name(
        **W4A4_BASELINE, args=args
    )
    consts = dict(S=64, H=H, I=I, N13=N13, E=E)
    assert w4a16._disk_name(False, consts, base) != w4a16._disk_name(
        False, consts, args
    )


def _tiled_consts(T: int = 64, act: str = "silu"):
    S = T * TOPK
    sf13 = w4a16._scale_shape(E, N13, H)
    sf2 = w4a16._scale_shape(E, H, I)
    BM = w4a16._pick_bm(S, E)
    consts, _, _ = w4a16._tiled_plan(T, H, I, N13, E, TOPK, S, BM, sf13, sf2, act)
    return consts


@pytest.mark.parametrize("key", sorted(_tiled_consts()))
def test_w4a16_disk_name_varies_with_every_constant(key):
    consts = _tiled_consts()
    value = consts[key]
    changed = dict(
        consts, **{key: (not value) if isinstance(value, bool) else value + 1}
    )
    assert w4a16._disk_name(False, consts, _args()) != w4a16._disk_name(
        False, changed, _args()
    ), f"w4a16 disk name ignores plan constant {key!r}"


def test_w4a16_disk_name_distinguishes_launcher_and_activation():
    silu, geglu = _tiled_consts(act="silu"), _tiled_consts(act="geglu_tanh")
    names = {
        w4a16._disk_name(persistent, consts, _args())
        for persistent in (False, True)
        for consts in (silu, geglu)
    }
    assert len(names) == 4


def test_w4a16_persistent_disk_name_tracks_sm_count(monkeypatch):
    """The persistent grid follows the SM count, so two GPUs of one compile
    arch with different SM counts must not share an artifact."""
    T = 16  # S == E: one routed slot per expert tile -> persistent launcher
    S = T * TOPK
    sf13 = w4a16._scale_shape(E, N13, H)
    sf2 = w4a16._scale_shape(E, H, I)
    assert w4a16._pick_bm(S, E) == 1
    plans = {}
    for sms in (188, 82):  # RTX PRO 6000 / RTX PRO 4500, both sm_120
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda _dev, sms=sms: SimpleNamespace(multi_processor_count=sms),
        )
        plan = w4a16._persistent_plan(T, H, I, N13, TOPK, S, None, sf13, sf2, "silu")
        assert plan is not None
        plans[sms] = plan[0]
    assert plans[188] != plans[82], "pick a shape whose persistent grid depends on SMs"
    assert w4a16._disk_name(True, plans[188], _args()) != w4a16._disk_name(
        True, plans[82], _args()
    )


def test_disk_names_are_symbol_safe_and_stable():
    names = [
        w4a4._disk_name(**W4A4_BASELINE, args=_args()),
        w4a16._disk_name(False, _tiled_consts(), _args()),
        w4a16._disk_name(True, _tiled_consts(), _args()),
    ]
    for name in names:
        assert re.fullmatch(r"[0-9A-Za-z_]+", name), name
    assert names[0] == w4a4._disk_name(**dict(W4A4_BASELINE), args=_args())
    assert names[1] == w4a16._disk_name(False, dict(_tiled_consts()), _args())


def test_source_fingerprint_covers_both_modules():
    """W4A4 compiles constants defined in the W4A16 module, and both share the
    cache module, so both sources key both."""
    files = set(w4a16.kernel_source_files())
    assert files == {w4a4.__file__, w4a16.__file__}


# --------------------------------------------------------------------------
# GPU round trip
# --------------------------------------------------------------------------


def _require_sm12x():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from flashinfer.utils import get_compute_capability

    if get_compute_capability(torch.device("cuda")) not in ((12, 0), (12, 1)):
        pytest.skip("SM12x canonical NVFP4 MoE requires SM120/SM121")


@pytest.fixture
def disk_cache(monkeypatch, tmp_path):
    """A fresh on-disk cache directory and empty in-process caches."""
    monkeypatch.setattr(cute_dsl_core.jit_env, "FLASHINFER_JIT_DIR", tmp_path)
    monkeypatch.setattr(cute_dsl_core, "get_tmpdir", lambda: tmp_path)
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.delenv("FLASHINFER_DISABLE_JIT", raising=False)
    monkeypatch.setattr(w4a4, "_COMPILED", {})
    monkeypatch.setattr(w4a16, "_COMPILED", {})
    return tmp_path


def _artifacts(root):
    return sorted(p.name for p in root.glob(f"{w4a16.CUTE_DSL_MODULE}_*/*.o"))


def _forbid_compile(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("cute.compile ran although the artifact was cached")

    monkeypatch.setattr(w4a4.cute, "compile", fail)


def _inputs(T: int, h: int = 512, i: int = 192, e: int = 8, topk: int = 2):
    # Random E2M1 codes and E4M3 block scales; the 1/32 global weight scale
    # keeps gate values O(1), where SwiGLU and GeGLU-tanh differ (both tend to
    # x for large x) and BF16 stays finite.
    torch.manual_seed(0)
    dev = torch.device("cuda")
    n13 = 2 * i
    pad = lambda n, m: (n + m - 1) // m * m  # noqa: E731

    def scales(rows, k):
        sf = torch.rand(e, pad(rows, 128), pad(k // 16, 4), device=dev) * 0.5 + 0.25
        return sf.to(torch.float8_e4m3fn)

    return dict(
        hidden_states=torch.randn(T, h, dtype=torch.bfloat16, device=dev),
        topk_ids=torch.stack([torch.randperm(e)[:topk] for _ in range(T)])
        .to(torch.int32)
        .to(dev),
        topk_weights=torch.rand(T, topk, dtype=torch.float32, device=dev),
        w13=torch.randint(0, 256, (e, n13, h // 2), dtype=torch.uint8, device=dev),
        w13_sf=scales(n13, h),
        w13_gs=torch.full((e,), 1 / 32, dtype=torch.float32, device=dev),
        w2=torch.randint(0, 256, (e, h, i // 2), dtype=torch.uint8, device=dev),
        w2_sf=scales(h, i),
        w2_gs=torch.full((e,), 1 / 32, dtype=torch.float32, device=dev),
    )


def _run_w4a4(inputs, activation):
    dev = inputs["hidden_states"].device
    out = torch.empty_like(inputs["hidden_states"])
    one = torch.ones(1, dtype=torch.float32, device=dev)
    w4a4.run_moe_w4a4(**inputs, a13_gs=one, a2_gs=one, out=out, activation=activation)
    torch.cuda.synchronize()
    return out.float()


def _run_w4a16(inputs, activation):
    out = torch.empty_like(inputs["hidden_states"])
    w4a16.run_moe_w4a16(**inputs, out=out, activation=activation)
    torch.cuda.synchronize()
    return out.float()


def _rel(a, b):
    return ((a - b).norm() / b.norm()).item()


@pytest.mark.parametrize("activation", ["silu", "geglu_tanh"])
@pytest.mark.parametrize(
    "run,T,persistent",
    [(_run_w4a4, 32, None), (_run_w4a16, 4, True), (_run_w4a16, 64, False)],
    ids=["w4a4", "w4a16_persistent", "w4a16_tiled"],
)
def test_reloads_from_disk_without_compiling(
    disk_cache, monkeypatch, run, T, persistent, activation
):
    _require_sm12x()
    inputs = _inputs(T)
    if persistent is not None:
        h, (e, n13, _) = inputs["hidden_states"].shape[1], inputs["w13"].shape
        entry = w4a16._entry(
            T, h, n13 // 2, n13, e, 2, torch.device("cuda"), activation
        )
        assert entry[0] == persistent, "shape does not select the intended launcher"
    first = run(inputs, activation)
    assert torch.isfinite(first).all() and first.abs().sum() > 0
    artifacts = _artifacts(disk_cache)
    assert len(artifacts) == 1, artifacts

    # A new process: empty in-process cache, compiling forbidden.
    monkeypatch.setattr(w4a4, "_COMPILED", {})
    monkeypatch.setattr(w4a16, "_COMPILED", {})
    _forbid_compile(monkeypatch)
    reloaded = run(inputs, activation)
    assert _artifacts(disk_cache) == artifacts
    # GEMM2 reduces with BF16 atomics, so runs agree up to summation order.
    assert _rel(reloaded, first) < 1e-2


@pytest.mark.parametrize("run", [_run_w4a4, _run_w4a16], ids=["w4a4", "w4a16"])
def test_gated_activations_do_not_share_artifacts(disk_cache, monkeypatch, run):
    _require_sm12x()
    inputs = _inputs(32)
    silu = run(inputs, "silu")
    geglu = run(inputs, "geglu_tanh")
    assert len(_artifacts(disk_cache)) == 2
    monkeypatch.setattr(w4a4, "_COMPILED", {})
    monkeypatch.setattr(w4a16, "_COMPILED", {})
    _forbid_compile(monkeypatch)
    assert _rel(run(inputs, "geglu_tanh"), geglu) < 1e-2
    assert _rel(run(inputs, "silu"), silu) < 1e-2
    assert _rel(geglu, silu) > 5e-2


def test_disable_env_keeps_the_disk_cache_empty(disk_cache, monkeypatch):
    _require_sm12x()
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
    _run_w4a4(_inputs(8), "silu")
    assert _artifacts(disk_cache) == []
