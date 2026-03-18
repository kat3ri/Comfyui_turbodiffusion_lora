"""
Unit tests for the TurboWan LoRA loader (nodes/lora_loader.py).

These tests exercise the four supported LoRA weight-delta formats:

  1. Diffusers / PEFT   (.lora_A.weight / .lora_B.weight)
  2. Kohya              (.lora_down.weight / .lora_up.weight)
  3. Direct weight/bias (.weight / .bias  — full-rank delta)
  4. Bare module-path   (bare key, 2-D tensor = weight delta)

They also cover:
  - prefix remapping (diffusion_model.* → blocks.*)
  - prefix auto-detection
  - mixed-prefix files
  - _parse_direct_weight_keys helper
"""

import sys
import os

import pytest
import torch
import torch.nn as nn

# conftest.py (loaded automatically by pytest) stubs out all ComfyUI modules
# before this file is imported, so the relative-import chain below works.
# _REPO_ROOT is the package root (parent of the tests/ directory).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from nodes.lora_loader import (  # noqa: E402
    _parse_lora_keys,
    _parse_direct_weight_keys,
    _apply_lora_to_module,
    _remap_key,
    _auto_detect_prefix_mapping,
    LoRAMergedModelLoader,
)


# ---------------------------------------------------------------------------
# Tiny model that mirrors the WanModel blocks structure
# ---------------------------------------------------------------------------

class _TinyAttn(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)


class _TinyBlock(nn.Module):
    def __init__(self, dim=8, ffn_dim=16):
        super().__init__()
        self.self_attn = _TinyAttn(dim)
        self.cross_attn = _TinyAttn(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )


class _TinyModel(nn.Module):
    def __init__(self, dim=8, ffn_dim=16, num_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [_TinyBlock(dim, ffn_dim) for _ in range(num_blocks)]
        )


DIM = 8
FFN_DIM = 16
RANK = 4


def _make_model():
    """Return a fresh, zeroed _TinyModel (weights set to zero for easy delta checks)."""
    model = _TinyModel(DIM, FFN_DIM, num_blocks=2)
    for p in model.parameters():
        p.data.zero_()
    return model


# ---------------------------------------------------------------------------
# Tests: _parse_lora_keys
# ---------------------------------------------------------------------------

class TestParseLoraKeys:
    def test_peft_format(self):
        sd = {
            "diffusion_model.blocks.0.cross_attn.o.lora_A.weight": torch.randn(RANK, DIM),
            "diffusion_model.blocks.0.cross_attn.o.lora_B.weight": torch.randn(DIM, RANK),
        }
        pairs = _parse_lora_keys(sd)
        assert "diffusion_model.blocks.0.cross_attn.o" in pairs
        assert "down" in pairs["diffusion_model.blocks.0.cross_attn.o"]
        assert "up" in pairs["diffusion_model.blocks.0.cross_attn.o"]

    def test_kohya_format(self):
        sd = {
            "diffusion_model.blocks.0.ffn.0.lora_down.weight": torch.randn(RANK, DIM),
            "diffusion_model.blocks.0.ffn.0.lora_up.weight": torch.randn(FFN_DIM, RANK),
        }
        pairs = _parse_lora_keys(sd)
        assert "diffusion_model.blocks.0.ffn.0" in pairs

    def test_alpha_parsed(self):
        sd = {
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight": torch.randn(RANK, DIM),
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight": torch.randn(DIM, RANK),
            "diffusion_model.blocks.0.self_attn.q.alpha": torch.tensor(float(RANK)),
        }
        pairs = _parse_lora_keys(sd)
        assert pairs["diffusion_model.blocks.0.self_attn.q"]["alpha"] == float(RANK)

    def test_direct_weight_returns_empty(self):
        """_parse_lora_keys must NOT consume direct .weight/.bias keys."""
        sd = {
            "blocks.0.cross_attn.k.weight": torch.randn(DIM, DIM),
            "blocks.0.cross_attn.k.bias": torch.randn(DIM),
        }
        pairs = _parse_lora_keys(sd)
        assert len(pairs) == 0

    def test_bare_module_path_returns_empty(self):
        """_parse_lora_keys must NOT consume bare module-path keys."""
        sd = {
            "diffusion_model.blocks.0.cross_attn.o": torch.randn(DIM, DIM),
        }
        pairs = _parse_lora_keys(sd)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# Tests: _parse_direct_weight_keys
# ---------------------------------------------------------------------------

class TestParseDirectWeightKeys:
    def test_weight_and_bias(self):
        sd = {
            "blocks.0.cross_attn.k.weight": torch.randn(DIM, DIM),
            "blocks.0.cross_attn.k.bias": torch.randn(DIM),
        }
        entries = _parse_direct_weight_keys(sd)
        assert "blocks.0.cross_attn.k" in entries
        assert "weight" in entries["blocks.0.cross_attn.k"]
        assert "bias" in entries["blocks.0.cross_attn.k"]

    def test_weight_only(self):
        sd = {
            "diffusion_model.blocks.0.ffn.0.weight": torch.randn(FFN_DIM, DIM),
        }
        entries = _parse_direct_weight_keys(sd)
        assert "diffusion_model.blocks.0.ffn.0" in entries
        assert "bias" not in entries.get("diffusion_model.blocks.0.ffn.0", {})

    def test_bare_module_path_2d(self):
        sd = {
            "diffusion_model.blocks.0.cross_attn.o": torch.randn(DIM, DIM),
        }
        entries = _parse_direct_weight_keys(sd)
        assert "diffusion_model.blocks.0.cross_attn.o" in entries
        assert "weight" in entries["diffusion_model.blocks.0.cross_attn.o"]

    def test_bare_module_path_1d_included(self):
        """1-D bare-path tensors are included (treated as potential bias/scale)."""
        sd = {
            "diffusion_model.blocks.0.cross_attn.o": torch.randn(DIM),
        }
        entries = _parse_direct_weight_keys(sd)
        assert "diffusion_model.blocks.0.cross_attn.o" in entries

    def test_lora_suffixes_excluded(self):
        """LoRA-format keys must NOT appear in direct entries."""
        sd = {
            "diffusion_model.blocks.0.cross_attn.o.lora_A.weight": torch.randn(RANK, DIM),
            "diffusion_model.blocks.0.cross_attn.o.lora_B.weight": torch.randn(DIM, RANK),
            "diffusion_model.blocks.0.cross_attn.o.lora_down.weight": torch.randn(RANK, DIM),
            "diffusion_model.blocks.0.cross_attn.o.lora_up.weight": torch.randn(DIM, RANK),
            "diffusion_model.blocks.0.cross_attn.o.alpha": torch.tensor(float(RANK)),
        }
        entries = _parse_direct_weight_keys(sd)
        assert len(entries) == 0

    def test_non_tensor_skipped(self):
        sd = {
            "blocks.0.cross_attn.k.weight": torch.randn(DIM, DIM),
            "some_metadata": "this is a string, not a tensor",
        }
        entries = _parse_direct_weight_keys(sd)
        assert "some_metadata" not in entries


# ---------------------------------------------------------------------------
# Tests: _apply_lora_to_module  —  Pass 1 (low-rank LoRA)
# ---------------------------------------------------------------------------

class TestApplyLoraToModuleLowRank:
    def test_peft_format_merges_correctly(self):
        model = _make_model()
        # Create a known lora_down / lora_up so we can verify the delta.
        lora_down = torch.ones(RANK, DIM) * 0.1
        lora_up = torch.ones(DIM, RANK) * 0.2
        expected_delta = lora_up @ lora_down  # strength=1.0, alpha=rank → scale=1

        sd = {
            "diffusion_model.blocks.0.cross_attn.o.lora_A.weight": lora_down,
            "diffusion_model.blocks.0.cross_attn.o.lora_B.weight": lora_up,
        }
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1
        assert n_skipped == 0
        result = model.blocks[0].cross_attn.o.weight.data
        assert torch.allclose(result, expected_delta.to(result.dtype), atol=1e-5)

    def test_kohya_format_merges_correctly(self):
        model = _make_model()
        lora_down = torch.ones(RANK, DIM) * 0.1
        lora_up = torch.ones(FFN_DIM, RANK) * 0.2
        expected_delta = lora_up @ lora_down

        sd = {
            "diffusion_model.blocks.0.ffn.0.lora_down.weight": lora_down,
            "diffusion_model.blocks.0.ffn.0.lora_up.weight": lora_up,
        }
        n_applied, _ = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1
        result = model.blocks[0].ffn[0].weight.data
        assert torch.allclose(result, expected_delta.to(result.dtype), atol=1e-5)

    def test_strength_scales_delta(self):
        model = _make_model()
        lora_down = torch.ones(RANK, DIM)
        lora_up = torch.ones(DIM, RANK)
        strength = 0.5

        sd = {
            "diffusion_model.blocks.0.self_attn.k.lora_down.weight": lora_down,
            "diffusion_model.blocks.0.self_attn.k.lora_up.weight": lora_up,
        }
        _apply_lora_to_module(
            model, sd, strength=strength,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        expected = (lora_up @ lora_down) * strength
        result = model.blocks[0].self_attn.k.weight.data
        assert torch.allclose(result, expected.to(result.dtype), atol=1e-5)

    def test_alpha_scaling(self):
        model = _make_model()
        lora_down = torch.ones(RANK, DIM)
        lora_up = torch.ones(DIM, RANK)
        alpha = 2.0  # alpha/rank = 0.5

        sd = {
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight": lora_down,
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight": lora_up,
            "diffusion_model.blocks.0.self_attn.q.alpha": torch.tensor(alpha),
        }
        _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        expected = (lora_up @ lora_down) * (alpha / RANK)
        result = model.blocks[0].self_attn.q.weight.data
        assert torch.allclose(result, expected.to(result.dtype), atol=1e-5)

    def test_auto_detection_wrong_prefix(self):
        """If the specified prefix is wrong, auto-detection should find the right one."""
        model = _make_model()
        lora_down = torch.ones(RANK, DIM)
        lora_up = torch.ones(DIM, RANK)

        sd = {
            "diffusion_model.blocks.0.cross_attn.v.lora_down.weight": lora_down,
            "diffusion_model.blocks.0.cross_attn.v.lora_up.weight": lora_up,
        }
        # Pass wrong prefix — auto-detection must fix it.
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="transformer.", model_key_prefix="",
        )
        assert n_applied == 1
        assert n_skipped == 0

    def test_no_prefix_keys(self):
        """LoRA file with no prefix at all (blocks.* directly)."""
        model = _make_model()
        lora_down = torch.ones(RANK, DIM)
        lora_up = torch.ones(DIM, RANK)

        sd = {
            "blocks.0.self_attn.v.lora_down.weight": lora_down,
            "blocks.0.self_attn.v.lora_up.weight": lora_up,
        }
        n_applied, _ = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1


# ---------------------------------------------------------------------------
# Tests: _apply_lora_to_module  —  Pass 2 (direct weight/bias delta)
# ---------------------------------------------------------------------------

class TestApplyLoraToModuleDirectWeight:
    def test_direct_weight_no_prefix(self):
        """Direct .weight key with no prefix matches model directly."""
        model = _make_model()
        delta = torch.ones(DIM, DIM) * 0.5

        sd = {"blocks.0.cross_attn.k.weight": delta}
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1
        assert n_skipped == 0
        result = model.blocks[0].cross_attn.k.weight.data
        assert torch.allclose(result, delta.to(result.dtype), atol=1e-5)

    def test_direct_weight_diffusion_model_prefix(self):
        """Direct .weight key with diffusion_model. prefix."""
        model = _make_model()
        delta = torch.ones(DIM, DIM) * 0.3

        sd = {"diffusion_model.blocks.0.cross_attn.q.weight": delta}
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1
        result = model.blocks[0].cross_attn.q.weight.data
        assert torch.allclose(result, delta.to(result.dtype), atol=1e-5)

    def test_direct_weight_and_bias(self):
        """Direct .weight + .bias keys are both applied."""
        model = _make_model()
        w_delta = torch.ones(DIM, DIM) * 0.1
        b_delta = torch.ones(DIM) * 0.2

        sd = {
            "blocks.0.self_attn.o.weight": w_delta,
            "blocks.0.self_attn.o.bias": b_delta,
        }
        n_applied, _ = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1
        w_result = model.blocks[0].self_attn.o.weight.data
        b_result = model.blocks[0].self_attn.o.bias.data
        assert torch.allclose(w_result, w_delta.to(w_result.dtype), atol=1e-5)
        assert torch.allclose(b_result, b_delta.to(b_result.dtype), atol=1e-5)

    def test_direct_weight_strength_scales(self):
        model = _make_model()
        delta = torch.ones(DIM, DIM)
        strength = 0.25

        sd = {"blocks.0.cross_attn.v.weight": delta}
        _apply_lora_to_module(
            model, sd, strength=strength,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        result = model.blocks[0].cross_attn.v.weight.data
        assert torch.allclose(result, (delta * strength).to(result.dtype), atol=1e-5)

    def test_direct_weight_shape_mismatch_skipped(self):
        """Wrong-shape delta must be skipped, not raise."""
        model = _make_model()
        sd = {"blocks.0.cross_attn.o.weight": torch.ones(DIM + 1, DIM)}
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 0
        assert n_skipped == 1

    def test_direct_weight_ffn_sequential(self):
        """ffn.0 and ffn.2 are nn.Sequential children — direct delta must reach them."""
        model = _make_model()
        w0 = torch.ones(FFN_DIM, DIM) * 0.1
        w2 = torch.ones(DIM, FFN_DIM) * 0.2

        sd = {
            "diffusion_model.blocks.0.ffn.0.weight": w0,
            "diffusion_model.blocks.0.ffn.2.weight": w2,
        }
        n_applied, _ = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 2
        assert torch.allclose(model.blocks[0].ffn[0].weight.data, w0.to(model.blocks[0].ffn[0].weight.dtype), atol=1e-5)
        assert torch.allclose(model.blocks[0].ffn[2].weight.data, w2.to(model.blocks[0].ffn[2].weight.dtype), atol=1e-5)

    def test_bare_module_path_2d(self):
        """Bare module-path key (no .weight suffix) with 2-D tensor."""
        model = _make_model()
        delta = torch.ones(DIM, DIM) * 0.4

        sd = {"diffusion_model.blocks.0.cross_attn.o": delta}
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 1
        result = model.blocks[0].cross_attn.o.weight.data
        assert torch.allclose(result, delta.to(result.dtype), atol=1e-5)

    def test_bare_module_path_auto_prefix_detection(self):
        """Auto-detection for bare module-path keys (wrong initial prefix)."""
        model = _make_model()
        delta = torch.ones(DIM, DIM) * 0.5

        sd = {"diffusion_model.blocks.1.self_attn.k": delta}
        n_applied, _ = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="transformer.", model_key_prefix="",
        )
        assert n_applied == 1
        result = model.blocks[1].self_attn.k.weight.data
        assert torch.allclose(result, delta.to(result.dtype), atol=1e-5)


# ---------------------------------------------------------------------------
# Tests: mixed format (both low-rank LoRA AND direct weight in same file)
# ---------------------------------------------------------------------------

class TestApplyLoraToModuleMixed:
    def test_mixed_formats_both_applied(self):
        """A checkpoint that mixes low-rank and direct delta keys."""
        model = _make_model()
        lora_down = torch.ones(RANK, DIM) * 0.1
        lora_up = torch.ones(DIM, RANK) * 0.2
        direct_delta = torch.ones(DIM, DIM) * 0.7

        sd = {
            # Low-rank for self_attn.q
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight": lora_down,
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight": lora_up,
            # Direct weight for cross_attn.k
            "diffusion_model.blocks.0.cross_attn.k.weight": direct_delta,
        }
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="diffusion_model.", model_key_prefix="",
        )
        assert n_applied == 2
        assert n_skipped == 0

        expected_lora = lora_up @ lora_down
        assert torch.allclose(
            model.blocks[0].self_attn.q.weight.data,
            expected_lora.to(model.blocks[0].self_attn.q.weight.dtype),
            atol=1e-5,
        )
        assert torch.allclose(
            model.blocks[0].cross_attn.k.weight.data,
            direct_delta.to(model.blocks[0].cross_attn.k.weight.dtype),
            atol=1e-5,
        )


# ---------------------------------------------------------------------------
# Tests: non-linear modules are skipped
# ---------------------------------------------------------------------------

class TestNonLinearSkipped:
    def test_non_linear_skipped_direct(self):
        """If key maps to a non-Linear module it must be counted as skipped."""
        model = _make_model()
        # blocks.0 is a _TinyBlock, not nn.Linear
        sd = {"blocks.0.weight": torch.randn(DIM, DIM)}
        n_applied, n_skipped = _apply_lora_to_module(
            model, sd, strength=1.0,
            lora_key_prefix="", model_key_prefix="",
        )
        assert n_applied == 0
        assert n_skipped == 1


# ---------------------------------------------------------------------------
# Tests: LoRAMergedModelLoader always loads base model to CPU
# ---------------------------------------------------------------------------

class TestLoRAMergedModelLoaderCPULoad:
    """Verify that _ensure_loaded always passes target_device="cpu" to load_fn.

    The root cause of both the key-mismatch and the device-mismatch bugs was
    that _ensure_loaded passed target_device=None, which allowed
    _load_model_impl to fall back to base_loader._target_device.  If the base
    loader had previously been moved to CUDA (e.g. because it was also wired
    directly to an inference node in the same ComfyUI session), the model would
    be loaded with an offload wrapper already applied.  This produced two
    problems:

      1. Key mismatch — named_modules() of a LayerwiseGPUOffloadWrapper yields
         keys like "blocks.0.module.self_attn.v", not "blocks.0.self_attn.v",
         so every LoRA key lookup failed.
      2. Device mismatch — a second offload wrapper was applied on top of the
         first, causing the "mat1 is on cuda:0, different from other tensors on
         cpu" RuntimeError during inference.

    The fix: pass target_device="cpu" so the fallback is bypassed and the
    returned model is always a plain, unwrapped nn.Module on CPU.
    """

    def _make_lora_loader(self, base_target_device, lora_state_dict,
                          lora_key_prefix="", model_key_prefix=""):
        """Create a LoRAMergedModelLoader backed by a fake load_fn.

        Returns (loader, recorded_dict) where recorded_dict["target_device"]
        will hold the target_device value that load_fn received.
        """
        recorded = {}

        def load_fn(path, load_args, target_device=None):
            recorded["target_device"] = target_device
            return _make_model()

        from Comfyui_turbodiffusion_lora.utils.lazy_loader import LazyModelLoader as _FLL

        base_loader = _FLL(
            model_path="fake_path",
            model_name="fake_model",
            load_fn=load_fn,
            load_args=None,
        )
        # Simulate a base loader that was (or wasn't) previously moved to CUDA.
        base_loader._target_device = base_target_device

        loader = LoRAMergedModelLoader(
            base_loader=base_loader,
            lora_state_dict=lora_state_dict,
            strength=1.0,
            lora_key_prefix=lora_key_prefix,
            model_key_prefix=model_key_prefix,
            lora_name="test.safetensors",
        )
        # Keep _target_device=None so the offload wrapper path is not entered.
        loader._target_device = None
        return loader, recorded

    def test_cpu_passed_when_base_target_device_is_none(self):
        """load_fn must receive target_device='cpu' even when base loader has no CUDA target."""
        lora_sd = {
            "blocks.0.self_attn.q.lora_down.weight": torch.randn(RANK, DIM),
            "blocks.0.self_attn.q.lora_up.weight": torch.randn(DIM, RANK),
        }
        loader, recorded = self._make_lora_loader(base_target_device=None, lora_state_dict=lora_sd)
        loader._ensure_loaded()
        assert recorded.get("target_device") == "cpu", (
            "_ensure_loaded must pass target_device='cpu' to load_fn, "
            f"but got {recorded.get('target_device')!r}"
        )

    def test_cpu_passed_when_base_target_device_is_cuda(self):
        """load_fn must receive target_device='cpu' even when base loader targets CUDA.

        This is the critical scenario: if base_loader._target_device was set to
        a CUDA device (because the base model was used directly in the same
        session), passing None would have caused load_fn to load the model to
        CUDA with an offload wrapper applied, breaking LoRA key resolution and
        causing a double-wrap device mismatch.
        """
        lora_sd = {
            "blocks.0.self_attn.v.lora_down.weight": torch.randn(RANK, DIM),
            "blocks.0.self_attn.v.lora_up.weight": torch.randn(DIM, RANK),
        }
        loader, recorded = self._make_lora_loader(
            base_target_device="cuda:0", lora_state_dict=lora_sd
        )
        loader._ensure_loaded()
        assert recorded.get("target_device") == "cpu", (
            "_ensure_loaded must pass target_device='cpu' to load_fn, "
            f"but got {recorded.get('target_device')!r}"
        )
        assert loader._loaded is True

    def test_lora_applied_after_cpu_load(self):
        """LoRA weights must be correctly merged when base loader had CUDA target.

        With the previous bug (target_device=None), if base_loader._target_device
        was 'cuda:0', load_fn would load the model wrapped and the module map
        would use wrong key paths — causing all LoRA key lookups to fail.
        """
        lora_down = torch.randn(RANK, DIM)
        lora_up = torch.randn(DIM, RANK)
        lora_sd = {
            "diffusion_model.blocks.0.self_attn.q.lora_down.weight": lora_down,
            "diffusion_model.blocks.0.self_attn.q.lora_up.weight": lora_up,
        }
        loader, _ = self._make_lora_loader(
            base_target_device="cuda:0",
            lora_state_dict=lora_sd,
            lora_key_prefix="diffusion_model.",
            model_key_prefix="",
        )
        loader._ensure_loaded()

        model = loader._model
        expected = (lora_up @ lora_down).to(model.blocks[0].self_attn.q.weight.dtype)
        assert torch.allclose(
            model.blocks[0].self_attn.q.weight.data,
            expected,
            atol=1e-5,
        ), "LoRA delta was not applied — key mismatch likely occurred"




if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
