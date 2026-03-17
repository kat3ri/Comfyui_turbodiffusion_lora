"""
TurboWan LoRA Loader - Merge LoRA weights into unquantized TurboDiffusion models.

Two LoRA key formats are supported:

  Diffusers/PEFT  — keys end in ``.lora_A.weight`` / ``.lora_B.weight``
  Kohya           — keys end in ``.lora_down.weight`` / ``.lora_up.weight``

The merged weight delta for each LoRA layer is computed as::

    delta_W = lora_up @ lora_down * (alpha / rank) * scale

where ``rank`` is inferred from the inner dimension of lora_down,
``alpha`` is read from the ``<base_key>.alpha`` key in the LoRA file
(defaults to ``rank`` when absent, which gives ``alpha/rank = 1``),
and ``scale`` is the user-supplied ``strength`` multiplier (default ``1.0``).

Only unquantized (plain float ``nn.Linear``) model layers are supported.
Quantized (``Int8Linear``) layers are silently skipped.

Key-prefix remapping:

  LoRA files trained on the original HuggingFace Wan model (e.g. via
  diffusers/PEFT) typically use ``transformer.`` as the top-level module
  prefix, while TurboDiffusion base models store weights under ``net.``.
  The defaults ``lora_key_prefix="transformer."`` and
  ``model_key_prefix="net."`` map between them automatically.

  If your LoRA already uses ``net.`` keys, set ``lora_key_prefix`` to
  ``"net."``.  If there is no prefix at all, set it to ``""``.
"""

import time
import torch
import torch.nn as nn
import folder_paths

from ..utils.lazy_loader import LazyModelLoader
from ..utils.timing import TimedLogger

# ---------------------------------------------------------------------------
# Safetensors support — required for .safetensors LoRA files.
# torch.load() cannot read the safetensors binary format and raises
# UnpicklingError ("invalid load key, 'x'") when given a .safetensors file.
# ---------------------------------------------------------------------------
try:
    from safetensors.torch import load_file as _safetensors_load_file

    _SAFETENSORS_AVAILABLE = True
except ImportError:
    _SAFETENSORS_AVAILABLE = False


def _load_lora_state_dict(path: str) -> dict:
    """Load a LoRA checkpoint from *path* using the correct loader.

    * ``.safetensors`` files are loaded with :func:`safetensors.torch.load_file`.
    * All other extensions (``.pt``, ``.pth``, …) fall back to
      :func:`torch.load` with ``weights_only=False`` (needed for checkpoints
      that store non-tensor metadata such as config dicts).

    Raises
    ------
    RuntimeError
        When a ``.safetensors`` file is requested but the ``safetensors``
        package is not installed.
    """
    if path.lower().endswith(".safetensors"):
        if not _SAFETENSORS_AVAILABLE:
            raise RuntimeError(
                "The 'safetensors' package is required to load .safetensors "
                "LoRA files.\nInstall it with:  pip install safetensors"
            )
        return _safetensors_load_file(path, device="cpu")

    # Legacy pickle-based checkpoint.  weights_only=False is required for
    # files that embed non-tensor objects (e.g. config dicts).  Only load
    # LoRA files from trusted sources — arbitrary pickle execution is possible.
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# LoRA key parsing helpers
# ---------------------------------------------------------------------------

def _parse_lora_keys(lora_state_dict: dict) -> dict:
    """Parse LoRA A/B (or down/up) pairs from a state dict.

    Returns a dict mapping *base_key* → ``{"down": tensor, "up": tensor,
    "alpha": float}``.  ``"alpha"`` is only present when an ``<base>.alpha``
    key was found in the checkpoint; otherwise the caller should default it to
    ``rank``.
    """
    pairs: dict = {}

    for key, tensor in lora_state_dict.items():
        # Diffusers / PEFT format
        if key.endswith(".lora_A.weight"):
            base = key[: -len(".lora_A.weight")]
            pairs.setdefault(base, {})["down"] = tensor
        elif key.endswith(".lora_B.weight"):
            base = key[: -len(".lora_B.weight")]
            pairs.setdefault(base, {})["up"] = tensor
        # Kohya format
        elif key.endswith(".lora_down.weight"):
            base = key[: -len(".lora_down.weight")]
            pairs.setdefault(base, {})["down"] = tensor
        elif key.endswith(".lora_up.weight"):
            base = key[: -len(".lora_up.weight")]
            pairs.setdefault(base, {})["up"] = tensor
        # Alpha scaling factor
        elif key.endswith(".alpha"):
            base = key[: -len(".alpha")]
            alpha_val = tensor.item() if isinstance(tensor, torch.Tensor) else float(tensor)
            pairs.setdefault(base, {})["alpha"] = alpha_val

    return pairs


def _remap_key(lora_key: str, lora_prefix: str, model_prefix: str) -> str:
    """Strip *lora_prefix* from *lora_key* and prepend *model_prefix*."""
    if lora_prefix and lora_key.startswith(lora_prefix):
        return model_prefix + lora_key[len(lora_prefix) :]
    return lora_key


# ---------------------------------------------------------------------------
# Core LoRA-merge routine
# ---------------------------------------------------------------------------

def _apply_lora_to_module(
    model: nn.Module,
    lora_state_dict: dict,
    strength: float,
    lora_key_prefix: str,
    model_key_prefix: str,
    logger=None,
) -> tuple:
    """Merge LoRA delta weights into *model* in-place (CPU).

    Only ``nn.Linear`` layers are targeted.  Any layer that is not a plain
    float ``nn.Linear`` (e.g. quantized ``Int8Linear`` layers) is silently
    skipped.

    For each matched layer the weight update is::

        delta_W = lora_up @ lora_down * (alpha / rank) * strength

    Returns:
        Tuple ``(n_applied, n_skipped)`` — count of layers successfully
        merged and layers that were skipped.
    """

    def _log(msg: str) -> None:
        if logger:
            logger.log(msg)

    pairs = _parse_lora_keys(lora_state_dict)

    # Build a fast name→module lookup (one pass over the whole model)
    module_map = {name: mod for name, mod in model.named_modules()}

    n_applied = 0
    n_skipped = 0

    for lora_base_key, matrices in pairs.items():
        if "down" not in matrices or "up" not in matrices:
            _log(f"Skip (incomplete pair): {lora_base_key}")
            n_skipped += 1
            continue

        # Map LoRA key → model module name
        model_key = _remap_key(lora_base_key, lora_key_prefix, model_key_prefix)

        module = module_map.get(model_key)
        if module is None:
            _log(f"Skip (no match): {model_key}")
            n_skipped += 1
            continue

        if not isinstance(module, nn.Linear) or not hasattr(module, "weight"):
            _log(f"Skip (not nn.Linear): {model_key}")
            n_skipped += 1
            continue

        lora_down = matrices["down"].to(torch.float32)
        lora_up = matrices["up"].to(torch.float32)

        # lora_down: [rank, in_features], lora_up: [out_features, rank]
        if lora_down.ndim != 2 or lora_up.ndim != 2:
            _log(f"Skip (unexpected tensor dims – down={lora_down.shape}, up={lora_up.shape}): {model_key}")
            n_skipped += 1
            continue

        rank = lora_down.shape[0]
        alpha = matrices.get("alpha", rank)

        # delta_W shape: [out_features, in_features]
        delta_W = lora_up @ lora_down * (alpha / rank) * strength

        try:
            delta = delta_W.to(
                dtype=module.weight.data.dtype,
                device=module.weight.data.device,
            )
            module.weight.data.add_(delta)
            _log(f"Merged: {model_key}")
            n_applied += 1
        except Exception as exc:
            _log(f"Skip (error – {exc}): {model_key}")
            n_skipped += 1

    return n_applied, n_skipped


# ---------------------------------------------------------------------------
# Lazy-loader subclass with LoRA merging
# ---------------------------------------------------------------------------

class LoRAMergedModelLoader(LazyModelLoader):
    """Lazy model loader that merges LoRA weights on first use.

    On ``_ensure_loaded``:
    1. The base model is loaded to **CPU** (no offload wrapper yet).
    2. LoRA delta weights are merged into the CPU model in-place.
    3. The appropriate offload wrapper is applied if the target device is CUDA.

    This ordering ensures LoRA is merged into plain float weights **before**
    any quantised-aware offload wrapper is constructed.
    """

    def __init__(
        self,
        base_loader: LazyModelLoader,
        lora_state_dict: dict,
        strength: float,
        lora_key_prefix: str,
        model_key_prefix: str,
        lora_name: str,
        logger=None,
    ) -> None:
        super().__init__(
            model_path=base_loader.model_path,
            model_name=base_loader.model_name,
            load_fn=base_loader.load_fn,
            load_args=base_loader.load_args,
        )
        self._lora_state_dict = lora_state_dict
        self._lora_strength = strength
        self._lora_key_prefix = lora_key_prefix
        self._model_key_prefix = model_key_prefix
        self._lora_name = lora_name
        self._lora_logger = logger

    # Override the private list of "own" attribute names so __getattr__
    # doesn't try to forward them to the (not-yet-loaded) inner model.
    def __getattr__(self, name: str):
        _own = {
            "_lora_state_dict",
            "_lora_strength",
            "_lora_key_prefix",
            "_model_key_prefix",
            "_lora_name",
            "_lora_logger",
        }
        if name in _own:
            return object.__getattribute__(self, name)
        return super().__getattr__(name)

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load base model to CPU, merge LoRA, then wrap for offload."""
        if self._loaded:
            return

        from ..utils.timing import timed_print

        start = time.time()
        timed_print(
            f"⏳ Loading model+LoRA: {self.model_name} + {self._lora_name}...",
            start,
        )

        # Step 1: load the base model to CPU only (target_device=None prevents
        # any offload wrapper being applied inside _load_model_impl).
        self._model = self.load_fn(
            self.model_path,
            self.load_args,
            target_device=None,
        )

        # Step 2: merge LoRA delta weights into the CPU model
        n_applied, n_skipped = _apply_lora_to_module(
            self._model,
            self._lora_state_dict,
            self._lora_strength,
            self._lora_key_prefix,
            self._model_key_prefix,
            logger=self._lora_logger,
        )
        timed_print(
            f"✓ LoRA merged: {n_applied} layer(s) updated, {n_skipped} skipped",
            start,
        )

        # Step 3: apply the same offload wrapper the base loader would use
        if (
            self._target_device is not None
            and str(self._target_device).startswith("cuda")
        ):
            self._model = self._apply_offload_wrapper(
                self._model, self._target_device
            )

        self._loaded = True
        self._load_time = time.time() - start
        timed_print(
            f"✓ Model+LoRA ready in {self._load_time:.2f}s: {self.model_name}",
            start,
        )

    def _apply_offload_wrapper(self, model: nn.Module, target_device):
        """Apply the same offload wrapper that the base loader would apply."""
        # load_args is a (args, logger, lazy_loader) tuple from TurboWanModelLoader
        load_args = self.load_args
        if isinstance(load_args, (tuple, list)) and len(load_args) >= 1:
            args = load_args[0]
        else:
            args = load_args

        offload_mode = getattr(args, "offload_mode", "comfy_native")

        if offload_mode == "cpu_only":
            from ..utils.cpu_offload_wrapper import CPUOffloadWrapper

            return CPUOffloadWrapper(model, target_device)
        elif offload_mode == "comfy_native":
            from ..utils.comfy_native_offload import ComfyNativeOffloadCallable

            return ComfyNativeOffloadCallable(model, load_device=target_device)
        else:
            from ..utils.layerwise_gpu_offload_wrapper import (
                LayerwiseGPUOffloadWrapper,
            )

            return LayerwiseGPUOffloadWrapper(
                model, target_device, empty_cache_every=8  # same value as TurboWanModelLoader
            )


# ---------------------------------------------------------------------------
# ComfyUI node
# ---------------------------------------------------------------------------

class TurboWanLoRALoader:
    """Merge a LoRA checkpoint into an unquantized TurboDiffusion model.

    Two LoRA key formats are supported:

    * **Diffusers / PEFT** — keys end in ``.lora_A.weight`` /
      ``.lora_B.weight``
    * **Kohya** — keys end in ``.lora_down.weight`` / ``.lora_up.weight``

    The merged weight delta for each LoRA layer is::

        delta_W = lora_up @ lora_down * (alpha / rank) * strength

    Only unquantized (plain float ``nn.Linear``) layers are merged.
    Quantized (``Int8Linear``) layers are skipped automatically.

    LoRA weights are merged lazily: the merge is performed the first time
    the model is moved to a device (i.e. at the start of inference), so
    there is no upfront cost in the ComfyUI workflow graph.

    Key-prefix remapping
    --------------------
    ``lora_key_prefix`` is stripped from every LoRA key and replaced with
    ``model_key_prefix`` before the corresponding model layer is looked up.
    Defaults map **Diffusers** ``transformer.*`` keys to TurboDiffusion's
    ``net.*`` namespace.  Change these if your LoRA was trained with a
    different prefix scheme.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_files = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (lora_files,),
                "strength": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": -10.0,
                        "max": 10.0,
                        "step": 0.01,
                        "tooltip": (
                            "LoRA scale multiplier (strength).  "
                            "1.0 = full LoRA effect; 0.0 = no change."
                        ),
                    },
                ),
            },
            "optional": {
                "lora_key_prefix": (
                    "STRING",
                    {
                        "default": "transformer.",
                        "tooltip": (
                            "Prefix used in LoRA keys.  "
                            "Use 'transformer.' for Diffusers/PEFT LoRAs, "
                            "'net.' if the LoRA already uses net. keys, "
                            "or '' if there is no prefix."
                        ),
                    },
                ),
                "model_key_prefix": (
                    "STRING",
                    {
                        "default": "net.",
                        "tooltip": (
                            "Prefix used in TurboDiffusion model keys.  "
                            "Default is 'net.'."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_lora"
    CATEGORY = "loaders"
    DESCRIPTION = (
        "Merge a LoRA checkpoint into an unquantized TurboDiffusion model.  "
        "Supports Diffusers/PEFT (.lora_A / .lora_B) and Kohya "
        "(.lora_down / .lora_up) LoRA formats.  "
        "Accepts .safetensors files (safe, recommended) and legacy .pt/.pth "
        "pickle checkpoints.  "
        "Merging is performed lazily on first inference.  "
        "Use with unquantized model checkpoints only."
    )

    def load_lora(
        self,
        model,
        lora_name: str,
        strength: float = 1.0,
        lora_key_prefix: str = "transformer.",
        model_key_prefix: str = "net.",
    ):
        """Return a lazy model loader that will merge LoRA on first use.

        Args:
            model: MODEL output from ``TurboWanModelLoader`` (a
                ``LazyModelLoader``).
            lora_name: LoRA filename from the ComfyUI ``loras/`` folder.
            strength: Scale multiplier for the LoRA delta (default 1.0).
            lora_key_prefix: Prefix to strip from LoRA keys before lookup.
            model_key_prefix: Prefix to prepend when looking up model layers.

        Returns:
            Tuple containing a ``LoRAMergedModelLoader``.
        """
        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)

        logger = TimedLogger("TurboWanLoRALoader")
        logger.section("Preparing LoRA Loader")
        logger.log(f"LoRA file   : {lora_name}")
        logger.log(f"Base model  : {getattr(model, 'model_name', 'unknown')}")
        logger.log(f"Strength    : {strength}")
        logger.log(f"Key mapping : '{lora_key_prefix}' → '{model_key_prefix}'")

        # Load the LoRA checkpoint eagerly – it is typically small (< 1 GB)
        # and we need to inspect its keys before building the lazy loader.
        # .safetensors files are loaded via safetensors.torch.load_file();
        # legacy pickle checkpoints fall back to torch.load().
        logger.log(f"Loading LoRA checkpoint...")
        lora_state_dict = _load_lora_state_dict(str(lora_path))

        # Summarise what was found
        suffixes = (
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora_down.weight",
            ".lora_up.weight",
        )
        n_lora_tensors = sum(
            1 for k in lora_state_dict if any(k.endswith(s) for s in suffixes)
        )
        logger.log(f"Found {n_lora_tensors} LoRA weight tensors in checkpoint")
        logger.log("✓ LoRA loader created (merge happens on first model use)")
        print("=" * 60)
        print()

        lora_loader = LoRAMergedModelLoader(
            base_loader=model,
            lora_state_dict=lora_state_dict,
            strength=strength,
            lora_key_prefix=lora_key_prefix,
            model_key_prefix=model_key_prefix,
            lora_name=lora_name,
            logger=logger,
        )

        return (lora_loader,)
