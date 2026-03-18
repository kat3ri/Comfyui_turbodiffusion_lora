"""
pytest conftest.py — stub out ComfyUI and other external dependencies so that
nodes/lora_loader.py can be imported and tested without a full ComfyUI
installation.

Strategy
--------
We register a minimal fake package hierarchy under the directory's actual
package name (``Comfyui_turbodiffusion_lora``) so that lora_loader.py's
relative imports resolve:

    from ..utils.lazy_loader import LazyModelLoader   →  ok
    from ..utils.timing import TimedLogger             →  ok

We do NOT execute the package's real ``__init__.py`` (which would pull in
ComfyUI nodes that depend on dozens of external packages).  Instead, the
package object in sys.modules is a lightweight stub.

This file lives in tests/ (a directory WITHOUT __init__.py) so that Python
can import it without first running the parent package's __init__.py.
"""

import sys
import os
import types
import importlib.util as _ilu

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_TESTS_DIR)   # …/Comfyui_turbodiffusion_lora/

# Add package root to sys.path so that ``import nodes.lora_loader`` can be
# used as a fallback, but the real import will go through the fake package.
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)


def _stub(name: str, **attrs):
    """Create and register a minimal stub module under *name*."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# ---------------------------------------------------------------------------
# Third-party stubs (ComfyUI and other deps that some nodes import)
# ---------------------------------------------------------------------------
_stub("folder_paths",
      get_filename_list=lambda _: [],
      get_full_path_or_raise=lambda *a, **kw: "")
_stub("comfy")
_stub("comfy.sd")
_stub("comfy.model_management")
_stub("comfy.model_patcher")
_stub("comfy.utils")
_stub("node_helpers")
_stub("safetensors")
_stub("safetensors.torch", load_file=lambda path, device="cpu": {})

# ---------------------------------------------------------------------------
# Build a fake «Comfyui_turbodiffusion_lora» package in sys.modules so that
# relative imports inside lora_loader.py resolve without running __init__.py.
# ---------------------------------------------------------------------------
_PKG_NAME = "Comfyui_turbodiffusion_lora"

# Top-level package stub (replaces the real __init__.py)
_pkg = _stub(_PKG_NAME)
_pkg.__path__ = [_PKG_ROOT]
_pkg.__package__ = _PKG_NAME

# utils sub-package stub
_utils = _stub(f"{_PKG_NAME}.utils")
_utils.__path__ = [os.path.join(_PKG_ROOT, "utils")]
_utils.__package__ = f"{_PKG_NAME}.utils"

# utils.lazy_loader stub
class _FakeLazyModelLoader:
    def __init__(self, model_path=None, model_name=None, load_fn=None, load_args=None):
        self.model_path = model_path
        self.model_name = model_name
        self.load_fn = load_fn
        self.load_args = load_args
        self._model = None
        self._loaded = False
        self._load_time = None
        self._target_device = None
        self._target_kwargs = {}

_stub(f"{_PKG_NAME}.utils.lazy_loader", LazyModelLoader=_FakeLazyModelLoader)

# utils.timing stub
class _FakeTimedLogger:
    def __init__(self, *a, **kw): pass
    def section(self, *a, **kw): pass
    def log(self, *a, **kw): pass

def _fake_timed_print(*a, **kw): pass

_stub(f"{_PKG_NAME}.utils.timing",
      TimedLogger=_FakeTimedLogger,
      timed_print=_fake_timed_print)

# nodes sub-package stub
_nodes = _stub(f"{_PKG_NAME}.nodes")
_nodes.__path__ = [os.path.join(_PKG_ROOT, "nodes")]
_nodes.__package__ = f"{_PKG_NAME}.nodes"

# ---------------------------------------------------------------------------
# Now load lora_loader.py as part of the fake package.
# ---------------------------------------------------------------------------
_ll_path = os.path.join(_PKG_ROOT, "nodes", "lora_loader.py")
_ll_spec = _ilu.spec_from_file_location(
    f"{_PKG_NAME}.nodes.lora_loader",
    _ll_path,
    submodule_search_locations=[],
)
_ll_mod = _ilu.module_from_spec(_ll_spec)
_ll_mod.__package__ = f"{_PKG_NAME}.nodes"
sys.modules[f"{_PKG_NAME}.nodes.lora_loader"] = _ll_mod
_ll_spec.loader.exec_module(_ll_mod)

# Make it importable also as ``nodes.lora_loader`` (the path used in tests).
sys.modules["nodes.lora_loader"] = _ll_mod
