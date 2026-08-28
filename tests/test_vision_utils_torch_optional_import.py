# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""vision_utils.py (and the hf_utils/dataset_utils it imports) must import
without torch, so unsloth_zoo/mlx/utils.py's `_extract_vlm_images` -- which
imports vision_utils.process_vision_info lazily on the torch-free MLX path --
doesn't crash a Mac's VLM training on the first batch that needs the
extract_vision_info/fetch_image fallback.

This repo's real Apple Silicon test lane runs against real MLX with no torch
installed (matching an actual Mac install), but most of the "MLX" suite here
runs its assertions against mlx_simulation, which fakes MLX's API *on top of*
torch -- so torch is always importable in that environment by construction,
and can never observe an unconditional `import torch` breaking the real
torch-free path. This test forces that condition directly (regardless of
whether the machine running it has torch installed) so it's caught in any CI
environment, not only on real Apple Silicon.
"""

from __future__ import annotations

import builtins
import sys

import pytest

# Every module reachable from vision_utils's own import chain that used to
# import torch unconditionally. Cleared from sys.modules before each test so
# the re-import below is genuine, not served from a previously cached module
# object (which could hide a regression if some other test imported it
# first, with torch already blocked or not).
_TORCH_TOUCHING_MODULES = (
    "unsloth_zoo.vision_utils",
    "unsloth_zoo.hf_utils",
    "unsloth_zoo.dataset_utils",
)


def _import_with_torch_blocked(monkeypatch, module_name: str):
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    for name in _TORCH_TOUCHING_MODULES:
        sys.modules.pop(name, None)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    try:
        import importlib

        return importlib.import_module(module_name)
    finally:
        monkeypatch.undo()
        # Leave a clean slate: the next test (or this process's own later
        # imports) must get the real module, not this torch-free one.
        for name in _TORCH_TOUCHING_MODULES:
            sys.modules.pop(name, None)


def test_vision_utils_imports_without_torch(monkeypatch):
    vision_utils = _import_with_torch_blocked(monkeypatch, "unsloth_zoo.vision_utils")
    assert vision_utils.HAS_TORCH is False
    assert vision_utils.torch is None


def test_process_vision_info_extracts_a_plain_image_without_torch(monkeypatch):
    vision_utils = _import_with_torch_blocked(monkeypatch, "unsloth_zoo.vision_utils")
    from PIL import Image

    image = Image.new("RGB", (32, 32))
    messages = [
        {
            "role": "user",
            "content": [{"type": "image", "image": image}],
        }
    ]
    images, videos = vision_utils.process_vision_info(messages)
    assert images is not None and len(images) == 1
    assert videos is None


def test_hf_utils_imports_without_torch(monkeypatch):
    hf_utils = _import_with_torch_blocked(monkeypatch, "unsloth_zoo.hf_utils")
    assert hf_utils.torch is None


def test_dataset_utils_imports_without_torch(monkeypatch):
    dataset_utils = _import_with_torch_blocked(monkeypatch, "unsloth_zoo.dataset_utils")
    assert dataset_utils.torch is None


def test_defining_the_vision_data_collator_does_not_require_torch(monkeypatch):
    # UnslothVisionDataCollator is itself CUDA-only and cannot be *used*
    # without torch, but merely defining the class (which happens at module
    # import time) must not require it -- that was the failure mode of the
    # unguarded `@torch.no_grad()` class-body decorator.
    vision_utils = _import_with_torch_blocked(monkeypatch, "unsloth_zoo.vision_utils")
    assert vision_utils.UnslothVisionDataCollator is not None
