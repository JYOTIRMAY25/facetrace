"""Every FaceTrace module must import cleanly on a bare checkout.

python-dotenv is optional, and no STEP 2 dependency (OpenCV, Web3, ...) may be
required at import time.
"""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "app",
    "app.config",
    "app.main",
    "app.models",
    "app.models.pipeline",
    "app.orchestrator",
    "app.services",
    "app.services.blockchain",
    "app.services.face",
    "app.services.fingerprint",
    "app.services.matching",
    "app.services.search",
    "app.services.verification",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports(module_name: str) -> None:
    assert importlib.import_module(module_name) is not None


def test_package_exposes_version_and_disclaimer() -> None:
    import app

    assert isinstance(app.__version__, str) and app.__version__
    # The safety constraint is part of the package contract, not just docs.
    assert "legal identity" in app.CONTENT_MATCH_DISCLAIMER


def test_import_does_not_pull_in_step2_dependencies() -> None:
    """Importing the package must not require OpenCV, Web3, and friends."""
    import sys

    step2 = {"cv2", "web3", "insightface", "onnxruntime", "httpx", "bs4", "numpy"}
    already_loaded = step2 & set(sys.modules)

    for module_name in MODULES:
        sys.modules.pop(module_name, None)
    for module_name in MODULES:
        importlib.import_module(module_name)

    newly_loaded = (step2 & set(sys.modules)) - already_loaded
    assert not newly_loaded, f"import pulled in STEP 2 dependencies: {newly_loaded}"
