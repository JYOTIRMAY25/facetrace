"""Configuration loading for FaceTrace.

Secrets are read from the process environment only. A local ``.env`` file is
loaded when ``python-dotenv`` is installed; it is optional, so importing this
module never fails on a machine without the dependency and never requires a
populated ``.env`` to run ``--health``.

Never commit ``.env``, API keys, or private keys. See ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Mapping

__all__ = [
    "Settings",
    "load_settings",
    "load_dotenv_if_available",
    "REQUIRED_FOR_SEARCH",
    "REQUIRED_FOR_BLOCKCHAIN",
    "PROJECT_ROOT",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: Environment variables a *live* genuine search needs (STEP 2).
REQUIRED_FOR_SEARCH: tuple[str, ...] = ("SEARCH_PROVIDER", "SERPAPI_API_KEY")

#: Environment variables a *live* blockchain write needs (STEP 2).
REQUIRED_FOR_BLOCKCHAIN: tuple[str, ...] = (
    "BLOCKCHAIN_RPC_URL",
    "BLOCKCHAIN_PRIVATE_KEY",
    "CONTRACT_ADDRESS",
)


def load_dotenv_if_available(dotenv_path: Path | None = None) -> bool:
    """Load ``.env`` into ``os.environ`` if ``python-dotenv`` is installed.

    Returns ``True`` when a file was loaded, ``False`` otherwise. Absence of
    the library or the file is not an error.
    """
    try:
        from dotenv import load_dotenv  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return False

    path = dotenv_path or (PROJECT_ROOT / ".env")
    if not path.is_file():
        return False
    return bool(load_dotenv(path, override=False))


def _get(env: Mapping[str, str], key: str, default: str = "") -> str:
    return (env.get(key) or default).strip()


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _get(env, key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{key} must be a number, got {raw!r}") from exc


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _get(env, key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{key} must be an integer, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Resolved runtime configuration.

    Every field has a safe default so the package is importable and
    ``--health`` runs on a clean checkout. Secret-bearing fields default to
    empty strings; use :meth:`missing_for_search` / :meth:`missing_for_blockchain`
    to report what a live run would still need.
    """

    # --- search (STEP 3) ---
    #: Identifier of the provider adapter to use. See
    #: ``app.services.search_web.PROVIDERS`` for the registered names.
    search_provider: str = "wikimedia-commons"
    search_api_key: str = field(default="", repr=False)
    search_engine_id: str = ""
    search_max_results: int = 10
    #: Operator-supplied query terms. The search stage refuses to invent terms,
    #: so a text provider needs these (or ``--terms``) to run.
    search_terms: str = ""
    #: Public image URL for a reverse-image-search provider. FaceTrace never
    #: uploads the local image; the operator supplies a URL they control.
    search_image_url: str = ""
    #: Sent on every search request. Wikimedia's User-Agent policy requires a
    #: descriptive agent with a contact, so make this identifiable before
    #: running any volume of live queries.
    search_user_agent: str = "FaceTrace/0.1 (HH Goa 2026 Task 3 prototype)"

    # --- matching thresholds (STEP 2, to be tuned experimentally) ---
    face_match_threshold: float = 0.65
    face_detector_model: str = "insightface-det_10g"
    face_encoder_model: str = "insightface-w600k_r50"

    # --- face stage (STEP 2) ---
    #: InsightFace model pack. All inference is local; the pack is downloaded
    #: once to ``face_model_root`` (or ~/.insightface) on first use.
    face_model_pack: str = "buffalo_l"
    #: Square detector input size in pixels.
    face_det_size: int = 640
    #: Minimum detector score for a box to count as a face.
    face_min_det_score: float = 0.5
    #: Optional pre-provisioned model directory, for fully offline runs.
    face_model_root: str = ""

    # --- fingerprint ---
    canonicalization_version: str = "1"
    fingerprint_algorithm: str = "SHA-256"

    # --- blockchain (STEP 2) ---
    blockchain_network: str = "local"
    blockchain_rpc_url: str = ""
    blockchain_private_key: str = field(default="", repr=False)
    contract_address: str = ""
    blockchain_chain_id: int = 0
    blockchain_explorer_url: str = ""
    blockchain_tx_timeout_s: int = 120

    # --- runtime ---
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    request_timeout_s: int = 30
    max_retries: int = 2
    candidate_image_max_bytes: int = 10 * 1024 * 1024

    def missing_for_search(self, env: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """Environment variables still needed for a live genuine search."""
        source = env if env is not None else os.environ
        return tuple(name for name in REQUIRED_FOR_SEARCH if not _get(source, name))

    def missing_for_blockchain(
        self, env: Mapping[str, str] | None = None
    ) -> tuple[str, ...]:
        """Environment variables still needed for a live blockchain write."""
        source = env if env is not None else os.environ
        return tuple(name for name in REQUIRED_FOR_BLOCKCHAIN if not _get(source, name))

    def redacted(self) -> dict[str, object]:
        """Config snapshot safe to print or log — secrets become booleans."""
        return {
            "search_provider": self.search_provider or "(unset)",
            "search_api_key_present": bool(self.search_api_key),
            "search_engine_id": self.search_engine_id or "(unset)",
            "search_max_results": self.search_max_results,
            "search_terms": self.search_terms or "(unset)",
            "search_image_url": self.search_image_url or "(unset)",
            "search_user_agent": self.search_user_agent,
            "face_match_threshold": self.face_match_threshold,
            "face_detector_model": self.face_detector_model,
            "face_encoder_model": self.face_encoder_model,
            "face_model_pack": self.face_model_pack,
            "face_det_size": self.face_det_size,
            "face_min_det_score": self.face_min_det_score,
            "face_model_root": self.face_model_root or "(default cache)",
            "canonicalization_version": self.canonicalization_version,
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "blockchain_network": self.blockchain_network,
            "blockchain_rpc_url": self.blockchain_rpc_url or "(unset)",
            "blockchain_private_key_present": bool(self.blockchain_private_key),
            "contract_address": self.contract_address or "(unset)",
            "blockchain_chain_id": self.blockchain_chain_id or "(unset)",
            "blockchain_explorer_url": self.blockchain_explorer_url or "(unset)",
            "blockchain_tx_timeout_s": self.blockchain_tx_timeout_s,
            "data_dir": str(self.data_dir),
            "request_timeout_s": self.request_timeout_s,
            "max_retries": self.max_retries,
        }

    def with_overrides(self, **overrides: object) -> "Settings":
        """Return a copy with the given fields replaced (test helper)."""
        return replace(self, **overrides)  # type: ignore[arg-type]


def load_settings(
    env: Mapping[str, str] | None = None, *, use_dotenv: bool = True
) -> Settings:
    """Build :class:`Settings` from a mapping (defaults to ``os.environ``).

    Passing an explicit ``env`` keeps tests hermetic and skips ``.env`` loading.
    """
    if env is None:
        if use_dotenv:
            load_dotenv_if_available()
        env = os.environ

    data_dir_raw = _get(env, "DATA_DIR")
    return Settings(
        search_provider=_get(env, "SEARCH_PROVIDER", "wikimedia-commons"),
        search_api_key=_get(env, "SERPAPI_API_KEY") or _get(env, "SEARCH_API_KEY"),
        search_engine_id=_get(env, "SEARCH_ENGINE_ID"),
        search_max_results=_get_int(env, "SEARCH_MAX_RESULTS", 10),
        search_terms=_get(env, "SEARCH_TERMS"),
        search_image_url=_get(env, "SEARCH_IMAGE_URL"),
        search_user_agent=_get(
            env, "SEARCH_USER_AGENT", "FaceTrace/0.1 (HH Goa 2026 Task 3 prototype)"
        ),
        face_match_threshold=_get_float(env, "FACE_MATCH_THRESHOLD", 0.65),
        face_detector_model=_get(env, "FACE_DETECTOR_MODEL", "insightface-det_10g"),
        face_encoder_model=_get(env, "FACE_ENCODER_MODEL", "insightface-w600k_r50"),
        face_model_pack=_get(env, "FACE_MODEL_PACK", "buffalo_l"),
        face_det_size=_get_int(env, "FACE_DET_SIZE", 640),
        face_min_det_score=_get_float(env, "FACE_MIN_DET_SCORE", 0.5),
        face_model_root=_get(env, "FACE_MODEL_ROOT"),
        canonicalization_version=_get(env, "CANONICALIZATION_VERSION", "1"),
        fingerprint_algorithm=_get(env, "FINGERPRINT_ALGORITHM", "SHA-256"),
        blockchain_network=_get(env, "BLOCKCHAIN_NETWORK", "local"),
        blockchain_rpc_url=_get(env, "BLOCKCHAIN_RPC_URL"),
        blockchain_private_key=_get(env, "BLOCKCHAIN_PRIVATE_KEY"),
        contract_address=(
            _get(env, "BLOCKCHAIN_CONTRACT_ADDRESS")
            or _get(env, "CONTRACT_ADDRESS")
        ),
        blockchain_chain_id=_get_int(env, "BLOCKCHAIN_CHAIN_ID", 0),
        blockchain_explorer_url=_get(env, "BLOCKCHAIN_EXPLORER_URL"),
        blockchain_tx_timeout_s=_get_int(env, "BLOCKCHAIN_TX_TIMEOUT_S", 120),
        data_dir=Path(data_dir_raw) if data_dir_raw else PROJECT_ROOT / "data",
        request_timeout_s=_get_int(env, "REQUEST_TIMEOUT_S", 30),
        max_retries=_get_int(env, "MAX_RETRIES", 2),
        candidate_image_max_bytes=_get_int(
            env, "CANDIDATE_IMAGE_MAX_BYTES", 10 * 1024 * 1024
        ),
    )
