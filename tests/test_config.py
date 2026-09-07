"""Configuration loading.

``load_settings`` is always driven by an explicit mapping here, so these tests
never depend on the developer's real environment or a local ``.env``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    PROJECT_ROOT,
    REQUIRED_FOR_BLOCKCHAIN,
    REQUIRED_FOR_SEARCH,
    Settings,
    load_dotenv_if_available,
    load_settings,
)


def test_loads_with_empty_environment() -> None:
    """A clean checkout with no .env must still produce usable defaults."""
    settings = load_settings(env={})

    assert isinstance(settings, Settings)
    assert settings.fingerprint_algorithm == "SHA-256"
    assert settings.canonicalization_version == "1"
    assert settings.blockchain_network == "local"
    assert settings.face_match_threshold == pytest.approx(0.65)
    assert settings.search_max_results == 10
    assert settings.max_retries == 2
    assert settings.data_dir == PROJECT_ROOT / "data"


def test_reads_values_from_environment() -> None:
    settings = load_settings(
        env={
            "SEARCH_PROVIDER": "example-provider",
            "SEARCH_API_KEY": "  key-123  ",
            "SEARCH_MAX_RESULTS": "25",
            "FACE_MATCH_THRESHOLD": "0.8",
            "BLOCKCHAIN_NETWORK": "sepolia",
            "BLOCKCHAIN_RPC_URL": "http://127.0.0.1:8545",
            "CONTRACT_ADDRESS": "0xabc",
            "DATA_DIR": "custom-data",
        }
    )

    assert settings.search_provider == "example-provider"
    assert settings.search_api_key == "key-123"  # surrounding whitespace stripped
    assert settings.search_max_results == 25
    assert settings.face_match_threshold == pytest.approx(0.8)
    assert settings.blockchain_network == "sepolia"
    assert settings.data_dir == Path("custom-data")


def test_reads_face_stage_settings_from_environment() -> None:
    """The face stage is driven entirely by config, with no hardcoded model."""
    settings = load_settings(
        env={
            "FACE_MODEL_PACK": "antelopev2",
            "FACE_DET_SIZE": "320",
            "FACE_MIN_DET_SCORE": "0.7",
            "FACE_MODEL_ROOT": "/opt/face-models",
            "FACE_DETECTOR_MODEL": "custom-detector",
            "FACE_ENCODER_MODEL": "custom-encoder",
        }
    )

    assert settings.face_model_pack == "antelopev2"
    assert settings.face_det_size == 320
    assert settings.face_min_det_score == pytest.approx(0.7)
    assert settings.face_model_root == "/opt/face-models"
    assert settings.face_detector_model == "custom-detector"
    assert settings.face_encoder_model == "custom-encoder"


def test_face_stage_defaults_to_a_local_model_pack() -> None:
    settings = load_settings(env={})

    assert settings.face_model_pack == "buffalo_l"
    assert settings.face_det_size == 640
    assert settings.face_min_det_score == pytest.approx(0.5)
    assert settings.face_model_root == ""
    assert settings.redacted()["face_model_root"] == "(default cache)"


def test_reads_search_stage_settings_from_environment() -> None:
    """Everything the search stage needs comes from the environment."""
    settings = load_settings(
        env={
            "SEARCH_PROVIDER": "google-custom-search",
            "SEARCH_ENGINE_ID": "cx-123",
            "SEARCH_TERMS": " alpha , beta ",
            "SEARCH_IMAGE_URL": "https://example.test/photo.jpg",
            "SEARCH_USER_AGENT": "FaceTrace/test (contact@example.test)",
        }
    )

    assert settings.search_provider == "google-custom-search"
    assert settings.search_engine_id == "cx-123"
    assert settings.search_terms == "alpha , beta"
    assert settings.search_image_url == "https://example.test/photo.jpg"
    assert settings.search_user_agent == "FaceTrace/test (contact@example.test)"


def test_search_defaults_to_the_keyless_provider() -> None:
    """A clean checkout can run a genuine search without any credential."""
    settings = load_settings(env={})

    assert settings.search_provider == "wikimedia-commons"
    assert settings.search_api_key == ""
    assert settings.search_terms == ""
    assert settings.search_image_url == ""
    assert settings.search_user_agent.startswith("FaceTrace/")


def test_invalid_numeric_value_is_rejected() -> None:
    with pytest.raises(ValueError, match="SEARCH_MAX_RESULTS"):
        load_settings(env={"SEARCH_MAX_RESULTS": "ten"})

    with pytest.raises(ValueError, match="FACE_MATCH_THRESHOLD"):
        load_settings(env={"FACE_MATCH_THRESHOLD": "high"})


def test_missing_live_requirements_are_reported() -> None:
    settings = load_settings(env={})

    assert settings.missing_for_search(env={}) == REQUIRED_FOR_SEARCH
    assert settings.missing_for_blockchain(env={}) == REQUIRED_FOR_BLOCKCHAIN


def test_no_missing_requirements_when_fully_configured() -> None:
    env = {name: "value" for name in REQUIRED_FOR_SEARCH + REQUIRED_FOR_BLOCKCHAIN}
    settings = load_settings(env=env)

    assert settings.missing_for_search(env=env) == ()
    assert settings.missing_for_blockchain(env=env) == ()


def test_secrets_are_not_exposed_by_repr_or_redacted() -> None:
    settings = load_settings(
        env={"SEARCH_API_KEY": "super-secret", "BLOCKCHAIN_PRIVATE_KEY": "0xdeadbeef"}
    )

    rendered = repr(settings)
    assert "super-secret" not in rendered
    assert "0xdeadbeef" not in rendered

    redacted = settings.redacted()
    assert redacted["search_api_key_present"] is True
    assert redacted["blockchain_private_key_present"] is True
    assert "super-secret" not in str(redacted)
    assert "0xdeadbeef" not in str(redacted)


def test_settings_are_frozen_but_overridable_for_tests() -> None:
    settings = load_settings(env={})

    with pytest.raises(Exception):
        settings.face_match_threshold = 0.9  # type: ignore[misc]

    tuned = settings.with_overrides(face_match_threshold=0.9)
    assert tuned.face_match_threshold == pytest.approx(0.9)
    assert settings.face_match_threshold == pytest.approx(0.65)


def test_dotenv_loading_is_optional_and_safe(tmp_path: Path) -> None:
    """Missing python-dotenv or a missing file must both be non-fatal."""
    assert load_dotenv_if_available(tmp_path / "does-not-exist.env") is False


def test_env_example_documents_every_consumed_variable() -> None:
    """Keep .env.example in sync with what config.py actually reads."""
    template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    consumed = (
        "SEARCH_PROVIDER",
        "SEARCH_API_KEY",
        "SEARCH_ENGINE_ID",
        "SEARCH_MAX_RESULTS",
        "SEARCH_TERMS",
        "SEARCH_IMAGE_URL",
        "SEARCH_USER_AGENT",
        "FACE_DETECTOR_MODEL",
        "FACE_ENCODER_MODEL",
        "FACE_MODEL_PACK",
        "FACE_DET_SIZE",
        "FACE_MIN_DET_SCORE",
        "FACE_MODEL_ROOT",
        "FACE_MATCH_THRESHOLD",
        "CANONICALIZATION_VERSION",
        "FINGERPRINT_ALGORITHM",
        "BLOCKCHAIN_NETWORK",
        "BLOCKCHAIN_RPC_URL",
        "BLOCKCHAIN_PRIVATE_KEY",
        "CONTRACT_ADDRESS",
        "BLOCKCHAIN_TX_TIMEOUT_S",
        "DATA_DIR",
        "REQUEST_TIMEOUT_S",
        "MAX_RETRIES",
    )
    missing = [name for name in consumed if f"{name}=" not in template]
    assert not missing, f".env.example is missing: {missing}"


def test_env_example_contains_no_real_values() -> None:
    """Every key in the template must be blank — no committed secrets."""
    template = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
    populated = [
        line
        for line in template.splitlines()
        if "=" in line and not line.lstrip().startswith("#") and line.split("=", 1)[1]
    ]
    assert not populated, f"values committed in .env.example: {populated}"
