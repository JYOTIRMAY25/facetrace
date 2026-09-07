"""Orchestrator wiring and the STEP 1 CLI entry point."""

from __future__ import annotations

import json

import pytest

from app import CONTENT_MATCH_DISCLAIMER, __version__
from app.config import load_settings
from app.main import build_parser, health_report, main
from app.models.pipeline import (
    PIPELINE_STAGES,
    ErrorCode,
    StageStatus,
    VerificationStatus,
)
from app.orchestrator import Orchestrator, build_default_orchestrator


def test_default_orchestrator_wires_every_capability() -> None:
    orchestrator = build_default_orchestrator(load_settings(env={}))
    service_map = orchestrator.service_map()

    assert set(service_map) == {
        "FaceIdentifier",
        "SearchProvider",
        "MatchSelector",
        "FingerprintService",
        "BlockchainService",
        "VerificationService",
    }
    assert all(service_map.values()), "every capability must report an implementation"
    # Every placeholder names itself, so --health output is uniform.
    assert all(name.startswith("pending-") for name in service_map.values())


def test_orchestrator_stage_plan_is_the_documented_pipeline() -> None:
    orchestrator = build_default_orchestrator(load_settings(env={}))
    plan = orchestrator.describe()

    assert [item.stage for item in plan] == list(PIPELINE_STAGES)
    assert all(item.status is StageStatus.PENDING for item in plan)


def test_services_are_injectable() -> None:
    class StubFace:
        name = "stub-face"

        def load(self, image_path: str):  # pragma: no cover - not called
            raise AssertionError

        def detect(self, face_input):  # pragma: no cover - not called
            raise AssertionError

        def encode(self, face_input, face):  # pragma: no cover - not called
            raise AssertionError

    orchestrator = Orchestrator(settings=load_settings(env={}), face=StubFace())
    assert orchestrator.service_map()["FaceIdentifier"] == "stub-face"


def test_run_reports_not_implemented_without_inventing_a_result() -> None:
    orchestrator = build_default_orchestrator(load_settings(env={}))
    result = orchestrator.run("data/input.jpg")

    assert result.image_path == "data/input.jpg"
    assert result.status is VerificationStatus.INCONCLUSIVE
    assert result.ok is False

    assert result.error is not None
    assert result.error.code is ErrorCode.NOT_IMPLEMENTED
    assert result.error.retryable is False

    assert [item.stage for item in result.stages] == list(PIPELINE_STAGES)
    assert all(item.status is StageStatus.NOT_IMPLEMENTED for item in result.stages)

    # Nothing downstream may be populated with fabricated data.
    assert result.match is None
    assert result.fingerprint is None
    assert result.record is None
    assert result.verification is None


def test_health_report_is_structured_and_secret_free() -> None:
    report = health_report(build_default_orchestrator(load_settings(env={})))

    assert report["version"] == __version__
    assert report["step"] == 1
    assert report["pipeline_stages"] == [stage.value for stage in PIPELINE_STAGES]
    assert report["disclaimer"] == CONTENT_MATCH_DISCLAIMER

    serialised = json.dumps(report)
    assert "search_api_key_present" in serialised
    assert '"search_api_key"' not in serialised
    assert '"blockchain_private_key"' not in serialised


def test_cli_parser_accepts_the_step_1_flags() -> None:
    parser = build_parser()

    assert parser.parse_args(["--health"]).health is True
    assert parser.parse_args(["--health", "--json"]).json is True
    assert parser.parse_args(["--image", "data/input.jpg"]).image == "data/input.jpg"


def test_cli_health_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--health"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "FACETRACE" in captured.out
    assert "legal identity" in captured.out


def test_cli_health_json_is_parseable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["--health", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["step"] == 1
    assert len(payload["pipeline_stages"]) == len(PIPELINE_STAGES)


def test_cli_without_arguments_prints_help_and_fails(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main([])
    assert exit_code == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_image_run_reports_not_implemented(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(["--image", "data/input.jpg"])
    captured = capsys.readouterr()

    assert exit_code == 3, "STEP 1 must not report success for a pipeline run"
    assert "NOT_IMPLEMENTED" in captured.out
    assert "VERIFIED" not in captured.out
