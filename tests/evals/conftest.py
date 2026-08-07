"""The persona runs both eval modules read.

``run_persona`` makes a live Claude call per turn when a key is configured, so
the transcript is built once per session and shared — ``test_scenarios.py``'s
deterministic invariants and ``test_judges.py``'s LLM judges assert over the
same call rather than each paying for their own. Session scope rather than
module scope for exactly that reason: module scope would run all eight
personas once per file.
"""

from __future__ import annotations

import pytest

from tests.evals.personas import PERSONAS, Persona
from tests.evals.simulator import Transcript, run_persona


@pytest.fixture(scope="session", params=PERSONAS, ids=[p.key for p in PERSONAS])
def transcript(request: pytest.FixtureRequest) -> Transcript:
    persona: Persona = request.param
    return run_persona(persona)
