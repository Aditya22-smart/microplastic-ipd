"""Tests for the ``MP_SPECTRAL_ENABLED`` feature flag.

The dashboard renders a different set of tabs depending on the flag, so these
tests pin the environment variable explicitly rather than relying on the
default. That keeps the suite meaningful whichever default we settle on.
"""

from __future__ import annotations

import pytest

from src.config import (
    SPECTRAL_ENABLED_ENV,
    FeatureDisabledError,
    require_spectral,
    set_spectral,
    spectral_enabled,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts from an explicit, known state."""
    monkeypatch.delenv(SPECTRAL_ENABLED_ENV, raising=False)


@pytest.mark.parametrize("raw", ["0", "false", "FALSE", "no", "No", "off", "OFF", ""])
def test_falsey_values_disable(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, raw)
    assert spectral_enabled() is False


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "on", "anything"])
def test_other_values_enable(raw: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, raw)
    assert spectral_enabled() is True


def test_unset_uses_default_false(monkeypatch: pytest.MonkeyPatch) -> None:
    assert spectral_enabled() is False
    assert spectral_enabled(default=True) is True


def test_whitespace_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, "  1  ")
    assert spectral_enabled() is True


def test_set_spectral_round_trips(monkeypatch: pytest.MonkeyPatch) -> None:
    set_spectral(True)
    assert spectral_enabled() is True
    set_spectral(False)
    assert spectral_enabled() is False


def test_require_spectral_raises_when_disabled() -> None:
    with pytest.raises(FeatureDisabledError, match="MP_SPECTRAL_ENABLED"):
        require_spectral("Spectral polymer classification")


def test_require_spectral_passes_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, "1")
    require_spectral("Spectral polymer classification")  # must not raise


def test_analyze_spectrum_refuses_when_disabled() -> None:
    """The guard must fire before any checkpoint is touched."""
    import numpy as np

    from src.models.full_pipeline import analyze_spectrum

    with pytest.raises(FeatureDisabledError):
        analyze_spectrum(np.zeros(600, dtype=np.float32))


def test_analyze_spectrum_passes_the_flag_on_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the flag on, the guard defers to the real model (which then fails
    only because there is no trained checkpoint)."""
    import numpy as np

    from src.config import FeatureDisabledError as FDE
    from src.models.full_pipeline import analyze_spectrum

    monkeypatch.setenv(SPECTRAL_ENABLED_ENV, "1")
    try:
        analyze_spectrum(np.zeros(600, dtype=np.float32))
    except FDE:
        pytest.fail("flag guard fired even though the branch was enabled")
    except FileNotFoundError:
        pass  # expected: no trained spectral checkpoint in the test env
