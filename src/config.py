"""Runtime feature flags.

The project runs two independent branches that meet only at the dashboard:

* **visual**  — Stage 1A detection + Stage 1B morphology  (image in)
* **spectral** — Stage 2A preprocessing + Stage 2B polymer  (spectrum in)

``MP_SPECTRAL_ENABLED`` gates the spectral branch. When it is off, the spectral
stage is removed from the dashboard entirely and :func:`analyze_spectrum`
refuses to run. This keeps a partially-finished spectral branch (no trained
checkpoint yet) from breaking an otherwise complete image-only demo.

The default is **off**: set ``MP_SPECTRAL_ENABLED=1`` to enable it.

Only :mod:`os` is used here — no Streamlit import — so this module stays
unit-testable and usable from notebooks, Colab and CI alike.
"""

from __future__ import annotations

import os

#: Environment variable that controls the Stage 2A/2B spectral branch.
SPECTRAL_ENABLED_ENV = "MP_SPECTRAL_ENABLED"

#: Values (case-insensitive) that switch a flag OFF. Anything else switches it
#: ON, so ``MP_SPECTRAL_ENABLED=1``, ``=true`` and ``=yes`` all enable.
_FALSEY = frozenset({"0", "false", "no", "off", ""})


class FeatureDisabledError(RuntimeError):
    """Raised when a disabled branch of the pipeline is invoked."""


def spectral_enabled(default: bool = False) -> bool:
    """Return whether the Stage 2A/2B spectral branch is enabled.

    Args:
        default: value used when the environment variable is not set at all.
    """
    raw = os.environ.get(SPECTRAL_ENABLED_ENV)
    if raw is None:
        return default
    return raw.strip().lower() not in _FALSEY


def require_spectral(what: str) -> None:
    """Raise :class:`FeatureDisabledError` when the spectral branch is off.

    Args:
        what: short description of what was attempted, used in the message.
    """
    if not spectral_enabled():
        raise FeatureDisabledError(
            f"{what} needs the spectral branch (Stage 2A/2B), which is disabled. "
            f"Set {SPECTRAL_ENABLED_ENV}=1 to enable it."
        )


def set_spectral(enabled: bool) -> None:
    """Set the spectral flag for the current process.

    Used by the dashboard sidebar so an interactive toggle is visible to the
    rest of the pipeline, which reads the environment variable.
    """
    os.environ[SPECTRAL_ENABLED_ENV] = "1" if enabled else "0"
