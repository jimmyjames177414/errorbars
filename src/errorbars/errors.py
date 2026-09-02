"""Exception types. All user-facing failures inherit :class:`ErrorbarsError`.

The CLI catches that base class and prints the message without a traceback, so every
message raised from it has to read as something a user can act on, not as an internal
assertion.
"""

from __future__ import annotations

__all__ = [
    "BudgetExceededError",
    "ErrorbarsError",
    "IngestError",
    "ProviderError",
    "SpecError",
]


class ErrorbarsError(Exception):
    """Base class for every error errorbars raises deliberately."""


class SpecError(ErrorbarsError):
    """An experiment.yaml is missing something, or says something contradictory."""


class IngestError(ErrorbarsError):
    """A results directory could not be read as CXS."""


class ProviderError(ErrorbarsError):
    """A model backend failed, refused, or returned something unusable."""


class BudgetExceededError(ErrorbarsError):
    """The run hit its ``--max-calls`` ceiling and stopped rather than spending more."""
