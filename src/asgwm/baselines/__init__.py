"""Pluggable baseline registry. Import side-effect registers all adapters."""
from .base import (  # noqa: F401
    Baseline, register, get, all_names, available_names, display_name, family,
)
from . import adapters  # noqa: F401  (registers pysteps + the four NN stubs)

# Canonical headline comparison order (matches the manuscript skill table).
HEADLINE = ["pysteps", "rainnet", "nowcastnet", "langprecip", "thor"]
