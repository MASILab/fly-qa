"""Environment/config loading. Never logs or prints the neuprint token."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

_loaded = False


def _ensure_loaded() -> None:
    global _loaded
    if not _loaded:
        load_dotenv(find_dotenv(usecwd=True))
        _loaded = True


def get_neuprint_token() -> str | None:
    """Return the neuprint API token, or None if not configured."""
    _ensure_loaded()
    token = os.environ.get("NEUPRINT_API_TOKEN")
    return token or None


NEUPRINT_SERVER = "neuprint-cns.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"

CACHE_DIR = Path(os.environ.get("FLY_QA_CACHE_DIR", Path.home() / ".cache" / "fly_qa"))
