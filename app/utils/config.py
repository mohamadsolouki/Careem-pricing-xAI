from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SECRETS_PATH = PROJECT_ROOT / ".streamlit" / "secrets.toml"


@lru_cache(maxsize=1)
def _load_streamlit_secrets() -> dict[str, str]:
    if not SECRETS_PATH.exists():
        return {}

    with open(SECRETS_PATH, "rb") as secrets_file:
        payload = tomllib.load(secrets_file)

    return {
        key: str(value)
        for key, value in payload.items()
        if not isinstance(value, dict) and value is not None
    }


def get_config_value(name: str) -> str | None:
    env_value = os.getenv(name)
    if env_value:
        return env_value
    return _load_streamlit_secrets().get(name)