"""Run the standalone V10 assistant with the validated product defaults.

Two launch profiles share this code:

* ``normal``   Gemma 4 12B, thinking off, 95 s answers (port 8782)
* ``thinking`` Gemma 4 12B, thinking on, every budget thinking consumes raised
               (port 8783)

The profile comes from SISU_READER_VARIANT if set, otherwise from the
``VARIANT`` file next to this script. Any SISU_READER_* variable you set
yourself still wins over these defaults. The same values are defined as the
``normal`` and ``thinking`` variants in sisu_eval/variants.json; a test keeps
them identical so the copy you run is the configuration you evaluated.
"""
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent

PROFILES = {
    "normal": {
        "SISU_READER_MODEL": "gemma4:12b",
        "SISU_READER_REASONING": "off",
        "SISU_READER_CONTEXT_TOKENS": "32768",
        "SISU_READER_DEADLINE_S": "95",
        "SISU_READER_REQUEST_TIMEOUT_S": "88",
        "SISU_READER_MAX_MODEL_CALLS": "30",
        "SISU_READER_MAX_OUTPUT_TOKENS": "32768",
        "SISU_READER_LOCK_TIMEOUT_S": "120",
        "SISU_READER_WEB_PORT": "8782",
    },
    "thinking": {
        "SISU_READER_MODEL": "gemma4:12b",
        "SISU_READER_REASONING": "high",
        "SISU_READER_THINKING_EXTRA_TOKENS": "3072",
        "SISU_READER_THINKING_TIME_SCALE": "4.0",
        "SISU_READER_CONTEXT_TOKENS": "32768",
        "SISU_READER_DEADLINE_S": "420",
        "SISU_READER_REQUEST_TIMEOUT_S": "240",
        "SISU_READER_MAX_MODEL_CALLS": "30",
        "SISU_READER_MAX_OUTPUT_TOKENS": "65536",
        "SISU_READER_LOCK_TIMEOUT_S": "600",
        "SISU_READER_WEB_PORT": "8783",
    },
}


def selected_profile() -> str:
    name = os.environ.get("SISU_READER_VARIANT", "").strip()
    if not name:
        marker = ROOT / "VARIANT"
        name = marker.read_text(encoding="utf-8").strip() if marker.is_file() else "normal"
    if name not in PROFILES:
        raise SystemExit(f"error: unknown variant {name!r}; choose one of {', '.join(PROFILES)}")
    return name


def configure_environment() -> str:
    profile = selected_profile()
    defaults = {
        "SISU_READER_WORKSPACE": str(ROOT / "workspace"),
        "SISU_READER_TRACE_MODE": "answer",
        **PROFILES[profile],
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)
    # V9 controller actions cannot be applied to V10's different pipeline.
    os.environ["SISU_READER_STRATEGY_LEARNING"] = "false"
    return profile


def main(argv: list[str] | None = None) -> int:
    configure_environment()
    from assistant_runtime.__main__ import main as assistant_main
    return assistant_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
