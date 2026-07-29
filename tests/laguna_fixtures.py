"""Paths to the small config-only Laguna fixtures used by CPU-only tests."""

from pathlib import Path

_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "laguna_configs"
TARGET_CONFIG = _FIXTURE_ROOT / "target"
DRAFT_CONFIG = _FIXTURE_ROOT / "draft"
