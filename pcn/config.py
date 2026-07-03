"""
Configuration system for PCN.

Loads defaults from default_config.json and provides utilities for
merging custom config files and keyword overrides.
"""

import json
from pathlib import Path

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.json"

# Sentinel: used as default in method signatures to mean "use DEFAULTS"
_DEFAULT = object()


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base, one level deep per section."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = {**result[key], **val}
        else:
            result[key] = val
    return result


def load_config(config_file=None) -> dict:
    """Load DEFAULTS, optionally overlaid with a custom JSON file.

    Args:
        config_file: Optional path to a JSON file whose values override
            the built-in defaults.  The file may contain ``"model"``,
            ``"train"``, and/or ``"test"`` sections.

    Returns:
        Merged configuration dict with ``"model"``, ``"train"``, and
        ``"test"`` sections.
    """
    with open(_DEFAULT_CONFIG_PATH) as f:
        config = json.load(f)
    if config_file is not None:
        with open(config_file) as f:
            custom = json.load(f)
        config = _deep_merge(config, custom)
    return config


# Module-level DEFAULTS loaded once at import time
DEFAULTS = load_config()

# Aliases/shorthands accepted by config() on top of the literal
# default_config.json keys. Each is expanded or converted in code
# (e.g. ``learn_precision`` -> both ``learn_precision_*`` flags,
# ``init_log_precision`` -> ``init_precision``).
KEY_ALIASES = {
    "model": {"learn_precision", "init_log_precision", "precision_activation"},
    "train": set(),
    "test": set(),
}


def valid_keys(section: str) -> set:
    """All keys accepted by config() for a section.

    Derived from default_config.json plus KEY_ALIASES, so adding a new
    option to the JSON automatically makes it a valid config key.
    """
    return set(DEFAULTS[section]) | KEY_ALIASES[section]


def validate_keys(section, keys):
    """Raise ValueError if any key is not a known option for ``section``.

    ``section`` may be a single section name or an iterable of names whose
    valid keys are unioned (e.g. Simulation.config accepts train+test keys).
    """
    sections = (section,) if isinstance(section, str) else tuple(section)
    allowed = set().union(*(valid_keys(s) for s in sections))
    unknown = set(keys) - allowed
    if unknown:
        raise ValueError(
            f"Unknown config key(s) {sorted(unknown)} for section(s) "
            f"{sections}. Valid keys: {sorted(allowed)}")
