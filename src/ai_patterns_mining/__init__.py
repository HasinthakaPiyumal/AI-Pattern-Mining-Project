"""Expose the pipeline package under the ai_patterns_mining namespace."""

from pipeline import __version__, main
from config.settings import Config
from utils.json_utils import parse_json_safe

__all__ = ["main", "__version__", "Config", "parse_json_safe"]
