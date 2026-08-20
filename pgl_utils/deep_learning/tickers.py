"""
Curated lists of stock tickers (Brazil and US), grouped by sector.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=None)
def load_brazil_tickers() -> dict[str, list[str]]:
    """Load B3 tickers grouped by sector."""
    with resources.files("pgl_utils.deep_learning.data").joinpath(
        "brazil_tickers.json"
    ).open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def load_us_tickers() -> dict[str, list[str]]:
    """Load US tickers grouped by sector."""
    with resources.files("pgl_utils.deep_learning.data").joinpath(
        "us_tickers.json"
    ).open("r", encoding="utf-8") as f:
        return json.load(f)
