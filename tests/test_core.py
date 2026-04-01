"""
Tests for core module
"""

import pytest
from post_graduation_utils.core import utils


def test_placeholder():
    """Test placeholder function"""
    result = utils.placeholder()
    assert result == "Core utilities"
