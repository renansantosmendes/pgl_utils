"""
Tests for core module
"""

import pytest
from pgl_utils.core import utils


def test_placeholder():
    """Test placeholder function"""
    result = utils.placeholder()
    assert result == "Core utilities"
