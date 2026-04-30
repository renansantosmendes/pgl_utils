import matplotlib
matplotlib.use('Agg')

# groq.Groq must be mocked before pgl_utils.genai.llm is first imported,
# because llm.py calls _resolve_default_model() at module level which hits the API.
# conftest.py is loaded before any test file, so this pre-load is safe.
from unittest.mock import MagicMock, patch as _patch

_mock_model = MagicMock()
_mock_model.id = "llama-3.3-70b-versatile"
_mock_model.created = 1_000_000

with _patch("groq.Groq") as _groq_mock:
    _groq_mock.return_value.models.list.return_value.data = [_mock_model]
    import pgl_utils.genai.llm  # noqa: E402

del _mock_model, _groq_mock, _patch, MagicMock

import pytest
from unittest.mock import patch


@pytest.fixture(autouse=True)
def mock_api_key():
    with patch("pgl_utils.genai.llm._get_api_key", return_value="test-key"):
        yield
