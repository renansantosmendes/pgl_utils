"""
Unit tests for ChatPGL class.
All Groq / LangChain network calls are mocked — no real API key required.
"""
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

_LLM = "pgl_utils.genai.llm"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_chat_groq():
    with patch(f"{_LLM}.ChatGroq") as mock:
        yield mock


@pytest.fixture
def llm(mock_chat_groq):
    from pgl_utils.genai.llm import ChatPGL
    pgl = ChatPGL()
    pgl._chain = MagicMock()
    pgl._chain.invoke.return_value = AIMessage(content="Mocked response")
    pgl._chain.stream.return_value = iter([
        MagicMock(content="Mocked "),
        MagicMock(content="streamed "),
        MagicMock(content="response"),
    ])
    return pgl


@pytest.fixture
def llm_with_system(mock_chat_groq):
    from pgl_utils.genai.llm import ChatPGL
    pgl = ChatPGL(system_prompt="You are a helpful tutor.")
    pgl._chain = MagicMock()
    pgl._chain.invoke.return_value = AIMessage(content="Mocked response")
    return pgl


# ---------------------------------------------------------------------------
# _fetch_available_models
# ---------------------------------------------------------------------------

class TestFetchAvailableModels:
    def test_returns_sorted_by_created_descending(self):
        mock_a = MagicMock()
        mock_a.id = "model-old"
        mock_a.created = 1000
        mock_b = MagicMock()
        mock_b.id = "model-new"
        mock_b.created = 9000
        mock_response = MagicMock()
        mock_response.data = [mock_a, mock_b]
        with patch(f"{_LLM}.Groq") as mock_groq:
            mock_groq.return_value.models.list.return_value = mock_response
            from pgl_utils.genai.llm import _fetch_available_models
            result = _fetch_available_models()
            assert result[0].id == "model-new"
            assert result[1].id == "model-old"

    def test_returns_list(self):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(id="a", created=100)]
        with patch(f"{_LLM}.Groq") as mock_groq:
            mock_groq.return_value.models.list.return_value = mock_response
            from pgl_utils.genai.llm import _fetch_available_models
            result = _fetch_available_models()
            assert isinstance(result, list)

    def test_uses_pgl_api_key(self):
        mock_response = MagicMock()
        mock_response.data = []
        with patch(f"{_LLM}.Groq") as mock_groq, \
             patch(f"{_LLM}._PGL_API_KEY", "test-key"):
            mock_groq.return_value.models.list.return_value = mock_response
            from pgl_utils.genai.llm import _fetch_available_models
            _fetch_available_models()
            mock_groq.assert_called_once_with(api_key="test-key")


# ---------------------------------------------------------------------------
# _resolve_default_model
# ---------------------------------------------------------------------------

class TestResolveDefaultModel:
    def test_prefers_llama_33_model(self):
        mock_llama = MagicMock()
        mock_llama.id = "llama-3.3-70b-versatile"
        mock_other = MagicMock()
        mock_other.id = "other-model"
        with patch(f"{_LLM}._fetch_available_models", return_value=[mock_other, mock_llama]):
            from pgl_utils.genai.llm import _resolve_default_model
            assert _resolve_default_model() == "llama-3.3-70b-versatile"

    def test_falls_back_when_api_fails(self):
        with patch(f"{_LLM}._fetch_available_models", side_effect=Exception("API error")):
            from pgl_utils.genai.llm import _resolve_default_model, _FALLBACK_MODEL
            assert _resolve_default_model() == _FALLBACK_MODEL

    def test_falls_back_when_no_llama_33_available(self):
        mock_model = MagicMock()
        mock_model.id = "some-other-model"
        with patch(f"{_LLM}._fetch_available_models", return_value=[mock_model]):
            from pgl_utils.genai.llm import _resolve_default_model, _FALLBACK_MODEL
            assert _resolve_default_model() == _FALLBACK_MODEL


# ---------------------------------------------------------------------------
# ChatPGL — instantiation
# ---------------------------------------------------------------------------

class TestChatPGLInit:
    def test_default_instantiation(self, llm):
        assert llm is not None
        assert llm.temperature == 0.7
        assert llm.max_tokens == 1024
        assert llm.stream_delay == 0.02
        assert llm.session_id == "default"
        assert llm.system_prompt is None

    def test_custom_parameters(self, mock_chat_groq):
        from pgl_utils.genai.llm import ChatPGL
        pgl = ChatPGL(temperature=0.2, max_tokens=512, stream_delay=0.05)
        assert pgl.temperature == 0.2
        assert pgl.max_tokens == 512
        assert pgl.stream_delay == 0.05

    def test_invalid_model_raises_value_error(self, mock_chat_groq):
        from pgl_utils.genai.llm import ChatPGL
        with pytest.raises(ValueError, match="is not available"):
            ChatPGL(model="gpt-invalid-model")

    def test_llm_type(self, llm):
        assert llm._llm_type == "chat-pgl"

    def test_identifying_params_keys(self, llm):
        params = llm._identifying_params
        for key in ("model", "temperature", "max_tokens", "stream_delay"):
            assert key in params

    def test_identifying_params_values(self, llm):
        params = llm._identifying_params
        assert params["temperature"] == 0.7
        assert params["max_tokens"] == 1024

    def test_chat_groq_called_with_api_key(self, mock_chat_groq):
        from pgl_utils.genai.llm import ChatPGL, _PGL_API_KEY
        ChatPGL()
        _, kwargs = mock_chat_groq.call_args
        assert kwargs["api_key"] == _PGL_API_KEY

    def test_store_starts_empty(self, mock_chat_groq):
        from pgl_utils.genai.llm import ChatPGL
        pgl = ChatPGL()
        assert pgl._store == {}

    def test_chain_is_built_on_init(self, mock_chat_groq):
        from pgl_utils.genai.llm import ChatPGL
        pgl = ChatPGL()
        assert pgl._chain is not None


# ---------------------------------------------------------------------------
# ChatPGL — invoke
# ---------------------------------------------------------------------------

class TestChatPGLInvoke:
    def test_invoke_returns_ai_message(self, llm):
        response = llm.invoke("Hello")
        assert isinstance(response, AIMessage)

    def test_invoke_with_custom_session_id(self, llm):
        response = llm.invoke("Hello", session_id="student_1")
        assert isinstance(response, AIMessage)

    def test_invoke_calls_chain_with_correct_session(self, llm):
        llm.invoke("Hello", session_id="s1")
        config = llm._chain.invoke.call_args[1]["config"]
        assert config["configurable"]["session_id"] == "s1"

    def test_invoke_uses_default_session_id(self, llm):
        llm.invoke("Hello")
        config = llm._chain.invoke.call_args[1]["config"]
        assert config["configurable"]["session_id"] == "default"

    def test_invoke_uses_instance_session_id(self, mock_chat_groq):
        from pgl_utils.genai.llm import ChatPGL
        pgl = ChatPGL(session_id="my-session")
        pgl._chain = MagicMock()
        pgl._chain.invoke.return_value = AIMessage(content="ok")
        pgl.invoke("Hello")
        config = pgl._chain.invoke.call_args[1]["config"]
        assert config["configurable"]["session_id"] == "my-session"


# ---------------------------------------------------------------------------
# ChatPGL — memory
# ---------------------------------------------------------------------------

class TestChatPGLMemory:
    def test_get_memory_returns_list(self, llm):
        assert isinstance(llm.get_memory(), list)

    def test_get_memory_new_session_is_empty(self, llm):
        assert llm.get_memory("new-session") == []

    def test_memory_accumulates_messages(self, llm):
        llm._get_session_history("s").add_message(HumanMessage(content="first"))
        llm._get_session_history("s").add_message(AIMessage(content="second"))
        assert len(llm.get_memory("s")) == 2

    def test_clear_memory_empties_history(self, llm):
        llm._get_session_history("s").add_message(HumanMessage(content="hello"))
        llm.clear_memory("s")
        assert llm.get_memory("s") == []

    def test_clear_memory_uses_default_session(self, llm):
        sid = llm.session_id
        llm._get_session_history(sid).add_message(HumanMessage(content="hello"))
        llm.clear_memory()
        assert llm.get_memory() == []

    def test_clear_memory_specific_session(self, llm):
        llm._get_session_history("s1").add_message(HumanMessage(content="hello"))
        llm._get_session_history("s2").add_message(HumanMessage(content="world"))
        llm.clear_memory("s1")
        assert llm.get_memory("s1") == []
        assert len(llm.get_memory("s2")) > 0

    def test_sessions_are_isolated(self, llm):
        llm._get_session_history("a").add_message(HumanMessage(content="a"))
        llm._get_session_history("b").add_message(HumanMessage(content="b"))
        assert llm.get_memory("a") != llm.get_memory("b")

    def test_clear_nonexistent_session_does_not_raise(self, llm):
        llm.clear_memory("ghost")


# ---------------------------------------------------------------------------
# ChatPGL — streamed_invoke
# ---------------------------------------------------------------------------

class TestChatPGLStreamedInvoke:
    def _make_pgl(self, chunks, stream_delay=0):
        from pgl_utils.genai.llm import ChatPGL
        pgl = ChatPGL(stream_delay=stream_delay)
        pgl._chain = MagicMock()
        pgl._chain.stream.return_value = iter([MagicMock(content=c) for c in chunks])
        return pgl

    def test_streamed_invoke_returns_string(self, mock_chat_groq, capsys):
        pgl = self._make_pgl(["hello"])
        result = pgl.streamed_invoke("Hello")
        assert isinstance(result, str)

    def test_streamed_invoke_concatenates_chunks(self, mock_chat_groq, capsys):
        pgl = self._make_pgl(["Hello", " world"])
        result = pgl.streamed_invoke("hi")
        assert result == "Hello world"

    def test_streamed_invoke_prints_output(self, mock_chat_groq, capsys):
        pgl = self._make_pgl(["hello"])
        pgl.streamed_invoke("Hello")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_streamed_invoke_empty_stream_returns_empty_string(self, mock_chat_groq, capsys):
        pgl = self._make_pgl([])
        result = pgl.streamed_invoke("Hello")
        assert result == ""

    def test_streamed_invoke_with_custom_delay(self, mock_chat_groq, capsys):
        pgl = self._make_pgl(["a", "b"], stream_delay=0)
        result = pgl.streamed_invoke("Hello", stream_delay=0.0)
        assert result == "ab"

    def test_streamed_invoke_uses_session_id(self, mock_chat_groq, capsys):
        pgl = self._make_pgl(["ok"])
        pgl.streamed_invoke("Hello", session_id="student_stream")
        config = pgl._chain.stream.call_args[1]["config"]
        assert config["configurable"]["session_id"] == "student_stream"


# ---------------------------------------------------------------------------
# ChatPGL — delegation to _client
# ---------------------------------------------------------------------------

class TestChatPGLDelegation:
    def test_bind_tools_delegates_to_client(self, llm):
        mock_client = MagicMock()
        llm._client = mock_client
        tools = [MagicMock()]
        llm.bind_tools(tools)
        mock_client.bind_tools.assert_called_once_with(tools)

    def test_generate_delegates_to_client(self, llm):
        mock_client = MagicMock()
        llm._client = mock_client
        messages = [HumanMessage(content="hi")]
        llm._generate(messages)
        mock_client._generate.assert_called_once_with(messages, stop=None, run_manager=None)

    def test_stream_delegates_to_client(self, llm):
        mock_client = MagicMock()
        llm._client = mock_client
        mock_client._stream.return_value = iter([MagicMock()])
        messages = [HumanMessage(content="hi")]
        list(llm._stream(messages))
        mock_client._stream.assert_called_once_with(messages, stop=None, run_manager=None)


# ---------------------------------------------------------------------------
# ChatPGL — list_models
# ---------------------------------------------------------------------------

class TestChatPGLListModels:
    def test_list_models_returns_list_of_strings(self):
        with patch(f"{_LLM}._fetch_available_models") as mock_fetch:
            mock_fetch.return_value = [MagicMock(id="model-a"), MagicMock(id="model-b")]
            from pgl_utils.genai.llm import ChatPGL
            models = ChatPGL.list_models()
            assert isinstance(models, list)
            assert all(isinstance(m, str) for m in models)

    def test_list_models_returns_correct_ids(self):
        with patch(f"{_LLM}._fetch_available_models") as mock_fetch:
            mock_fetch.return_value = [MagicMock(id="model-a"), MagicMock(id="model-b")]
            from pgl_utils.genai.llm import ChatPGL
            assert ChatPGL.list_models() == ["model-a", "model-b"]

    def test_list_models_empty(self):
        with patch(f"{_LLM}._fetch_available_models", return_value=[]):
            from pgl_utils.genai.llm import ChatPGL
            assert ChatPGL.list_models() == []
