"""
Large Language Model utilities
"""


# Standard library
import os
import time
from typing import Any, Iterator, List, Optional

# Third-party
from groq import Groq
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from pydantic import ConfigDict


_FALLBACK_MODEL = "llama-3.3-70b-versatile"


def _get_api_key() -> Optional[str]:
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    try:
        from pgl_utils._keys import GROQ_API_KEY as _embedded
        return _embedded or None
    except ImportError:
        return None


def _fetch_available_models() -> List:
    api_key = _get_api_key()
    client = Groq(api_key=api_key)
    models = client.models.list()
    return sorted(models.data, key=lambda m: m.created, reverse=True)


def _resolve_default_model() -> str:
    try:
        models = _fetch_available_models()
        llama = next((m.id for m in models if "llama-3.3" in m.id), None)
        return llama if llama else _FALLBACK_MODEL
    except Exception:
        return _FALLBACK_MODEL


_DEFAULT_MODEL = _resolve_default_model()


class ChatPGL(BaseChatModel):
    """
    ChatGroq wrapper pre-configured for PGL classes.
    Students do not need to provide an API key.
    Uses LangChain's ChatMessageHistory for conversation memory.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: str = _DEFAULT_MODEL
    temperature: float = 0.7
    max_tokens: int = 1024
    stream_delay: float = 0.02
    system_prompt: Optional[str] = None
    session_id: str = "default"
    _client: Any = None
    _store: dict = {}
    _chain: Any = None

    def __init__(self, **kwargs):
        if "model" in kwargs:
            available = [m.id for m in _fetch_available_models()]
            if kwargs["model"] not in available:
                raise ValueError(
                    f"Model '{kwargs['model']}' is not available.\n"
                    f"Available models: {available}"
                )
        super().__init__(**kwargs)

        api_key = _get_api_key()
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY não encontrada. "
                "Defina a variável de ambiente antes de criar o ChatPGL:\n"
                '  import os; os.environ["GROQ_API_KEY"] = "sua-chave"'
            )

        self._store = {}
        self._client = ChatGroq(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
        )
        self._chain = self._build_chain()

    def _build_chain(self) -> RunnableWithMessageHistory:
        system = self.system_prompt or "You are a helpful assistant."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])

        return RunnableWithMessageHistory(
            prompt | self._client,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def invoke(self, input: Any, session_id: Optional[str] = None, **kwargs) -> Any:
        """
        Runs inference using LangChain memory. Conversation history is
        automatically managed per session_id.

        Args:
            input:      A string message.
            session_id: Conversation session identifier. Defaults to self.session_id.
        """
        sid = session_id or self.session_id
        return self._chain.invoke(
            {"input": input},
            config={"configurable": {"session_id": sid}},
            **kwargs,
        )

    def clear_memory(self, session_id: Optional[str] = None):
        """Clears the conversation memory for the given session."""
        sid = session_id or self.session_id
        if sid in self._store:
            self._store[sid].clear()

    def get_memory(self, session_id: Optional[str] = None) -> List[BaseMessage]:
        """Returns the conversation history for the given session."""
        sid = session_id or self.session_id
        return self._get_session_history(sid).messages

    @staticmethod
    def list_models() -> List[str]:
        """Fetches and returns the list of available models from the Groq API, sorted by most recent."""
        return [m.id for m in _fetch_available_models()]

    @property
    def _llm_type(self) -> str:
        return "chat-pgl"

    @property
    def _identifying_params(self) -> dict:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream_delay": self.stream_delay,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._client._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        yield from self._client._stream(messages, stop=stop, run_manager=run_manager, **kwargs)

    def bind_tools(self, tools, **kwargs):
        return self._client.bind_tools(tools, **kwargs)

    def streamed_invoke(self, input: Any, session_id: Optional[str] = None, stream_delay: Optional[float] = None) -> str:
        """
        Runs inference with a typing effect, printing the response token by token.
        Conversation history is automatically managed per session_id.
        Returns the full response as a string when complete.

        Args:
            input:        A string message.
            session_id:   Conversation session identifier. Defaults to self.session_id.
            stream_delay: Seconds to wait between each chunk. Defaults to self.stream_delay.
        """
        sid = session_id or self.session_id
        delay = stream_delay if stream_delay is not None else self.stream_delay
        full_response = ""

        for chunk in self._chain.stream(
            {"input": input},
            config={"configurable": {"session_id": sid}},
        ):
            token = chunk.content
            print(token, end="", flush=True)
            full_response += token
            time.sleep(delay)

        print()
        return full_response
