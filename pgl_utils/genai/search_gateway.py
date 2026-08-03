"""
PGL Gateway search utilities
"""


# Standard library
from typing import Any, Dict

# Third-party
import requests


_DEFAULT_BASE_URL = "https://pgl-gateway.vercel.app"
_DEFAULT_TIMEOUT = 30


class PGLSearchGateway:
    """
    Thin client for the PGL Gateway search API.

    Example:
        from pgl_utils.genai.search_gateway import PGLSearchGateway

        gateway = PGLSearchGateway()
        results = gateway.search("inteligência artificial generativa")
    """

    def __init__(self, base_url: str = _DEFAULT_BASE_URL, timeout: int = _DEFAULT_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(
        self,
        q: str,
        search_depth: str = "advanced",
        include_raw_content: bool = True,
        **params: Any,
    ) -> Dict[str, Any]:
        """
        Calls the GET /search endpoint of the PGL Gateway.

        Args:
            q:                    Search query text.
            search_depth:         "basic" or "advanced".
            include_raw_content:  Whether to include each result's raw page content.
            **params:             Extra query parameters forwarded to the gateway as-is.

        Returns:
            The parsed JSON response from the gateway.
        """
        query = {
            "q": q,
            "search_depth": search_depth,
            "include_raw_content": include_raw_content,
            **params,
        }
        response = requests.get(
            f"{self.base_url}/search",
            params=query,
            headers={"accept": "application/json"},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()
