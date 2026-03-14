"""
vLLM API Client.

Communicates with a vLLM server running in WSL2 via its
OpenAI-compatible REST API. Handles completions, health checks,
and weight synchronization signals.
"""

import httpx
import json
from typing import Dict, List, Optional, Any


class VLLMEngine:
    """
    Client for the vLLM inference server in WSL2.

    Wraps the OpenAI-compatible /v1/completions and /v1/chat/completions
    endpoints. Intended for high-throughput generation of reasoning
    trajectories during the validation phase of bilevel optimization.

    Args:
        api_base: Base URL for the vLLM server (default: http://localhost:8000/v1).
        model_name: HuggingFace model name served by vLLM.
        max_tokens: Max tokens to generate per request.
        temperature: Sampling temperature (0.0 = greedy).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_tokens: int = 512,
        temperature: float = 0.0,
        timeout: float = 60.0,
    ):
        self.api_base = api_base.rstrip("/")
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def health_check(self) -> bool:
        """Check if the vLLM server is reachable."""
        try:
            resp = self._client.get(f"{self.api_base}/models")
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text completion from the vLLM server.

        Args:
            prompt: Input prompt string.
            max_tokens: Override max tokens.
            temperature: Override temperature.
            stop: Stop sequences.
            logprobs: Number of logprobs to return.

        Returns:
            Dict with 'text', 'usage', and 'finish_reason'.
        """
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if stop:
            payload["stop"] = stop
        if logprobs is not None:
            payload["logprobs"] = logprobs

        resp = self._client.post(
            f"{self.api_base}/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return {
            "text": choice["text"],
            "finish_reason": choice.get("finish_reason", "unknown"),
            "usage": data.get("usage", {}),
            "logprobs": choice.get("logprobs"),
        }

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Chat completion via the vLLM server.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts.
            max_tokens: Override max tokens.
            temperature: Override temperature.

        Returns:
            Dict with 'text', 'usage', and 'finish_reason'.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }

        resp = self._client.post(
            f"{self.api_base}/chat/completions",
            json=payload,
        )
        resp.raise_for_status()
        data = resp.json()

        choice = data["choices"][0]
        return {
            "text": choice["message"]["content"],
            "finish_reason": choice.get("finish_reason", "unknown"),
            "usage": data.get("usage", {}),
        }

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate completions for multiple prompts.

        Args:
            prompts: List of prompt strings.
            max_tokens: Override max tokens.

        Returns:
            List of result dicts.
        """
        return [
            self.generate(prompt, max_tokens=max_tokens)
            for prompt in prompts
        ]

    def close(self):
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
