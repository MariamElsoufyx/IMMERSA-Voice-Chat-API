"""Local LLM service — drop-in for LLMGroqService.

Mirrors generate_reply(user_prompt, system_prompt, history) exactly so the
pipeline doesn't care whether replies come from Groq or a local runtime.

Two backends are supported (selected via config.local_llm_backend):
- "ollama"   → HTTP to a running `ollama serve` (default; zero-Python deps)
- "llamacpp" → in-process llama-cpp-python (no network; needs the .gguf path)

Both return a JSON-object reply string identical in shape to what Groq emits
(response_format=json_object), so app.services.pipeline.pipeline._parse_llm_reply
parses local output without any branching.
"""
import json

import httpx

import app.core.config as config


class LLMLocalService:
    def __init__(self, model_path: str | None = None):
        self.backend = config.local_llm_backend
        self.model_name = config.local_llm_model_name
        self.max_tokens = config.local_llm_max_tokens
        self.temperature = config.local_llm_temperature
        self._llama = None  # lazy llama-cpp-python handle

        if self.backend == "llamacpp":
            from llama_cpp import Llama
            self._llama = Llama(
                model_path=model_path or config.local_llm_model_path,
                n_ctx=config.local_llm_n_ctx,
                n_threads=config.local_llm_n_threads,
                verbose=False,
            )
        print(f"✅ [LLM] Local LLM ready (backend={self.backend}, model={self.model_name})")

    def generate_reply(self, user_prompt: str, system_prompt: str, history: list | None = None) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})

        try:
            if self.backend == "ollama":
                return self._reply_ollama(messages)
            return self._reply_llamacpp(messages)
        except Exception as e:
            print(f"[LOCAL LLM ERROR] {e}")
            return ""

    def _reply_ollama(self, messages: list) -> str:
        """POST /api/chat on the local Ollama daemon. Streams tokens like Groq."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "format": "json",
            "options": {
                "num_predict": self.max_tokens,
                "temperature": self.temperature,
            },
            "stream": True,
        }
        tokens: list[str] = []
        with httpx.stream(
            "POST",
            f"{config.local_llm_ollama_url}/api/chat",
            json=payload,
            timeout=config.local_llm_timeout_s,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                obj = json.loads(line)
                delta = (obj.get("message") or {}).get("content")
                if delta:
                    tokens.append(delta)
                if obj.get("done"):
                    break
        return "".join(tokens).strip().replace("\n", " ")

    def _reply_llamacpp(self, messages: list) -> str:
        """In-process llama-cpp-python — no daemon, no network."""
        completion = self._llama.create_chat_completion(
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
            stream=False,
        )
        return (completion["choices"][0]["message"]["content"] or "").strip().replace("\n", " ")
