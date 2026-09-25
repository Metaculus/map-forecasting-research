"""Provider-agnostic LLM interface for the tools that need a language model.

Every LLM-backed tool (alignment, capture grading, cleaning, key-factor grading,
the builders) takes an `llm` argument. Anything with a `generate(system, prompt,
max_tokens) -> str` method works; subclass `LLM` to get structured output for free:

    from map_forecasting.llm import FunctionLLM, OpenAICompatibleLLM, set_default_llm

    llm = OpenAICompatibleLLM("gpt-4.1")                        # OpenAI
    llm = OpenAICompatibleLLM("some/model", base_url="https://openrouter.ai/api/v1",
                              api_key_env="OPENROUTER_API_KEY")
    llm = OpenAICompatibleLLM("llama3", base_url="http://localhost:11434/v1")  # local
    llm = AnthropicLLM("claude-opus-5")
    llm = FunctionLLM(lambda system, prompt: my_client(system, prompt), name="mine")
    set_default_llm(llm)                                        # or pass llm= per call

Structured output: the JSON schema of the requested pydantic model is appended to
the prompt, the reply is parsed and validated, and one repair round is attempted
if it doesn't validate. Adapters with native structured output override this.

Responses are cached on disk, keyed on (llm name, system, prompt, schema):
  MAP_FORECASTING_CACHE_DIR  cache location, ~/.cache/map_forecasting if unset;
                             set to an empty string to disable caching
  MAP_FORECASTING_LLM        optional default, "openai:<model>" or "anthropic:<model>"
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from pathlib import Path
from typing import Callable, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_lock = threading.Lock()
usage = {"calls": 0, "cached": 0}
_default: "LLM | None" = None


class LLM:
    """Base class. Subclasses implement `generate`; `structured` comes for free."""

    name: str = "llm"

    def generate(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        raise NotImplementedError

    def structured(self, system: str, prompt: str, schema: Type[T],
                   max_tokens: int = 16000) -> T:
        full = f"{prompt}\n\n{_schema_instructions(schema)}"
        text = self.generate(system, full, max_tokens)
        try:
            return _parse(text, schema)
        except (ValueError, ValidationError) as err:
            repair = (f"{full}\n\nYour previous reply could not be used:\n{err}\n\n"
                      f"Previous reply:\n{text[:4000]}\n\nReturn only the corrected JSON.")
            return _parse(self.generate(system, repair, max_tokens), schema)


class FunctionLLM(LLM):
    """Wrap any function `fn(system, prompt) -> str`."""

    def __init__(self, fn: Callable[[str, str], str], name: str = "function"):
        self.fn, self.name = fn, name

    def generate(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        return self.fn(system, prompt)


class OpenAICompatibleLLM(LLM):
    """Any OpenAI-compatible chat-completions endpoint: OpenAI, OpenRouter, vLLM,
    Ollama, LM Studio and others. Needs the `openai` package.

    `json_mode=True` asks the server for a JSON object; leave it off for servers
    that don't support `response_format`. `max_tokens_param` is sent only if set
    (some servers call it max_completion_tokens)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 api_key_env: str = "OPENAI_API_KEY", json_mode: bool = False,
                 max_tokens_param: str | None = None, **extra):
        self.model, self.base_url, self.json_mode = model, base_url, json_mode
        self.api_key = api_key or os.environ.get(api_key_env) or "not-needed"
        self.max_tokens_param, self.extra = max_tokens_param, extra
        self.name = f"openai-compatible:{base_url or 'openai'}:{model}"
        self._client = None

    def generate(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        if self._client is None:
            import openai
            self._client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        kwargs = dict(self.extra)
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.max_tokens_param:
            kwargs[self.max_tokens_param] = max_tokens
        messages = ([{"role": "system", "content": system}] if system else []) + \
            [{"role": "user", "content": prompt}]
        resp = self._client.chat.completions.create(model=self.model, messages=messages, **kwargs)
        return resp.choices[0].message.content or ""


class AnthropicLLM(LLM):
    """Anthropic models, with native structured output. Needs the `anthropic` package."""

    STREAM_THRESHOLD = 16000  # the SDK requires streaming above this max_tokens

    def __init__(self, model: str = "claude-opus-5", api_key: str | None = None):
        self.model, self.api_key = model, api_key
        self.name = f"anthropic:{model}"
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self.api_key) if self.api_key \
                else anthropic.Anthropic()
        return self._client

    def _send(self, kwargs: dict, max_tokens: int, parse: bool):
        if max_tokens > self.STREAM_THRESHOLD:
            with self.client.messages.stream(**kwargs) as stream:
                resp = stream.get_final_message()
        elif parse:
            resp = self.client.messages.parse(**kwargs)
        else:
            resp = self.client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"model refused: {getattr(resp, 'stop_details', None)}")
        if resp.stop_reason == "max_tokens":
            raise RuntimeError("hit max_tokens; raise the limit for this call")
        return resp

    def generate(self, system: str, prompt: str, max_tokens: int = 16000) -> str:
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}])
        if system:
            kwargs["system"] = system
        resp = self._send(kwargs, max_tokens, parse=False)
        return "".join(b.text for b in resp.content if b.type == "text")

    def structured(self, system: str, prompt: str, schema: Type[T],
                   max_tokens: int = 16000) -> T:
        kwargs = dict(model=self.model, max_tokens=max_tokens,
                      messages=[{"role": "user", "content": prompt}], output_format=schema)
        if system:
            kwargs["system"] = system
        resp = self._send(kwargs, max_tokens, parse=True)
        parsed = getattr(resp, "parsed_output", None)
        if parsed is None:  # the streaming path can return text only
            parsed = schema.model_validate_json(
                next(b.text for b in resp.content if b.type == "text"))
        return parsed


# ---------------------------------------------------------------- defaults

def set_default_llm(llm: LLM | None) -> None:
    global _default
    _default = llm


def get_default_llm() -> LLM:
    global _default
    if _default is None:
        spec = os.environ.get("MAP_FORECASTING_LLM", "")
        provider, _, model = spec.partition(":")
        if provider == "openai" and model:
            _default = OpenAICompatibleLLM(model, base_url=os.environ.get("OPENAI_BASE_URL"))
        elif provider == "anthropic" and model:
            _default = AnthropicLLM(model)
        else:
            raise RuntimeError(
                "No LLM configured. Pass llm=..., call map_forecasting.llm.set_default_llm(...), "
                "or set MAP_FORECASTING_LLM to 'openai:<model>' or 'anthropic:<model>'.")
    return _default


# ---------------------------------------------------------------- structured calls + cache

def cache_dir() -> Path | None:
    raw = os.environ.get("MAP_FORECASTING_CACHE_DIR")
    if raw == "":
        return None
    return Path(raw).expanduser() if raw else Path.home() / ".cache" / "map_forecasting"


def _cache_file(name: str, system: str, prompt: str, schema: Type[T], variant: int) -> Path | None:
    root = cache_dir()
    if root is None:
        return None
    payload = {"llm": name, "system": system, "prompt": prompt,
               "schema": schema.model_json_schema(), "variant": variant}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    return root / f"{schema.__name__}_{key}.json"


def call_structured(prompt: str, schema: Type[T], system: str = "", llm: LLM | None = None,
                    max_tokens: int = 16000, cache: bool = True, variant: int = 0) -> T:
    """Ask `llm` (or the default) for a response matching the pydantic `schema`.

    `variant` lets the same prompt be sampled more than once without the cache
    returning the first answer every time."""
    llm = llm or get_default_llm()
    path = _cache_file(getattr(llm, "name", type(llm).__name__), system, prompt, schema,
                       variant) if cache else None
    if path is not None and path.exists():
        with _lock:
            usage["cached"] += 1
        return schema.model_validate(json.loads(path.read_text())["parsed"])
    structured = getattr(llm, "structured", None)
    if structured is not None:
        out = structured(system, prompt, schema, max_tokens)
    else:  # any object with generate()
        out = LLM.structured(llm, system, prompt, schema, max_tokens)
    with _lock:
        usage["calls"] += 1
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"llm": getattr(llm, "name", ""), "parsed": out.model_dump()},
                                   indent=1))
    return out


def _schema_instructions(schema: Type[BaseModel]) -> str:
    return ("Respond with a single JSON object, and nothing else, that validates against "
            "this JSON schema:\n" + json.dumps(schema.model_json_schema(), indent=1))


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _parse(text: str, schema: Type[T]) -> T:
    """Validate the first JSON object found in `text` (code fences allowed)."""
    m = _FENCE.search(text)
    body = m.group(1) if m else text
    start = body.find("{")
    if start < 0:
        raise ValueError("no JSON object in the reply")
    obj, _ = json.JSONDecoder().raw_decode(body[start:])
    return schema.model_validate(obj)
