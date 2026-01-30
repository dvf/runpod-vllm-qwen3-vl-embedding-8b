"""
RunPod serverless handler – routes job payloads to the correct vLLM endpoint:

  embed  → POST /v1/embeddings
  rerank → POST /v1/rerank
  chat   → POST /v1/chat/completions (multimodal chat flow in Qwen3-VL recipe)
"""

import os
from typing import Any, Dict, Optional

import requests
import runpod

from .vllm_server import VLLMConfig, VLLMServer

_SERVER: Optional[VLLMServer] = None


def _get_env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _init_server() -> VLLMServer:
    global _SERVER
    if _SERVER is not None:
        return _SERVER

    mode = os.environ.get("VLLM_MODE", "embed").strip().lower()
    model = os.environ.get("VLLM_MODEL", "Qwen/Qwen3-VL-Embedding-8B").strip()

    cfg = VLLMConfig(
        mode=mode,
        model=model,
        host=os.environ.get("VLLM_HOST", "127.0.0.1"),
        port=int(os.environ.get("VLLM_PORT", "8000")),
        dtype=os.environ.get("VLLM_DTYPE", "auto"),
        max_model_len=int(os.environ["VLLM_MAX_MODEL_LEN"]) if os.environ.get("VLLM_MAX_MODEL_LEN") else None,
        tensor_parallel_size=int(os.environ["VLLM_TP_SIZE"]) if os.environ.get("VLLM_TP_SIZE") else None,
        gpu_memory_utilization=float(os.environ["VLLM_GPU_MEM_UTIL"]) if os.environ.get("VLLM_GPU_MEM_UTIL") else None,
        trust_remote_code=_get_env_bool("VLLM_TRUST_REMOTE_CODE", True),
        extra_args=os.environ.get("VLLM_EXTRA_ARGS", ""),
    )

    server = VLLMServer(cfg)
    server.start(timeout_s=int(os.environ.get("VLLM_STARTUP_TIMEOUT_S", "900")))
    _SERVER = server
    return server


def _post_json(url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    # Optional auth header if you set an API key in front of vLLM; usually not needed
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("VLLM_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    r = requests.post(url, json=payload, headers=headers, timeout=int(os.environ.get("VLLM_HTTP_TIMEOUT_S", "600")))
    r.raise_for_status()
    return r.json()


def handler(job: Dict[str, Any]) -> Any:
    """
    RunPod job schema:
      { "id": "...", "input": {...} }
    The handler returns JSON-serializable output.
    """
    server = _init_server()
    inp = job.get("input", {}) or {}

    mode = (inp.get("mode") or os.environ.get("VLLM_MODE", "embed")).strip().lower()
    base = server.cfg.base_url

    # --- Embeddings (OpenAI-compatible) ---
    # Expected input example:
    # {
    #   "mode": "embed",
    #   "input": ["text1", "text2"]  OR "input": "single text",
    #   "model": "optional override",
    #   ... optionally "encoding_format", "dimensions", etc ...
    # }
    if mode in ("embed", "embeddings"):
        url = f"{base}/v1/embeddings"
        payload = dict(inp)
        payload.pop("mode", None)
        # vLLM requires model in many setups; default to our configured one
        payload.setdefault("model", server.cfg.model)
        return _post_json(url, payload)

    # --- Rerank (OpenAI-compatible rerank endpoint in vLLM) ---
    # Expected input example:
    # {
    #   "mode": "rerank",
    #   "query": "...",
    #   "documents": ["doc1", "doc2", ...],
    #   "model": "optional override"
    # }
    if mode == "rerank":
        url = f"{base}/v1/rerank"
        payload = dict(inp)
        payload.pop("mode", None)
        payload.setdefault("model", server.cfg.model)
        return _post_json(url, payload)

    # --- Chat completions (multimodal) ---
    # Expected input example (OpenAI chat format):
    # {
    #   "mode": "chat",
    #   "messages": [...],
    #   "max_tokens": 512,
    #   "model": "optional override"
    # }
    if mode in ("chat", "chat.completions", "chat_completions"):
        url = f"{base}/v1/chat/completions"
        payload = dict(inp)
        payload.pop("mode", None)
        payload.setdefault("model", server.cfg.model)
        return _post_json(url, payload)

    raise ValueError(f"Unknown mode={mode}. Use 'embed', 'rerank', or 'chat'.")


runpod.serverless.start({"handler": handler})
