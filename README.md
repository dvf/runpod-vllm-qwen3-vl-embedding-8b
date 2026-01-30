# runpod-vllm-qwen3

A RunPod Serverless (queue-based) worker that starts a local vLLM OpenAI-compatible server
and forwards RunPod jobs to it.

Supports:
- Qwen3-VL-Embedding via `/v1/embeddings` (pooling runner)
- Qwen3-VL-Reranker via `/v1/rerank` (pooling runner + hf_overrides + custom template)
- Qwen3-VL chat via `/v1/chat/completions`

## Why this exists

RunPod's prebuilt vLLM serverless worker is great for text-generation models, but embedding/reranking
variants frequently need explicit vLLM flags (`--runner pooling` and `--task embed`, reranker overrides, etc.).

### Recommended environment variables

**Embeddings (Qwen3-VL-Embedding-8B)**

```
VLLM_MODE=embed
VLLM_MODEL=Qwen/Qwen3-VL-Embedding-8B
VLLM_TRUST_REMOTE_CODE=true
```

Optional:

```
VLLM_DTYPE=auto
VLLM_MAX_MODEL_LEN=32768        # context of qwen3-vl
VLLM_EXTRA_ARGS=--gpu-memory-utilization 0.90
```

**Rerank (Qwen3-VL-Reranker-8B)**

```
VLLM_MODE=rerank
VLLM_MODEL=Qwen/Qwen3-VL-Reranker-8B
VLLM_TRUST_REMOTE_CODE=true
```

Optional:

```
VLLM_MAX_MODEL_LEN=4096
RERANKER_CHAT_TEMPLATE=/app/src/templates/qwen3_vl_reranker.jinja
```

**HF token** (only if gated/private):

```
HF_TOKEN=...   # RunPod Secret recommended
```

## Calling the endpoint

RunPod queue endpoints accept requests like:

```json
{
  "input": { ... }
}
```

### Embeddings example

```json
{
  "input": {
    "mode": "embed",
    "input": ["The capital of China is Beijing.", "Gravity is a force..."]
  }
}
```

### Rerank example

```json
{
  "input": {
    "mode": "rerank",
    "query": "What is the capital of China?",
    "documents": [
      "The capital of China is Beijing.",
      "Gravity is a force that attracts two bodies..."
    ]
  }
}
```

### Chat (multimodal) example

```json
{
  "input": {
    "mode": "chat",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://.../receipt.png"}},
          {"type": "text", "text": "Read all the text in the image."}
        ]
      }
    ],
    "max_tokens": 512
  }
}
```

## Notes

- For Qwen3-VL embedding/rerank, vLLM uses pooling mode and OpenAI-compatible endpoints.
- The worker starts vLLM once per container boot and reuses it across jobs.

## Default behavior you'll get (no extra config)

If you deploy with no env vars, it defaults to:

- `VLLM_MODE=embed`
- `VLLM_MODEL=Qwen/Qwen3-VL-Embedding-8B`

...and it will start vLLM with pooling runner (embedding-ready) and forward jobs to `/v1/embeddings`.
