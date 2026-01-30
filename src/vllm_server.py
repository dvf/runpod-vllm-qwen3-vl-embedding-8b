"""
Spawns the vLLM OpenAI-compatible server inside the worker process and waits
until it's ready.
"""

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Optional

import requests


@dataclass(frozen=True)
class VLLMConfig:
    mode: str  # "embed" | "rerank" | "chat"
    model: str
    host: str = "127.0.0.1"
    port: int = 8000
    dtype: str = "auto"
    max_model_len: Optional[int] = None
    tensor_parallel_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    trust_remote_code: bool = True
    extra_args: str = ""  # appended raw args

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def build_command(self) -> list[str]:
        """
        Build a `vllm serve ...` command.
        """
        cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server", "--model", self.model]
        cmd += ["--host", self.host, "--port", str(self.port)]
        cmd += ["--dtype", self.dtype]

        if self.trust_remote_code:
            cmd += ["--trust-remote-code"]

        # Mode-specific defaults (grounded in vLLM docs for pooling models / Qwen3-VL embedding & reranker)
        # Embedding models: `--runner pooling` and use /v1/embeddings.
        if self.mode == "embed":
            cmd += ["--runner", "pooling", "--task", "embed"]

        # Reranker: pooling runner + overrides + template + /v1/rerank.
        if self.mode == "rerank":
            cmd += ["--runner", "pooling", "--task", "score"]
            # defaults as per vLLM guidance for Qwen3-VL-Reranker-8B
            if self.max_model_len is None:
                cmd += ["--max-model-len", "4096"]
            # hf_overrides required for this reranker series
            cmd += [
                "--hf-overrides",
                '{"architectures": ["Qwen3VLForSequenceClassification"],'
                '"classifier_from_token": ["no", "yes"],'
                '"is_original_qwen3_reranker": true}',
            ]
            template_path = os.environ.get(
                "RERANKER_CHAT_TEMPLATE",
                "/app/src/templates/qwen3_vl_reranker.jinja",
            )
            cmd += ["--chat-template", template_path]

        # Generic knobs
        if self.max_model_len is not None:
            cmd += ["--max-model-len", str(self.max_model_len)]
        if self.tensor_parallel_size is not None:
            cmd += ["--tensor-parallel-size", str(self.tensor_parallel_size)]
        if self.gpu_memory_utilization is not None:
            cmd += ["--gpu-memory-utilization", str(self.gpu_memory_utilization)]

        if self.extra_args.strip():
            cmd += shlex.split(self.extra_args)

        return cmd


class VLLMServer:
    def __init__(self, cfg: VLLMConfig):
        self.cfg = cfg
        self.proc: Optional[subprocess.Popen] = None

    def start(self, timeout_s: int = 600) -> None:
        if self.proc is not None:
            return

        cmd = self.cfg.build_command()
        # Forward stdout/stderr so RunPod logs show vLLM startup details.
        self.proc = subprocess.Popen(cmd, stdout=None, stderr=None)

        # Wait for server to respond.
        self._wait_ready(timeout_s=timeout_s)

    def _wait_ready(self, timeout_s: int) -> None:
        t0 = time.time()
        last_err: Optional[str] = None

        # Prefer /v1/models when available; otherwise just check TCP via HTTP.
        models_url = f"{self.cfg.base_url}/v1/models"

        while time.time() - t0 < timeout_s:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"vLLM exited early with code {self.proc.returncode}")

            try:
                r = requests.get(models_url, timeout=2)
                if r.status_code == 200:
                    return
                last_err = f"status={r.status_code} body={r.text[:200]}"
            except Exception as e:
                last_err = repr(e)

            time.sleep(1)

        raise TimeoutError(f"vLLM not ready after {timeout_s}s. Last error: {last_err}")

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None
