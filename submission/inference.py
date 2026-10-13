"""ORena SAVE FOCUS FRAME submission using the all-public Qwen3.8-27B FP8 model."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SEED = 20260728
MAX_MODEL_LEN = 4224
MAX_NEW_TOKENS = 128
IMAGE_MIN_PIXELS = 32 * 32
IMAGE_MAX_PIXELS = 768 * 768
KV_CACHE_BYTES = 1 * 1024**3

APP_PATH = Path(__file__).resolve().parent
# The Docker build copies the exported checkpoint into resources/model.
# ORENA_MODEL_PATH remains available for non-platform diagnostics.
MODEL_PATH = Path(os.environ.get("ORENA_MODEL_PATH", APP_PATH / "resources" / "model"))
INPUT_PATH = Path(os.environ.get("ORENA_INPUT_PATH", "/input"))
OUTPUT_PATH = Path(os.environ.get("ORENA_OUTPUT_PATH", "/output"))
FRAME_DIR = INPUT_PATH / "frames"

# This is deliberately identical to the system prompt used for fine-tuning.
SYSTEM_PROMPT = (
    "You are a surgical assistant. You are given endoscopic video from a "
    "minimally invasive procedure. Analyze the footage and answer the surgical "
    "question based on the visual evidence. Be precise and concise."
)


@dataclass
class Request:
    """Wire-compatible subset of ``focus.Request`` used by the FRAME track."""

    qID: str  # noqa: N815
    videoID: str  # noqa: N815
    start_time: float
    end_time: float
    procedure_type: str
    question: str


@dataclass
class Response:
    """Wire-compatible representation of ``focus.Response``."""

    qID: str  # noqa: N815
    content: str
    latency: float = 0.0


def load_requests(path: Path) -> list[Request]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("request.json must contain a JSON list")
    return [Request(**row) for row in rows]


def save_responses(items: list[Response], path: Path) -> None:
    path.write_text(
        json.dumps([asdict(item) for item in items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_definitions(path: Path) -> str:
    """Load plain-text, JSON-string, JSON-value, or empty FO definitions."""
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return ""
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return text
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def frame_path_for(req: Request) -> Path:
    """Return the frame associated with a request, rejecting unsafe qIDs."""
    frame = (FRAME_DIR / f"{req.qID}.png").resolve()
    root = FRAME_DIR.resolve()
    if not frame.is_relative_to(root):
        raise ValueError(f"Unsafe qID: {req.qID!r}")
    if not frame.is_file():
        raise FileNotFoundError(f"Frame not found: {frame}")
    return frame


def make_conversation(req: Request, fo_definitions: str) -> list[dict[str, Any]]:
    """Construct the same multimodal prompt structure used during training."""
    frame = frame_path_for(req)
    system = f"{SYSTEM_PROMPT}\n\n{fo_definitions.strip()}"
    instruction = f"Procedure: {req.procedure_type}\nQuestion: {req.question}"
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": frame.as_uri()}},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def clean_answer(text: str) -> str:
    """Remove an accidental thinking block while otherwise preserving the answer."""
    answer = text.strip()
    answer = re.sub(r"^<think>.*?</think>\s*", "", answer, flags=re.DOTALL)
    return answer.strip()


def load_engine():
    """Load the merged model exactly once for the container run."""
    import vllm
    from vllm import LLM
    from vllm.v1.worker import gpu_worker

    if not (MODEL_PATH / "config.json").is_file():
        raise FileNotFoundError(f"Merged model is incomplete: {MODEL_PATH}")

    if vllm.__version__ != "0.21.0":
        raise RuntimeError(f"Submission expects vLLM 0.21.0, found {vllm.__version__}")

    # vLLM's automatic memory probe performs a full dummy forward pass. On an
    # L40S that adds roughly two minutes even in eager mode. This image pins the
    # vLLM version and reserves a 1 GiB KV cache explicitly. That still holds
    # four maximum-length sequences (the scheduler cap), while minimizing cold
    # allocation time and leaving ample room for vision/eager activations.
    def fixed_kv_cache_memory(_worker: gpu_worker.Worker) -> int:
        log.info(
            "Using fixed %.1f GiB KV cache; skipping vLLM dummy profile",
            KV_CACHE_BYTES / 1024**3,
        )
        return KV_CACHE_BYTES

    gpu_worker.Worker.determine_available_memory = fixed_kv_cache_memory
    # The general warm-up ends with a dummy sampler forward that triggers about
    # two minutes of first-use compilation on a cold L40S. Returning the neutral
    # timing result skips only warm-up; the real batch initializes exactly the
    # kernels it requires and completed within the pooled official-batch budget.
    gpu_worker.Worker.compile_or_warm_up_model = (
        lambda _worker: gpu_worker.CompilationTimes(language_model=0.0, encoder=0.0)
    )

    # These settings match the 768x768 cap used by LLaMA-Factory at training.
    return LLM(
        model=str(MODEL_PATH.resolve()),
        dtype="bfloat16",
        seed=SEED,
        tensor_parallel_size=1,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=4,
        max_num_batched_tokens=4096,
        gpu_memory_utilization=0.90,
        load_format="instanttensor",
        allowed_local_media_path=str(FRAME_DIR.resolve()),
        limit_mm_per_prompt={"image": 1},
        mm_processor_kwargs={
            "min_pixels": IMAGE_MIN_PIXELS,
            "max_pixels": IMAGE_MAX_PIXELS,
        },
        mm_processor_cache_gb=0,
        skip_mm_profiling=True,
        trust_remote_code=True,
        enforce_eager=True,
        gdn_prefill_backend="triton",
        disable_log_stats=True,
    )


def sampling_params():
    """Return deterministic decoding parameters."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        seed=SEED,
    )


def generate_batch(engine, conversations: list[list[dict[str, Any]]]) -> list[str]:
    """Generate a batch in request order."""
    outputs = engine.chat(
        conversations,
        sampling_params=sampling_params(),
        use_tqdm=False,
        chat_template_kwargs={"enable_thinking": False},
    )
    if len(outputs) != len(conversations):
        raise RuntimeError(
            f"vLLM returned {len(outputs)} outputs for {len(conversations)} requests"
        )
    return [clean_answer(output.outputs[0].text) for output in outputs]


def run() -> int:
    started = time.monotonic()
    log.info("ORena FRAME inference starting")
    log.info("Container variant: camma-qwen38-frame-native-fp8")

    requests = load_requests(INPUT_PATH / "request.json")
    fo_definitions = load_definitions(INPUT_PATH / "FO_definitions.json")
    log.info("Loaded %d request(s)", len(requests))

    answers = [""] * len(requests)
    latencies = [0.0] * len(requests)
    valid_indices: list[int] = []
    conversations: list[list[dict[str, Any]]] = []

    for index, req in enumerate(requests):
        try:
            conversations.append(make_conversation(req, fo_definitions))
            valid_indices.append(index)
        except Exception:
            log.exception("qID=%s has invalid input; emitting an empty answer", req.qID)

    if conversations:
        log.info("Loading Qwen3.8 FP8 model once for %d valid request(s)", len(conversations))
        engine = load_engine()
        log.info("Model ready after %.2f s", time.monotonic() - started)
        inference_started = time.monotonic()
        try:
            batch_answers = generate_batch(engine, conversations)
            elapsed_per_item = (time.monotonic() - inference_started) / len(conversations)
            for index, answer in zip(valid_indices, batch_answers, strict=True):
                answers[index] = answer
                latencies[index] = elapsed_per_item
        except Exception:
            # A malformed image must not discard all otherwise valid questions.
            log.exception("Batch generation failed; retrying requests independently")
            for index, conversation in zip(valid_indices, conversations, strict=True):
                item_started = time.monotonic()
                try:
                    answers[index] = generate_batch(engine, [conversation])[0]
                except Exception:
                    log.exception(
                        "qID=%s generation failed; emitting an empty answer",
                        requests[index].qID,
                    )
                latencies[index] = time.monotonic() - item_started

    responses = [
        Response(qID=req.qID, content=answers[index], latency=latencies[index])
        for index, req in enumerate(requests)
    ]
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    save_responses(responses, OUTPUT_PATH / "answer.json")

    failed = sum(not answer for answer in answers)
    log.info(
        "Wrote %d response(s), %d empty, in %.2f s",
        len(responses),
        failed,
        time.monotonic() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
