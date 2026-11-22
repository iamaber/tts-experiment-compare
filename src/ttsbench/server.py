import argparse
import io
import json
import threading
import time
from collections import defaultdict
from pathlib import Path

import psutil
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ttsbench.benchmark import RESULTS_DIR, load_texts
from ttsbench.core import LANGUAGE_NAMES, Engine
from ttsbench.registry import MODELS, ModelSpec

WEB_DIR = Path(__file__).parent / "web"

app = FastAPI(title="ttsbench")
_engines: dict[str, Engine] = {}
_locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
_warm_voices: set[tuple[str, str]] = set()


class SynthesisRequest(BaseModel):
    model: str
    voice: str
    text: str = Field(min_length=1, max_length=2000)
    speed: float = Field(1.0, ge=0.5, le=2.0)


def _describe(spec: ModelSpec) -> dict:
    return {
        "id": spec.id,
        "name": spec.name,
        "description": spec.description,
        "languages": list(spec.languages),
        "voices": [
            {"id": v.id, "name": v.name, "language": v.language} for v in spec.voices
        ],
        "online": spec.online,
        "speed_control": spec.speed_control,
        "license": spec.license,
        "available": spec.available,
        "install_hint": f"uv sync --extra {spec.extra}",
    }


@app.get("/api/models")
def list_models() -> list[dict]:
    return [_describe(spec) for spec in MODELS.values()]


@app.get("/api/languages")
def list_languages() -> dict:
    texts = load_texts()
    return {
        code: {"name": name, "sample": texts[code][0]}
        for code, name in LANGUAGE_NAMES.items()
    }


@app.post("/api/synthesize")
def synthesize(request: SynthesisRequest) -> Response:
    spec = MODELS.get(request.model)
    if spec is None:
        raise HTTPException(404, f"unknown model {request.model!r}")
    if not spec.available:
        raise HTTPException(
            400, f"{spec.name} is not installed. Run: uv sync --extra {spec.extra}"
        )
    try:
        voice = spec.voice(request.voice)
    except KeyError as error:
        raise HTTPException(400, str(error.args[0])) from error

    with _locks[spec.id]:
        if spec.id not in _engines:
            _engines[spec.id] = spec.load()
        cold = (spec.id, voice.id) not in _warm_voices
        start = time.perf_counter()
        try:
            audio = _engines[spec.id].synthesize(
                request.text.strip(), voice, request.speed
            )
        except Exception as error:
            raise HTTPException(500, f"{spec.name} failed: {error}") from error
        latency_s = time.perf_counter() - start
        _warm_voices.add((spec.id, voice.id))

    buffer = io.BytesIO()
    sf.write(buffer, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
    return Response(
        buffer.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Latency-Seconds": f"{latency_s:.4f}",
            "X-Cold": "1" if cold else "0",
            "X-Audio-Seconds": f"{audio.duration:.4f}",
            "X-RTF": f"{latency_s / audio.duration:.4f}",
            "X-Sample-Rate": str(audio.sample_rate),
        },
    )


@app.get("/api/benchmark")
def benchmark_results() -> list[dict]:
    path = RESULTS_DIR / "results.json"
    return json.loads(path.read_text()) if path.exists() else []


@app.get("/api/system")
def system_info() -> dict:
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_mb": (memory.total - memory.available) / 1e6,
        "ram_total_mb": memory.total / 1e6,
        "process_rss_mb": psutil.Process().memory_info().rss / 1e6,
        "loaded_models": list(_engines),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the ttsbench web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
