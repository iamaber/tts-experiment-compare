import time
from pathlib import Path
from typing import Optional

import torch
import psutil
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import nemo.collections.tts as nemo_tts
import soundfile as sf


app = FastAPI(title="NeMo TTS Benchmark")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


models_cache = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }
    return None


class ResourceMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.start_mem = None
        self.start_gpu = None
        self.peak_mem = None
        self.peak_gpu = None
        self.cpu_samples = []

    def start(self):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        self.start_mem = self.process.memory_info().rss / 1024**2
        self.peak_mem = self.start_mem
        self.start_gpu = get_gpu_memory()
        self.peak_gpu = self.start_gpu.copy() if self.start_gpu else None

        self.process.cpu_percent()

    def end(self):
        end_cpu = self.process.cpu_percent()

        end_mem = self.process.memory_info().rss / 1024**2
        self.peak_mem = max(self.peak_mem, end_mem)

        end_gpu = get_gpu_memory()
        if end_gpu:
            self.peak_gpu = {
                "allocated_mb": max(
                    self.peak_gpu.get("allocated_mb", 0), end_gpu.get("allocated_mb", 0)
                ),
                "reserved_mb": max(
                    self.peak_gpu.get("reserved_mb", 0), end_gpu.get("reserved_mb", 0)
                ),
            }

        peak_pytorch_gpu = None
        if torch.cuda.is_available():
            peak_pytorch_gpu = torch.cuda.max_memory_allocated() / 1024**2

        return {
            "cpu_percent": round(end_cpu, 2),
            "memory_mb": round(end_mem, 2),
            "memory_delta_mb": round(end_mem - self.start_mem, 2),
            "peak_memory_mb": round(self.peak_mem, 2),
            "gpu_allocated_mb": round(self.peak_gpu.get("allocated_mb", 0), 2)
            if self.peak_gpu
            else None,
            "gpu_reserved_mb": round(self.peak_gpu.get("reserved_mb", 0), 2)
            if self.peak_gpu
            else None,
            "gpu_peak_pytorch_mb": round(peak_pytorch_gpu, 2)
            if peak_pytorch_gpu
            else None,
        }


class SynthesisRequest(BaseModel):
    model: str
    text: str


class SynthesisResponse(BaseModel):
    success: bool
    audio_url: Optional[str] = None
    latency_s: Optional[float] = None
    duration_s: Optional[float] = None
    rtf: Optional[float] = None
    sample_rate: Optional[int] = None
    resources: Optional[dict] = None
    error: Optional[str] = None


def load_model(model_name: str, device: str):
    if model_name in models_cache:
        return models_cache[model_name]

    if model_name == "fastpitch_hifigan":
        spec_gen = nemo_tts.models.FastPitchModel.from_pretrained(
            "nvidia/tts_en_fastpitch"
        ).to(device)
        vocoder = nemo_tts.models.HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(
            device
        )
        models_cache[model_name] = {
            "spec_gen": spec_gen,
            "vocoder": vocoder,
            "sr": 22050,
        }
        return models_cache[model_name]

    elif model_name == "vits":
        model = nemo_tts.models.VitsModel.from_pretrained("tts_en_lj_vits").to(device)
        sr = int(getattr(model._cfg, "sample_rate", 22050))
        models_cache[model_name] = {"model": model, "sr": sr}
        return models_cache[model_name]

    else:
        raise ValueError(f"Unknown model: {model_name}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    html_path = Path("static/index.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Create static/index.html</h1>")


@app.get("/api/models")
async def get_models():
    """Return available models."""
    return {
        "models": [
            {"id": "fastpitch_hifigan", "name": "FastPitch + HiFi-GAN"},
            {"id": "vits", "name": "VITS"},
        ]
    }


@app.post("/api/synthesize", response_model=SynthesisResponse)
async def synthesize(request: SynthesisRequest):
    """Synthesize audio from text using selected model."""
    try:
        device = _device()
        monitor = ResourceMonitor()

        model_data = load_model(request.model, device)

        monitor.start()
        start_time = time.perf_counter()

        if request.model == "fastpitch_hifigan":
            tokens = model_data["spec_gen"].parse(request.text)
            with torch.no_grad():
                mels = model_data["spec_gen"].generate_spectrogram(tokens=tokens)
                audio = model_data["vocoder"].convert_spectrogram_to_audio(spec=mels)
            audio = audio.squeeze(0)
            sr = model_data["sr"]

        elif request.model == "vits":
            tokens = model_data["model"].parse(request.text)
            with torch.no_grad():
                audio = model_data["model"].convert_text_to_waveform(tokens=tokens)
            audio = audio.squeeze(0) if audio.dim() > 1 else audio
            sr = model_data["sr"]

        latency = time.perf_counter() - start_time
        resources = monitor.end()

        filename = f"{request.model}_{int(time.time())}.wav"
        filepath = OUTPUT_DIR / filename
        wave = audio.detach().to("cpu").float().squeeze().numpy()
        sf.write(filepath, wave, sr)

        duration = audio.shape[-1] / float(sr)
        rtf = latency / duration if duration > 0 else 0

        return SynthesisResponse(
            success=True,
            audio_url=f"/outputs/{filename}",
            latency_s=round(latency, 3),
            duration_s=round(duration, 3),
            rtf=round(rtf, 3),
            sample_rate=sr,
            resources=resources,
        )

    except Exception as e:
        return SynthesisResponse(
            success=False,
            error=str(e),
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
