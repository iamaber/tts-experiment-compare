import time
from pathlib import Path
from typing import Optional

import torch
import psutil
import ml_dtypes
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ONNX/ml_dtypes compatibility aliases
if not hasattr(ml_dtypes, "float4_e2m1fn"):
    ml_dtypes.float4_e2m1fn = ml_dtypes.float8_e4m3fn
if not hasattr(ml_dtypes, "float4_e2m1fnuz"):
    ml_dtypes.float4_e2m1fnuz = ml_dtypes.float8_e4m3fn
if not hasattr(ml_dtypes, "float8_e8m0fnu"):
    ml_dtypes.float8_e8m0fnu = ml_dtypes.float8_e4m3fn
if not hasattr(ml_dtypes, "float8_e8m0fnuz"):
    ml_dtypes.float8_e8m0fnuz = ml_dtypes.float8_e4m3fn

import nemo.collections.tts as nemo_tts

try:
    from nemo.collections.tts.models import FastPitchHifiGanE2EModel
except ImportError:
    FastPitchHifiGanE2EModel = None

import soundfile as sf


app = FastAPI(title="NeMo TTS Benchmark")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


class SimpleFastPitchHifiGanE2E:
    """Fallback E2E wrapper when NeMo's official model is unavailable."""

    def __init__(self, device: str):
        self.device = device
        self.spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
            "nvidia/tts_en_fastpitch"
        ).to(device)
        self.vocoder = nemo_tts.models.HifiGanModel.from_pretrained(
            "nvidia/tts_hifigan"
        ).to(device)
        self.sample_rate = 22050

    def parse(self, text: str):
        return self.spec_generator.parse(text)

    @torch.no_grad()
    def convert_text_to_waveform(self, tokens):
        mels = self.spec_generator.generate_spectrogram(tokens=tokens)
        return self.vocoder.convert_spectrogram_to_audio(spec=mels)


# Model cache
models_cache = {}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_gpu_memory():
    """Get GPU memory usage in MB if CUDA is available."""
    if torch.cuda.is_available():
        return {
            "allocated_mb": torch.cuda.memory_allocated() / 1024**2,
            "reserved_mb": torch.cuda.memory_reserved() / 1024**2,
        }
    return None


class ResourceMonitor:
    """Monitor CPU/memory/GPU during synthesis."""

    def __init__(self):
        self.process = psutil.Process()
        self.start_mem = None
        self.start_gpu = None
        self.peak_mem = None
        self.peak_gpu = None
        self.cpu_samples = []

    def start(self):
        # Clear GPU cache to get accurate measurements
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        self.start_mem = self.process.memory_info().rss / 1024**2
        self.peak_mem = self.start_mem
        self.start_gpu = get_gpu_memory()
        self.peak_gpu = self.start_gpu.copy() if self.start_gpu else None

        # Sample CPU percent (needs ~100ms interval to be accurate)
        self.process.cpu_percent()

    def end(self):
        # Sample CPU after a small delay for more accurate measurement
        end_cpu = self.process.cpu_percent(interval=0.1)

        # Memory at end of synthesis
        end_mem = self.process.memory_info().rss / 1024**2
        self.peak_mem = max(self.peak_mem, end_mem)

        # GPU memory - torch tracks peak usage
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

        # Get PyTorch's tracked peak if available
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
    """Load and cache TTS models."""
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

    elif model_name == "fastpitch_hifigan_e2e":
        if FastPitchHifiGanE2EModel is not None:
            model = FastPitchHifiGanE2EModel.from_pretrained(
                "tts_en_e2e_fastpitchhifigan"
            ).to(device)
            sr = 22050
        else:
            model = SimpleFastPitchHifiGanE2E(device)
            sr = model.sample_rate
        models_cache[model_name] = {"model": model, "sr": sr}
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
            {"id": "fastpitch_hifigan_e2e", "name": "FastPitch + HiFi-GAN (E2E)"},
            {"id": "vits", "name": "VITS"},
        ]
    }


@app.post("/api/synthesize", response_model=SynthesisResponse)
async def synthesize(request: SynthesisRequest):
    """Synthesize audio from text using selected model."""
    try:
        device = _device()
        monitor = ResourceMonitor()

        # Load model
        model_data = load_model(request.model, device)

        # Start monitoring
        monitor.start()
        start_time = time.perf_counter()

        # Synthesize based on model type
        if request.model == "fastpitch_hifigan":
            tokens = model_data["spec_gen"].parse(request.text)
            with torch.no_grad():
                mels = model_data["spec_gen"].generate_spectrogram(tokens=tokens)
                audio = model_data["vocoder"].convert_spectrogram_to_audio(spec=mels)
            audio = audio.squeeze(0)
            sr = model_data["sr"]

        elif request.model == "fastpitch_hifigan_e2e":
            tokens = model_data["model"].parse(request.text)
            with torch.no_grad():
                audio = model_data["model"].convert_text_to_waveform(tokens=tokens)
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

        # Save audio
        filename = f"{request.model}_{int(time.time())}.wav"
        filepath = OUTPUT_DIR / filename
        wave = audio.detach().to("cpu").float().squeeze().numpy()
        sf.write(filepath, wave, sr)

        # Calculate metrics
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
