import time
from pathlib import Path

import soundfile as sf
import torch
import ml_dtypes

# ONNX expects float4 types; older ml_dtypes versions lack them, so alias to float8 to satisfy imports.
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


class SimpleFastPitchHifiGanE2E:
    """Lightweight E2E wrapper when NeMo's FastPitchHifiGanE2EModel is unavailable."""

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


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _device() -> str:
    """Pick GPU when available so timing reflects accelerator speed."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _time_call(label: str, func):
    start = time.perf_counter()
    result = func()
    latency = time.perf_counter() - start
    print(f"{label} latency: {latency:.3f}s")
    return result, latency


def _summarize_perf(
    name: str, latency_s: float, audio_tensor: torch.Tensor, sample_rate: int
):
    num_samples = audio_tensor.shape[-1]
    duration_s = num_samples / float(sample_rate)
    rtf = latency_s / duration_s if duration_s > 0 else float("nan")
    print(f"{name} audio duration: {duration_s:.3f}s | RTF: {rtf:.3f}")
    return {"model": name, "latency_s": latency_s, "duration_s": duration_s, "rtf": rtf}


def _save_wav(audio_tensor: torch.Tensor, sample_rate: int, filename: str):
    path = OUTPUT_DIR / filename
    wave = audio_tensor.detach().to("cpu").float().squeeze().numpy()
    sf.write(path, wave, sample_rate)
    print(f"Saved: {path}")


def run_fastpitch_hifigan(text: str):
    device = _device()
    print(f"Running FastPitch + HiFi-GAN on {device}")

    spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
        "nvidia/tts_en_fastpitch"
    ).to(device)
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(
        device
    )

    def _infer():
        tokens = spec_generator.parse(text)
        with torch.no_grad():
            mels = spec_generator.generate_spectrogram(tokens=tokens)
            audio = vocoder.convert_spectrogram_to_audio(spec=mels)
        return audio.squeeze(0)

    audio, latency = _time_call("FastPitch+HiFi-GAN", _infer)
    sample_rate = 22050
    _save_wav(audio, sample_rate, "fastpitch_hifigan.wav")
    return _summarize_perf("FastPitch+HiFi-GAN", latency, audio, sample_rate)


def run_fastpitch_hifigan_e2e(text: str):
    device = _device()
    print(f"Running FastPitch+HiFi-GAN (E2E) on {device}")

    if FastPitchHifiGanE2EModel is not None:
        model = FastPitchHifiGanE2EModel.from_pretrained(
            model_name="tts_en_e2e_fastpitchhifigan"
        ).to(device)
        sample_rate = 22050
    else:
        print("FastPitchHifiGanE2EModel not found; using simple wrapper fallback.")
        model = SimpleFastPitchHifiGanE2E(device)
        sample_rate = model.sample_rate

    def _infer():
        tokens = model.parse(text)
        with torch.no_grad():
            audio = model.convert_text_to_waveform(tokens=tokens)
        return audio.squeeze(0)

    audio, latency = _time_call("FastPitch+HiFi-GAN E2E", _infer)
    _save_wav(audio, sample_rate, "fastpitch_hifigan_e2e.wav")
    return _summarize_perf("FastPitch+HiFi-GAN E2E", latency, audio, sample_rate)


def run_vits_hifigan(text: str):
    device = _device()
    print(f"Running VITS + HiFi-GAN on {device}")

    model = nemo_tts.models.VitsModel.from_pretrained("tts_en_lj_vits").to(device)

    def _infer():
        tokens = model.parse(text)
        with torch.no_grad():
            audio = model.convert_text_to_waveform(tokens=tokens)
        return audio.squeeze(0) if audio.dim() > 1 else audio

    audio, latency = _time_call("VITS+HiFi-GAN", _infer)
    sample_rate = int(getattr(model._cfg, "sample_rate", 22050))
    _save_wav(audio, sample_rate, "vits_hifigan.wav")
    return _summarize_perf("VITS+HiFi-GAN", latency, audio, sample_rate)


def main():
    text = "Hello world, benchmarking NeMo TTS pipelines."
    results = []

    try:
        results.append(run_fastpitch_hifigan(text))
    except Exception as exc:
        print(f"FastPitch+HiFi-GAN failed: {exc}")

    try:
        results.append(run_fastpitch_hifigan_e2e(text))
    except Exception as exc:
        print(f"FastPitch+HiFi-GAN E2E failed: {exc}")

    try:
        results.append(run_vits_hifigan(text))
    except Exception as exc:
        print(f"VITS+HiFi-GAN failed: {exc}")

    if results:
        print("\nPerformance summary (latency | audio duration | real-time factor):")
        for row in results:
            print(
                f"- {row['model']}: {row['latency_s']:.3f}s | {row['duration_s']:.3f}s | {row['rtf']:.3f}"
            )


if __name__ == "__main__":
    main()
