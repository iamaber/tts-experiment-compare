import time
from pathlib import Path

import soundfile as sf
import torch
import nemo.collections.tts as nemo_tts

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _save(path: Path, audio: torch.Tensor, sr: int):
    wave = audio.detach().to("cpu").float().squeeze().numpy()
    sf.write(path, wave, sr)


def _time_call(label: str, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"{label} latency: {dt:.3f}s")
    return out, dt


def main():
    device = _device()
    print(f"Running FastPitch + HiFi-GAN on {device}")

    spec_generator = nemo_tts.models.FastPitchModel.from_pretrained(
        "nvidia/tts_en_fastpitch"
    ).to(device)
    vocoder = nemo_tts.models.HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(
        device
    )

    text = "Hello world, this is NVIDIA NeMo FastPitch with HiFi-GAN."

    def infer():
        tokens = spec_generator.parse(text)
        with torch.no_grad():
            mels = spec_generator.generate_spectrogram(tokens=tokens)
            audio = vocoder.convert_spectrogram_to_audio(spec=mels)
        return audio.squeeze(0)

    audio, latency = _time_call("FastPitch+HiFi-GAN", infer)
    sr = 22050
    _save(OUTPUT_DIR / "fastpitch_hifigan.wav", audio, sr)

    duration = audio.shape[-1] / float(sr)
    rtf = latency / duration if duration > 0 else float("nan")
    print(f"Audio duration: {duration:.3f}s | RTF: {rtf:.3f}")
    print("Done synthesis!")


if __name__ == "__main__":
    main()
