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
    print(f"Running VITS + HiFi-GAN on {device}")

    model = nemo_tts.models.VitsModel.from_pretrained("tts_en_lj_vits").to(device)
    text = "This is VITS synthesizing audio end to end!"

    def infer():
        tokens = model.parse(text)
        with torch.no_grad():
            audio = model.convert_text_to_waveform(tokens=tokens)
        return audio.squeeze(0) if audio.dim() > 1 else audio

    audio, latency = _time_call("VITS+HiFi-GAN", infer)
    sr = int(getattr(model._cfg, "sample_rate", 22050))
    _save(OUTPUT_DIR / "vits_tts.wav", audio, sr)

    duration = audio.shape[-1] / float(sr)
    rtf = latency / duration if duration > 0 else float("nan")
    print(f"Audio duration: {duration:.3f}s | RTF: {rtf:.3f}")
    print("VITS synthesis completed!")


if __name__ == "__main__":
    main()
