import time
from pathlib import Path

import soundfile as sf
import torch
import ml_dtypes

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

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class SimpleFastPitchHifiGanE2E:
	def __init__(self, device: str):
		self.device = device
		self.spec_generator = nemo_tts.models.FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch").to(device)
		self.vocoder = nemo_tts.models.HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(device)
		self.sample_rate = 22050

	def parse(self, text: str):
		return self.spec_generator.parse(text)

	@torch.no_grad()
	def convert_text_to_waveform(self, tokens):
		mels = self.spec_generator.generate_spectrogram(tokens=tokens)
		return self.vocoder.convert_spectrogram_to_audio(spec=mels)


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
	print(f"Running FastPitch+HiFi-GAN (E2E) on {device}")

	if FastPitchHifiGanE2EModel is not None:
		model = FastPitchHifiGanE2EModel.from_pretrained(model_name="tts_en_e2e_fastpitchhifigan").to(device)
		sr = 22050
	else:
		print("FastPitchHifiGanE2EModel not found; using simple wrapper fallback.")
		model = SimpleFastPitchHifiGanE2E(device)
		sr = model.sample_rate

	def infer():
		tokens = model.parse("Hey, I can speak!")
		with torch.no_grad():
			audio = model.convert_text_to_waveform(tokens=tokens)
		return audio.squeeze(0)

	audio, latency = _time_call("FastPitch+HiFi-GAN E2E", infer)
	out = OUTPUT_DIR / "speech.wav"
	_save(out, audio, sr)

	duration = audio.shape[-1] / float(sr)
	rtf = latency / duration if duration > 0 else float("nan")
	print(f"Audio duration: {duration:.3f}s | RTF: {rtf:.3f}")


if __name__ == "__main__":
	main()
