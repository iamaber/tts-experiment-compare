import urllib.request
from functools import cached_property
from pathlib import Path

import onnxruntime
from kokoro_onnx import Kokoro as KokoroOnnx

from ttsbench.core import MODELS_DIR, Audio, Voice, onnx_providers, to_audio

RELEASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")
LANG_BY_PREFIX = {"a": "en-us", "b": "en-gb"}


def _fetch(name: str) -> Path:
    path = MODELS_DIR / "kokoro" / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE}/{name}", partial)
        partial.rename(path)
    return path


class Kokoro:
    @classmethod
    def download(cls, voices: tuple[Voice, ...]) -> None:
        for name in FILES:
            _fetch(name)

    @cached_property
    def model(self) -> KokoroOnnx:
        session = onnxruntime.InferenceSession(
            str(_fetch(FILES[0])), providers=onnx_providers()
        )
        return KokoroOnnx.from_session(session, str(_fetch(FILES[1])))

    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio:
        samples, sample_rate = self.model.create(
            text, voice=voice.id, speed=speed, lang=LANG_BY_PREFIX[voice.id[0]]
        )
        return to_audio(samples, sample_rate)
