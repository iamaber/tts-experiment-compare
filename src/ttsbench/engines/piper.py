from functools import cache
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from piper import PiperVoice, SynthesisConfig

from ttsbench.core import Audio, Voice, to_audio

REPO = "rhasspy/piper-voices"


def _fetch(voice_id: str) -> Path:
    lang_code, name, quality = voice_id.split("-")
    folder = f"{lang_code.split('_')[0]}/{lang_code}/{name}/{quality}"
    hf_hub_download(REPO, f"{folder}/{voice_id}.onnx.json")
    return Path(hf_hub_download(REPO, f"{folder}/{voice_id}.onnx"))


class Piper:
    @classmethod
    def download(cls, voices: tuple[Voice, ...]) -> None:
        for voice in voices:
            _fetch(voice.id)

    @staticmethod
    @cache
    def _voice(voice_id: str) -> PiperVoice:
        return PiperVoice.load(_fetch(voice_id))

    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio:
        model = self._voice(voice.id)
        config = SynthesisConfig(length_scale=1.0 / speed)
        chunks = [chunk.audio_float_array for chunk in model.synthesize(text, config)]
        return to_audio(np.concatenate(chunks), model.config.sample_rate)
