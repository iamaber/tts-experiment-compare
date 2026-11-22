from functools import cached_property

import torch
from nemo.collections.tts.models import FastPitchModel, HifiGanModel, VitsModel

from ttsbench.core import Audio, Voice, to_audio, torch_device

SAMPLE_RATE = 22050
FASTPITCH = "nvidia/tts_en_fastpitch"
HIFIGAN = "nvidia/tts_hifigan"
VITS = "tts_en_lj_vits"


def _load(cls, name: str):
    return cls.from_pretrained(name, map_location=torch_device()).eval()


class FastPitchHifiGan:
    @classmethod
    def download(cls, voices: tuple[Voice, ...]) -> None:
        _load(FastPitchModel, FASTPITCH)
        _load(HifiGanModel, HIFIGAN)

    @cached_property
    def models(self) -> tuple[FastPitchModel, HifiGanModel]:
        return _load(FastPitchModel, FASTPITCH), _load(HifiGanModel, HIFIGAN)

    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio:
        fastpitch, vocoder = self.models
        with torch.inference_mode():
            spec = fastpitch.generate_spectrogram(
                tokens=fastpitch.parse(text), pace=speed
            )
            audio = vocoder.convert_spectrogram_to_audio(spec=spec)
        return to_audio(audio.cpu().numpy(), SAMPLE_RATE)


class Vits:
    @classmethod
    def download(cls, voices: tuple[Voice, ...]) -> None:
        _load(VitsModel, VITS)

    @cached_property
    def model(self) -> VitsModel:
        return _load(VitsModel, VITS)

    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio:
        with torch.inference_mode():
            audio = self.model.convert_text_to_waveform(tokens=self.model.parse(text))
        return to_audio(audio.cpu().numpy(), SAMPLE_RATE)
