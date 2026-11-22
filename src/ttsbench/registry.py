import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

from ttsbench.core import Engine, Language, Voice


@dataclass(frozen=True)
class ModelSpec:
    id: str
    name: str
    description: str
    engine: str
    extra: str
    requires: tuple[str, ...]
    voices: tuple[Voice, ...]
    license: str
    online: bool = False
    speed_control: bool = True

    @property
    def languages(self) -> tuple[Language, ...]:
        return tuple(dict.fromkeys(v.language for v in self.voices))

    @property
    def available(self) -> bool:
        return all(importlib.util.find_spec(name) for name in self.requires)

    def voice(self, voice_id: str) -> Voice:
        for voice in self.voices:
            if voice.id == voice_id:
                return voice
        raise KeyError(f"{self.id} has no voice {voice_id!r}")

    def engine_class(self) -> Any:
        module, cls = self.engine.split(":")
        return getattr(importlib.import_module(module), cls)

    def load(self) -> Engine:
        return self.engine_class()()

    def download(self) -> None:
        self.engine_class().download(self.voices)


def _voices(language: Language, *pairs: tuple[str, str]) -> tuple[Voice, ...]:
    return tuple(Voice(id, name, language) for id, name in pairs)


MODELS: dict[str, ModelSpec] = {
    spec.id: spec
    for spec in (
        ModelSpec(
            id="piper",
            name="Piper",
            description="VITS voices exported to ONNX. Runs faster than real time on a CPU.",
            engine="ttsbench.engines.piper:Piper",
            extra="piper",
            requires=("piper",),
            voices=_voices(
                "en",
                ("en_US-lessac-medium", "Lessac (US)"),
                ("en_US-amy-medium", "Amy (US)"),
                ("en_GB-alan-medium", "Alan (UK)"),
            )
            + _voices("bn", ("bn_BD-google-medium", "Google (Bangladesh)")),
            license="MIT code; per-voice dataset licenses",
        ),
        ModelSpec(
            id="kokoro",
            name="Kokoro 82M",
            description="82M-parameter StyleTTS2-based model with ONNX runtime. High English quality for its size.",
            engine="ttsbench.engines.kokoro:Kokoro",
            extra="kokoro",
            requires=("kokoro_onnx",),
            voices=_voices(
                "en",
                ("af_heart", "Heart (US, female)"),
                ("af_bella", "Bella (US, female)"),
                ("am_michael", "Michael (US, male)"),
                ("bf_emma", "Emma (UK, female)"),
                ("bm_george", "George (UK, male)"),
            ),
            license="Apache-2.0",
        ),
        ModelSpec(
            id="mms",
            name="MMS-TTS",
            description="Meta Massively Multilingual Speech VITS models, one checkpoint per language.",
            engine="ttsbench.engines.mms:Mms",
            extra="mms",
            requires=("transformers", "torch"),
            voices=_voices("en", ("eng", "English"))
            + _voices("bn", ("ben", "Bengali")),
            license="CC-BY-NC-4.0",
        ),
        ModelSpec(
            id="nemo-fastpitch",
            name="NeMo FastPitch + HiFi-GAN",
            description="Non-autoregressive FastPitch spectrogram model with a HiFi-GAN vocoder, trained on LJSpeech.",
            engine="ttsbench.engines.nemo:FastPitchHifiGan",
            extra="nemo",
            requires=("nemo",),
            voices=_voices("en", ("ljspeech", "LJSpeech (US, female)")),
            license="CC-BY-4.0",
        ),
        ModelSpec(
            id="nemo-vits",
            name="NeMo VITS",
            description="End-to-end VITS model trained on LJSpeech.",
            engine="ttsbench.engines.nemo:Vits",
            extra="nemo",
            requires=("nemo",),
            voices=_voices("en", ("ljspeech", "LJSpeech (US, female)")),
            license="CC-BY-4.0",
            speed_control=False,
        ),
        ModelSpec(
            id="edge",
            name="Edge TTS",
            description="Microsoft Edge neural voices. Cloud service, needs an internet connection.",
            engine="ttsbench.engines.edge:Edge",
            extra="edge",
            requires=("edge_tts",),
            voices=_voices(
                "en",
                ("en-US-AriaNeural", "Aria (US, female)"),
                ("en-US-GuyNeural", "Guy (US, male)"),
                ("en-GB-SoniaNeural", "Sonia (UK, female)"),
            )
            + _voices(
                "bn",
                ("bn-BD-NabanitaNeural", "Nabanita (Bangladesh, female)"),
                ("bn-BD-PradeepNeural", "Pradeep (Bangladesh, male)"),
                ("bn-IN-TanishaaNeural", "Tanishaa (India, female)"),
            ),
            license="Microsoft service terms",
            online=True,
        ),
    )
}
