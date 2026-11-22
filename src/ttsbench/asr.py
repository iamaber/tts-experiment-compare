import unicodedata
from collections.abc import Callable
from functools import cache

import jiwer
import numpy as np
from scipy.signal import resample_poly

from ttsbench.core import METRIC, Audio, Language, Metric, torch_device

ASR_MODELS: dict[Language, str] = {
    "en": "nvidia/parakeet-tdt-0.6b-v2",
    "bn": "hishab/titu_stt_bn_conformer_large",
}
ASR_RATE = 16_000


def cer_ignoring_spaces(reference: str, hypothesis: str) -> float:
    return jiwer.cer(reference.replace(" ", ""), hypothesis.replace(" ", ""))


SCORERS: dict[Metric, Callable[[str, str], float]] = {
    "WER": jiwer.wer,
    "CER": cer_ignoring_spaces,
}


def normalize(text: str) -> str:
    kept = (" " if unicodedata.category(c).startswith("P") else c for c in text.lower())
    return " ".join("".join(kept).split())


def error_rate(reference: str, hypothesis: str, language: Language) -> float:
    return SCORERS[METRIC[language]](normalize(reference), normalize(hypothesis))


@cache
def load_asr(language: Language):
    from nemo.collections.asr.models import ASRModel

    return ASRModel.from_pretrained(
        ASR_MODELS[language], map_location=torch_device()
    ).eval()


def transcribe(clips: list[Audio], language: Language) -> list[str]:
    arrays = [
        resample_poly(clip.samples, ASR_RATE, clip.sample_rate).astype(np.float32)
        for clip in clips
    ]
    outputs = load_asr(language).transcribe(arrays, batch_size=1, verbose=False)
    return [getattr(output, "text", output) for output in outputs]
