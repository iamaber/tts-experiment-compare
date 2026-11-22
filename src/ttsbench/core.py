import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

Language = Literal["en", "bn"]
LANGUAGE_NAMES: dict[Language, str] = {"en": "English", "bn": "Bangla"}

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = Path(os.environ.get("TTSBENCH_MODELS", REPO_ROOT / "models"))


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    language: Language


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray
    sample_rate: int

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


class Engine(Protocol):
    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio: ...


def to_audio(samples, sample_rate: int) -> Audio:
    """Normalize engine output to mono float32 in [-1, 1]."""
    array = np.asarray(samples, dtype=np.float32).squeeze()
    if np.issubdtype(np.asarray(samples).dtype, np.integer):
        array = array / 32768.0
    return Audio(np.clip(array, -1.0, 1.0), sample_rate)


def torch_device() -> str:
    import torch

    if requested := os.environ.get("TTSBENCH_DEVICE"):
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"
