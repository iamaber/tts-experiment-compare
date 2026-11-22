import importlib.util
import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

Language = Literal["en", "bn"]
Metric = Literal["WER", "CER"]
LANGUAGE_NAMES: dict[Language, str] = {"en": "English", "bn": "Bangla"}
METRIC: dict[Language, Metric] = {"en": "WER", "bn": "CER"}

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
    array = np.asarray(samples, dtype=np.float32).squeeze()
    return Audio(np.clip(array, -1.0, 1.0), sample_rate)


def installed(*modules: str) -> bool:
    return all(importlib.util.find_spec(module) for module in modules)


ASR_REQUIRES = ("jiwer", "nemo")


def torch_device() -> str:
    import torch

    if requested := os.environ.get("TTSBENCH_DEVICE"):
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


@cache
def onnx_providers() -> list[str]:
    import onnxruntime

    if os.environ.get("TTSBENCH_DEVICE") == "cpu":
        return ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" not in onnxruntime.get_available_providers():
        return ["CPUExecutionProvider"]
    onnxruntime.preload_dlls()
    return ["CUDAExecutionProvider", "CPUExecutionProvider"]
