from functools import cache

import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase, VitsModel

from ttsbench.core import Audio, Voice, to_audio, torch_device


def _repo(voice_id: str) -> str:
    return f"facebook/mms-tts-{voice_id}"


class Mms:
    @classmethod
    def download(cls, voices: tuple[Voice, ...]) -> None:
        for voice in voices:
            snapshot_download(_repo(voice.id))

    @staticmethod
    @cache
    def _model(voice_id: str) -> tuple[VitsModel, PreTrainedTokenizerBase]:
        model = VitsModel.from_pretrained(_repo(voice_id)).to(torch_device()).eval()
        return model, AutoTokenizer.from_pretrained(_repo(voice_id))

    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio:
        model, tokenizer = self._model(voice.id)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        model.speaking_rate = speed
        with torch.inference_mode():
            waveform = model(**inputs).waveform[0]
        return to_audio(waveform.cpu().numpy(), model.config.sampling_rate)
