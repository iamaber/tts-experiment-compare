import asyncio
import io

import edge_tts
import soundfile as sf

from ttsbench.core import Audio, Voice, to_audio


class Edge:
    @classmethod
    def download(cls) -> None:
        pass

    def synthesize(self, text: str, voice: Voice, speed: float) -> Audio:
        rate = f"{round((speed - 1) * 100):+d}%"
        mp3 = asyncio.run(self._fetch(text, voice.id, rate))
        samples, sample_rate = sf.read(io.BytesIO(mp3), dtype="float32")
        return to_audio(samples, sample_rate)

    @staticmethod
    async def _fetch(text: str, voice: str, rate: str) -> bytes:
        chunks = []
        async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])
        return b"".join(chunks)
