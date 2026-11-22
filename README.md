# TTS experiments

`ttsbench` runs several open text-to-speech models behind one web UI and one benchmark, so you can hear and measure them side by side. It covers English and Bangla. The older one-off experiments live in [`experiments/`](experiments/).

## Models

| Model | Languages | Voices | Runs on | Speed control | License |
|---|---|---|---|---|---|
| Piper | English, Bangla | Lessac, Amy, Alan, Google bn_BD | CPU (ONNX) | yes | MIT code, per-voice data licenses |
| Kokoro 82M | English | 5 US and UK voices | CPU (ONNX) | yes | Apache-2.0 |
| MMS-TTS | English, Bangla | one per language | CPU or CUDA (PyTorch) | yes | CC-BY-NC-4.0 |
| NeMo FastPitch + HiFi-GAN | English | LJSpeech | CPU or CUDA (PyTorch) | yes | CC-BY-4.0 |
| NeMo VITS | English | LJSpeech | CPU or CUDA (PyTorch) | no | CC-BY-4.0 |
| Edge TTS | English, Bangla | 3 English, 3 Bangla | Microsoft cloud | yes | Microsoft service terms |

MMS-TTS is licensed for non-commercial use only. Edge TTS sends your text to Microsoft and needs an internet connection.

## Get started

You need [uv](https://docs.astral.sh/uv/) and about 8 GB of disk: 2 GB for the environment, 2.3 GB for the TTS weights, and 2.9 GB for the two ASR models the benchmark uses. A CUDA GPU is used when present, otherwise everything runs on the CPU. Set `TTSBENCH_DEVICE` to override the device.

1. Install every engine. To install only some, pass `--extra piper`, `--extra kokoro`, `--extra mms`, `--extra edge`, `--extra nemo`, and `--extra asr` (the benchmark's accuracy check) instead.

   ```bash
   uv sync --all-extras
   ```

2. Download all model weights ahead of time. Without this step each model downloads on first use. Running it again skips files already present.

   ```bash
   uv run ttsbench-download
   ```

   Pass `--models piper kokoro` to download a subset, or `--skip-asr` to skip the ASR models. Weights go to the Hugging Face cache, the NeMo cache, and `models/` (Kokoro, or `TTSBENCH_MODELS` if set).

3. Start the web UI and open http://127.0.0.1:8000.

   ```bash
   uv run ttsbench-serve
   ```

   The **Try a model** tab synthesizes one clip and shows latency, audio length, and real-time factor. The **Compare** tab runs the same text through every model you tick and lists the results in one table. The **Benchmark** tab shows the latest benchmark results.

## Run the benchmark

```bash
uv run ttsbench-bench
```

For each installed model and language, the benchmark synthesizes the sentences in [`benchmarks/texts.json`](benchmarks/texts.json) three times each and records:

- first call: time to load the model and synthesize the first sentence
- median latency per sentence
- real-time factor (RTF): total synthesis time divided by total audio length, so 0.1 means ten times faster than real time
- peak RAM of the process
- ASR error: an ASR model transcribes each clip, then the benchmark computes word error rate (WER) for English and character error rate (CER) for Bangla against the input text

English clips go to [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). Bangla clips go to [hishab/titu_stt_bn_conformer_large](https://huggingface.co/hishab/titu_stt_bn_conformer_large), which has the lowest public-model WER on FLEURS Bengali in the [Bengali ASR benchmark](https://github.com/ehzawad/bengali-asr-benchmark). Whisper large-v3 turbo was tried first and dropped. It gave 17% CER on Edge TTS Bangla, a near-human voice, against 1% for the Conformer, so it hid the differences between TTS models. The Conformer's training data includes OpenSLR 37, the dataset behind the Piper Bangla voice, which may flatter Piper's Bangla score.

Each model runs in its own subprocess, so load time and memory are not shared. Results go to [`benchmarks/results/`](benchmarks/results/) as `results.json` and `results.md`, with every transcript in the JSON. The generated audio goes to `benchmarks/audio/`.

Useful flags: `--models kokoro piper`, `--languages bn`, `--runs 5`, and `--no-asr`.

## Results on an Apple M5 CPU

One `uv run ttsbench-bench --runs 3` run on an Apple M5 with 24 GB of RAM, CPU only. Each row is 15 syntheses: 5 sentences, 3 runs each. Models ran one after another, not interleaved, with each runtime's default thread settings.

| Model | Language | Voice | First call (s) | Median latency (s) | RTF | Peak RAM (MB) | ASR error |
|---|---|---|---|---|---|---|---|
| Piper | English | en_US-lessac-medium | 0.90 | 0.069 | 0.017 | 602 | 1.5% WER |
| Piper | Bangla | bn_BD-google-medium | 0.74 | 0.048 | 0.017 | 602 | 2.8% CER |
| Kokoro 82M | English | af_heart | 1.53 | 0.856 | 0.200 | 747 | 0.0% WER |
| MMS-TTS | English | eng | 4.12 | 0.409 | 0.080 | 1959 | 6.3% WER |
| MMS-TTS | Bangla | ben | 2.00 | 0.306 | 0.082 | 1959 | 6.3% CER |
| NeMo FastPitch + HiFi-GAN | English | ljspeech | 5.94 | 0.500 | 0.104 | 2739 | 3.3% WER |
| NeMo VITS | English | ljspeech | 4.30 | 0.553 | 0.108 | 3844 | 4.9% WER |
| Edge TTS | English | en-US-AriaNeural | 1.28 | 1.131 | 0.231 | 59 | 0.0% WER |
| Edge TTS | Bangla | bn-BD-NabanitaNeural | 1.02 | 1.065 | 0.307 | 59 | 0.9% CER |

What the numbers say:

- Piper is the fastest model by a wide margin, about 60 times faster than real time in both languages, with the least memory of the local models. Its slowest English synthesis took 91 ms against Kokoro's fastest at 721 ms.
- Kokoro and Edge TTS had no English word errors. Kokoro is the most accurate English model that runs offline.
- For Bangla, Edge TTS is the most accurate (0.9% CER) but sends text to Microsoft. Piper is the best offline Bangla model here, with two caveats: it swallowed "দয়া" at the start of one sentence, and the Bangla ASR model was partly trained on Piper's source dataset. MMS-TTS made more real word errors, such as "বইমেলায়" heard as "বীণএ রায়".
- Error rates under about 5% are mostly spelling variants rather than mispronunciations: "traveling" heard as "travelling", "riverbank" as "river bank", and "ভালোবাসি" as "ভালবাসি". Treat those models as tied and compare them by ear in the UI. Every transcript is in `results.json`.
- Edge TTS latency is a network round trip to Microsoft, so it depends on your connection, not your hardware.
- ASR error measures intelligibility, not naturalness. A clear but robotic voice scores well.

## Add a model

1. Write an engine class in `src/ttsbench/engines/`. It needs a `download(voices)` classmethod that fetches the weights and a `synthesize(text, voice, speed)` method that returns `ttsbench.core.Audio`. Load weights lazily so the server starts fast.
2. Add a `ModelSpec` to `MODELS` in [`src/ttsbench/registry.py`](src/ttsbench/registry.py) with its voices, the import names it requires, and its `pyproject.toml` extra.
3. Add its dependencies as a new extra in `pyproject.toml` and to the `all` extra.

The web UI, the benchmark, and the download script read the registry, so they pick up the new model without changes.

## Repository layout

```text
src/ttsbench/
  core.py          Voice, Audio, and Engine types
  registry.py      every model, its voices, and its dependencies
  engines/         one module per model family
  server.py        FastAPI app behind ttsbench-serve
  web/             static frontend
  benchmark.py     ttsbench-bench
  download.py      ttsbench-download
  asr.py           ASR round trip and WER/CER
benchmarks/
  texts.json       English and Bangla test sentences
  results/         latest benchmark results
experiments/       earlier standalone projects, each with its own environment
```

## Experiments

These projects predate `ttsbench`. Each has its own `pyproject.toml` and README.

- [`experiments/fastpitch-hifigan-finetune`](experiments/fastpitch-hifigan-finetune/) tries voice cloning with a multi-speaker FastPitch model and TitaNet speaker embeddings.
- [`experiments/voice-swap-NeMo`](experiments/voice-swap-NeMo/) fine-tunes FastPitch and HiFi-GAN on 5 minutes of a new speaker and has an ASR-to-TTS voice swap notebook.
- [`experiments/tts-test-kayutai`](experiments/tts-test-kayutai/) is a Kyutai TTS notebook with voice conditioning. It needs a CUDA GPU, so it is not part of `ttsbench`.
