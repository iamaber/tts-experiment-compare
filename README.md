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

MMS-TTS is licensed for non-commercial use only. Edge TTS sends your text to Microsoft and needs an internet connection. On macOS and ARM Linux, NeMo installs without its text normalizer, so the NeMo models read digits poorly. Write numbers as words for them.

## Get started

You need [uv](https://docs.astral.sh/uv/) and about 8 GB of disk. The environment takes 2 GB on macOS, the TTS weights 2.3 GB, and the two ASR models for the benchmark 2.9 GB. `ttsbench` uses a CUDA GPU when one is present and the CPU otherwise. Set `TTSBENCH_DEVICE` to override the device.

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

For each installed model and language, the benchmark synthesizes the sentences in [`benchmarks/texts.json`](benchmarks/texts.json) and records:

- first call: time to load the model and synthesize the first sentence
- median latency per sentence
- real-time factor (RTF): total synthesis time divided by total audio length, so 0.1 means ten times faster than real time
- peak RAM of the process
- ASR error: an ASR model transcribes each clip, then the benchmark computes word error rate (WER) for English and character error rate (CER) for Bangla against the input text

English clips go to [NVIDIA Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2). Bangla clips go to [hishab/titu_stt_bn_conformer_large](https://huggingface.co/hishab/titu_stt_bn_conformer_large), which has the lowest public-model WER on FLEURS Bengali in the [Bengali ASR benchmark](https://github.com/ehzawad/bengali-asr-benchmark). An earlier version used Whisper large-v3 turbo. Whisper gave 17% CER on Edge TTS Bangla, a near-human voice, against 1% for the Conformer, so it hid the differences between TTS models. The Conformer's training data includes OpenSLR 37, the dataset behind the Piper Bangla voice, which may flatter Piper's Bangla score.

The benchmark runs in rounds. Each round starts a fresh subprocess for every model in turn, so load time and memory are not shared, and drift or background load hits every model alike. First call is the median across rounds. Results go to [`benchmarks/results/`](benchmarks/results/) as `results.json` and `results.md`, with every transcript in the JSON. The generated audio goes to `benchmarks/audio/`.

Useful flags: `--models kokoro piper`, `--languages bn`, `--rounds 5`, `--runs 2` (timed runs per sentence per round), and `--no-asr`.

## Results on an Apple M5 CPU

One `uv run ttsbench-bench` run on an Apple M5 with 24 GB of RAM, CPU only, with each runtime's default thread settings. Each row is 15 syntheses: 3 rounds of 5 sentences, with the models alternating between rounds. The ASR error comes from the last round's clips.

| Model | Language | Voice | First call (s) | Median latency (s) | Latency range (s) | RTF | Peak RAM (MB) | ASR error |
|---|---|---|---|---|---|---|---|---|
| Piper | English | en_US-lessac-medium | 0.93 | 0.066 | 0.060 to 0.089 | 0.018 | 573 | 0.0% WER |
| Piper | Bangla | bn_BD-google-medium | 0.73 | 0.051 | 0.045 to 0.065 | 0.017 | 573 | 3.9% CER |
| Kokoro 82M | English | af_heart | 1.57 | 0.845 | 0.717 to 1.019 | 0.200 | 741 | 0.0% WER |
| MMS-TTS | English | eng | 3.71 | 0.399 | 0.344 to 0.492 | 0.080 | 1686 | 4.5% WER |
| MMS-TTS | Bangla | ben | 1.91 | 0.300 | 0.269 to 0.460 | 0.083 | 1686 | 8.5% CER |
| NeMo FastPitch + HiFi-GAN | English | ljspeech | 5.63 | 0.495 | 0.436 to 0.574 | 0.106 | 2779 | 0.0% WER |
| NeMo VITS | English | ljspeech | 4.34 | 0.508 | 0.456 to 0.674 | 0.108 | 3101 | 1.3% WER |
| Edge TTS | English | en-US-AriaNeural | 1.47 | 1.128 | 1.032 to 1.482 | 0.238 | 59 | 0.0% WER |
| Edge TTS | Bangla | bn-BD-NabanitaNeural | 1.07 | 1.132 | 0.985 to 1.432 | 0.326 | 59 | 0.5% CER |

What the numbers say:

- Piper is the fastest model, about 55 times faster than real time in both languages, and uses the least memory of the local models. Its slowest English synthesis took 89 ms. Kokoro's fastest took 717 ms.
- Piper, Kokoro, NeMo FastPitch + HiFi-GAN, and Edge TTS made no English word errors on these sentences. Piper is also the fastest of them, so it is the best default for fast, accurate English. Kokoro is slower but has more voices.
- For Bangla, Edge TTS is the most accurate (0.5% CER) but sends text to Microsoft. Piper is the best offline Bangla model here, with two caveats. It swallowed "দয়া" at the start of one sentence, and the Bangla ASR model was partly trained on Piper's source dataset. MMS-TTS made more real errors, such as "বইমেলায়" heard as "পিনেরায়".
- Some Bangla errors are spelling variants, such as "ভালোবাসি" heard as "ভালবাসি" and "ঠান্ডা" as "ঠাণ্ডা". CER ignores spaces, so a split word such as "প্রতি দিন" costs nothing. Check the transcripts in `results.json` and compare close models by ear in the UI.
- Piper, MMS-TTS, and NeMo VITS add random noise when they synthesize, so their ASR error changes from run to run. Piper's Bangla CER was 2.8%, 2.3%, and 3.9% across three runs on the same sentences.
- Edge TTS latency is a network round trip to Microsoft, so it depends on your connection, not your hardware.
- ASR error measures intelligibility, not naturalness. A clear but robotic voice scores well.

The same install, download, and benchmark also ran in a Linux aarch64 Docker container with 5 CPUs and 8 GB of RAM, with no failures. That run used the earlier test sentences. Local models ran 1.1 to 2.2 times slower than on the Mac, with Piper at RTF 0.028 and Kokoro at 0.229. The container's first `uv sync` downloaded more than 3.7 GB of packages. No GPU run has been done.

## Add a model

1. Write an engine class in `src/ttsbench/engines/`. It needs a `download(voices)` classmethod that fetches the weights and a `synthesize(text, voice, speed)` method that returns `ttsbench.core.Audio`. Load weights lazily so the server starts fast.
2. Add a `ModelSpec` to `MODELS` in [`src/ttsbench/registry.py`](src/ttsbench/registry.py) with its voices, the import names it requires, and its `pyproject.toml` extra. Set `weights_per_voice=True` if each voice is a separate checkpoint, so the UI flags the first request for each voice as a cold start.
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
