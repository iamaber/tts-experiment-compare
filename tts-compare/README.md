# TTS model comparison

Benchmark and test scripts for NVIDIA NeMo TTS models: FastPitch, HiFi-GAN, and VITS.

## Features

### CLI scripts
- One standalone performance test script per TTS model
- `main.py` runs all models together
- Each run reports latency, duration, and RTF (real-time factor)

### Web application
- Browser UI for testing TTS models
- Choose FastPitch+HiFi-GAN or VITS
- Generates speech and reports performance metrics
- Reports CPU, memory, and GPU usage
- Plays the generated audio in the browser

## Setup

1) Create and activate a virtual env with uv:

```bash
uv venv .venv
source .venv/bin/activate
```

2) Install dependencies (NeMo, Torch, FastAPI):

```bash
uv pip install -e .
```

## Usage

### Web application (recommended)

Start the FastAPI server:

```bash
uv run python app.py
```

Open your browser to http://localhost:8000 and:
1. Select a TTS model from the dropdown
2. Enter text to synthesize
3. Click "Generate Speech" or press Ctrl+Enter
4. View performance metrics and resource usage
5. Play the generated audio

### CLI scripts

#### Run all models
```bash
uv run python main.py
```

#### Individual models
```bash
# FastPitch + HiFi-GAN (two-stage)
uv run python fastpitch+HiFi-GAN.py

# VITS
uv run python vits+hifigan.py
```

Each script writes audio to `outputs/` and prints latency, duration, and real-time factor.

## Models

### FastPitch + HiFi-GAN (two-stage)
- FastPitch generates the mel spectrogram.
- HiFi-GAN is the neural vocoder.
- Pretrained: `tts_en_fastpitch` + `tts_en_lj_hifigan_ft_mixertts`

### VITS
- End-to-end TTS with variational inference
- Pretrained: `tts_en_lj_vits`
- One model generates both the spectrogram and the waveform

## Performance metrics

All scripts measure:
- Latency: time to generate audio (seconds)
- Duration: length of generated audio (seconds)
- RTF: real-time factor (latency / duration)
- Sample rate: audio sample rate (Hz)

The web app also tracks:
- CPU usage during synthesis (percent)
- RAM usage (MB)
- VRAM usage (MB, if available)

## Dependencies

- Python 3.12+
- NeMo Toolkit 2.6.0+
- PyTorch with CUDA support
- FastAPI + Uvicorn (for web app)
- psutil (for resource monitoring)
- soundfile (for audio I/O)

## Troubleshooting

### ml_dtypes import error
`pyproject.toml` pins `onnx<1.18`. Newer onnx releases reference `ml_dtypes.float4_e2m1fn`, which the pinned `ml-dtypes<0.5` lacks, so importing NeMo fails.

### CUDA/GPU issues
Models fall back to CPU if CUDA is unavailable. A CUDA GPU is faster.

### Model download
On first run, NeMo downloads pretrained models (~500MB-1GB each). This can take several minutes.

