# TTS Model Comparison

A comprehensive benchmarking and testing suite for NVIDIA NeMo TTS models including FastPitch, HiFi-GAN, and VITS.

## Features

### CLI Scripts
- **Individual Model Scripts**: Standalone performance testing for each TTS model
- **Unified Benchmark**: Run all models together with `main.py`
- **Performance Metrics**: Latency, duration, RTF (Real-Time Factor) measurements

### Web Application
- **Interactive UI**: Browser-based interface for testing TTS models
- **Model Selection**: Choose between FastPitch+HiFi-GAN, E2E, or VITS
- **Real-time Synthesis**: Generate speech with performance tracking
- **Resource Monitoring**: CPU, memory, and GPU usage analysis
- **Audio Playback**: Instant preview of generated audio

## Setup

1) Create and activate a virtual env with uv (fast and self-contained):

```bash
uv venv .venv
source .venv/bin/activate
```

2) Install deps (pulls NeMo + Torch + FastAPI):

```bash
uv pip install -e .
```

## Usage

### Web Application (Recommended)

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

### CLI Scripts

#### Run All Models
```bash
uv run python main.py
```

#### Individual Models
```bash
# FastPitch + HiFi-GAN (two-stage)
uv run python fastpitch+HiFi-GAN.py

# FastPitch + HiFi-GAN (end-to-end)
uv run python "fastpitch+HiFi-GAN(E2E).py"

# VITS
uv run python vits+hifigan.py
```

Outputs land in `outputs/` with latency, duration, and real-time factor printed for each model.

## Models

### FastPitch + HiFi-GAN (Two-Stage)
- **FastPitch**: Mel-spectrogram generator
- **HiFi-GAN**: Neural vocoder
- Pretrained: `tts_en_fastpitch` + `tts_en_lj_hifigan_ft_mixertts`

### FastPitch + HiFi-GAN (E2E)
- End-to-end synthesis with fallback wrapper
- Faster inference with combined model
- Uses same pretrained models as two-stage

### VITS
- End-to-end TTS with variational inference
- Pretrained: `tts_en_lj_vits`
- Single model for both spectrogram and waveform generation

## Performance Metrics

All scripts measure:
- **Latency**: Time to generate audio (seconds)
- **Duration**: Length of generated audio (seconds)
- **RTF**: Real-Time Factor (latency / duration)
- **Sample Rate**: Audio sample rate (Hz)

Web app additionally tracks:
- **CPU Usage**: Percentage during synthesis
- **Memory**: RAM usage in MB
- **GPU Memory**: VRAM usage in MB (if available)

## Dependencies

- Python 3.12+
- NeMo Toolkit 2.6.0+
- PyTorch with CUDA support
- FastAPI + Uvicorn (for web app)
- psutil (for resource monitoring)
- soundfile (for audio I/O)

## Troubleshooting

### ml_dtypes Import Error
The project includes compatibility aliases for `ml_dtypes` to prevent ONNX import errors. These are defined before NeMo imports in all scripts.

### CUDA/GPU Issues
Models automatically fall back to CPU if CUDA is unavailable. For best performance, use a CUDA-enabled GPU.

### Model Download
On first run, NeMo will download pretrained models (~500MB-1GB each). This may take several minutes depending on your connection.

