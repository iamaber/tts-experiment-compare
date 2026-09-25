# TTS experiments

Text-to-speech experiments with NVIDIA NeMo (FastPitch, HiFi-GAN, VITS, TitaNet) and Kyutai TTS. Each folder is a separate project with its own dependencies.

| Project | What it does |
|---|---|
| [`tts-compare`](tts-compare/) | Benchmarks FastPitch + HiFi-GAN and VITS for latency, real-time factor, CPU, RAM, and VRAM, from the CLI or a web UI |
| [`fastpitch-hifigan-finetune`](fastpitch-hifigan-finetune/) | Multi-speaker FastPitch TTS, speaker embedding extraction (TitaNet, Resemblyzer, ECAPA), and a speaker adapter for voice cloning |
| [`voice-swap-NeMo`](voice-swap-NeMo/) | Fine-tunes FastPitch and HiFi-GAN on 5 minutes of a new speaker, plus an ASR-to-TTS voice swap demo |
| [`tts-test-kayutai`](tts-test-kayutai/) | Kyutai TTS notebook with voice conditioning, and an HTML frontend for a Kyutai TTS backend |

## Requirements

- [uv](https://docs.astral.sh/uv/) to manage each project's Python environment
- An NVIDIA GPU with CUDA. Most scripts fall back to CPU, which is much slower.
- [Git LFS](https://git-lfs.com/) for the `.nemo` checkpoint and the Kyutai tokenizer files. Run `git lfs pull` after cloning.

NeMo downloads pretrained models on first use, about 100 MB to 1 GB per model.

## tts-compare

Python 3.12.

```bash
cd tts-compare
uv sync
```

Web UI: start the server, then open http://localhost:8000. Pick a model, enter text, and press "Synthesize" or Ctrl+Enter (Cmd+Enter on macOS).

```bash
uv run python app.py
```

CLI benchmarks. Audio goes to `outputs/`, and each run prints latency, audio duration, and real-time factor.

```bash
uv run python main.py                  # all models
uv run python fastpitch+HiFi-GAN.py    # FastPitch + HiFi-GAN
uv run python vits+hifigan.py          # VITS
```

`pyproject.toml` pins `onnx<1.18`. Newer onnx releases need `ml_dtypes` types that `ml-dtypes<0.5` doesn't have, so NeMo fails to import.

## fastpitch-hifigan-finetune

Python 3.10.

```bash
cd fastpitch-hifigan-finetune
uv sync
```

Multi-speaker TTS with the pretrained voices (`speaker=92` by default, edit `__main__` to change text or speaker). Writes `test.wav`.

```bash
uv run python tts.py
```

Extract a speaker embedding from a recording. Backends are `titanet` (default), `resemblyzer`, and `ecapa`.

```bash
uv run python speaker_encoder.py custom-voice/record.wav my_voice.pt titanet
```

Generate speech conditioned on a voice. The input can be a `.wav` file or a saved `.pt` embedding.

```bash
uv run python custom_voice_tts.py my_voice.pt "Your text here" output.wav 1.0
uv run python custom_fastpitch.py custom-voice/record.wav "Your text here" custom_output.wav 1.0
```

The speaker adapter adds the TitaNet embedding to FastPitch's encoder output, but it is untrained, so the output does not match the reference voice yet.

Microsoft Edge TTS, which needs an internet connection:

```bash
uv run python edge-xtts/synthesize.py --list
uv run python edge-xtts/synthesize.py --text "Hello there" --voice en-US-AriaNeural --output output.mp3
```

`experiments/` holds earlier voice cloning attempts. See [`experiments/README.md`](fastpitch-hifigan-finetune/experiments/README.md) and [`VOICE_CLONING_COMPARISON.md`](fastpitch-hifigan-finetune/experiments/VOICE_CLONING_COMPARISON.md) for what each one does and why it falls short.

## voice-swap-NeMo

Python 3.12.

```bash
cd voice-swap-NeMo
uv sync
uv run --with jupyter jupyter lab
```

Notebooks:

- `FastPitch_Finetuning.ipynb` walks through the full fine-tuning pipeline: download the speaker 9017 data, fine-tune FastPitch, generate mel spectrograms with the fine-tuned model, then fine-tune HiFi-GAN on them.
- `voice_swap.ipynb` transcribes `2086-149220-0033.wav` with QuartzNet ASR, then speaks the transcript with TTS.

The training scripts can also run directly. FastPitch, starting from the bundled `tts_en_fastpitch_align.nemo`:

```bash
uv run python fastpitch_finetune.py --config-name=fastpitch_align_v1.05 \
  train_dataset=./9017_manifest_train_dur_5_mins_local.json \
  validation_datasets=./9017_manifest_dev_ns_all_local.json \
  sup_data_path=./fastpitch_sup_data \
  phoneme_dict_path=tts_dataset_files/cmudict-0.7b_nv22.10 \
  heteronyms_path=tts_dataset_files/heteronyms-052722.txt \
  exp_manager.exp_dir=./ljspeech_to_9017_no_mixing_5_mins \
  +init_from_nemo_model=./tts_en_fastpitch_align.nemo \
  +trainer.max_steps=1000 ~trainer.max_epochs \
  trainer.check_val_every_n_epoch=25 \
  model.train_ds.dataloader_params.batch_size=4 model.validation_ds.dataloader_params.batch_size=4 \
  model.train_ds.dataloader_params.num_workers=0 model.validation_ds.dataloader_params.num_workers=0 \
  model.n_speakers=1 model.pitch_mean=152.3 model.pitch_std=64.0 \
  model.pitch_fmin=30 model.pitch_fmax=512 model.optim.lr=2e-4 \
  ~model.optim.sched model.optim.name=adam trainer.devices=1 trainer.strategy=auto \
  model.train_ds.dataset.min_duration=0.5 \
  ++model.text_tokenizer.add_blank_at=true
```

HiFi-GAN, starting from the pretrained `tts_en_hifigan`. The manifests need `audio_filepath`, `mel_filepath`, and `duration` for each entry. The notebook creates `hifigan_train_ft.json` and `hifigan_val_ft.json` from the fine-tuned FastPitch.

```bash
uv run python hifigan_finetune.py --config-name=hifigan \
  +init_from_pretrained_model=tts_en_hifigan \
  train_dataset=./hifigan_train_ft.json \
  validation_datasets=./hifigan_val_ft.json \
  model.train_ds.dataloader_params.batch_size=8 \
  model.validation_ds.dataloader_params.batch_size=8 \
  model.max_steps=500 model.optim.lr=0.00001 ~model.optim.sched \
  exp_manager.exp_dir=./hifigan_ft \
  trainer.check_val_every_n_epoch=10 trainer.devices=1 trainer.strategy=auto
```

## tts-test-kayutai

This project has no `pyproject.toml`. The notebook installs its own dependency.

```bash
cd tts-test-kayutai
uv run --with "moshi==0.2.11" --with jupyter jupyter lab
```

`tts_kayutai.ipynb` loads Kyutai TTS from Hugging Face and generates speech in the voice of `demo_audio/american_accent_0.wav`. It runs on CUDA.

`frontend-html/` is a static web UI with word highlighting, voice upload, and system metrics. It calls a backend at `http://localhost:8000` (`/api/tts`, `/api/voices`, `/api/clone-voice`, `/api/system-info`, `/api/metrics`). That backend is not in this repository. To serve the UI:

```bash
cd tts-test-kayutai/frontend-html
python3 -m http.server 3000
```

Then open http://localhost:3000.
