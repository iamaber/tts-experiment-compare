"""Benchmark every registered model: first-call time, latency, real-time factor, memory, and ASR error rate.

Each model runs in its own subprocess so load time and peak memory are not shared between models.
Rounds alternate between models, so drift and background load affect every model alike.
"""

import argparse
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import soundfile as sf

from ttsbench.core import (
    ASR_REQUIRES,
    LANGUAGE_NAMES,
    METRIC,
    REPO_ROOT,
    Audio,
    Language,
    installed,
)
from ttsbench.registry import MODELS

TEXTS_PATH = REPO_ROOT / "benchmarks" / "texts.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
AUDIO_DIR = REPO_ROOT / "benchmarks" / "audio"


class WorkerError(RuntimeError):
    pass


def load_texts() -> dict[Language, list[str]]:
    return json.loads(TEXTS_PATH.read_text())


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3


def run_model(model_id: str, languages: list[Language], runs: int) -> list[dict]:
    spec = MODELS[model_id]
    texts = load_texts()
    start = time.perf_counter()
    engine = spec.load()
    construct_s = time.perf_counter() - start
    samples = []
    for language in (lang for lang in spec.languages if lang in languages):
        voice = next(v for v in spec.voices if v.language == language)
        start = time.perf_counter()
        engine.synthesize(texts[language][0], voice, 1.0)
        first_call_s = time.perf_counter() - start + (0.0 if samples else construct_s)
        latencies, audio_seconds, clips = [], [], []
        for index, text in enumerate(texts[language]):
            for _ in range(runs):
                start = time.perf_counter()
                audio: Audio = engine.synthesize(text, voice, 1.0)
                latencies.append(time.perf_counter() - start)
                audio_seconds.append(audio.duration)
            path = AUDIO_DIR / model_id / language / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio.samples, audio.sample_rate)
            clips.append({"text": text, "path": str(path.relative_to(REPO_ROOT))})
        samples.append(
            {
                "language": language,
                "voice": voice.id,
                "first_call_s": first_call_s,
                "latencies": latencies,
                "audio_seconds": audio_seconds,
                "clips": clips,
            }
        )
    for sample in samples:
        sample["peak_rss_mb"] = peak_rss_mb()
    return samples


def run_isolated(model_id: str, languages: list[Language], runs: int) -> list[dict]:
    cmd = [sys.executable, "-m", "ttsbench.benchmark", "--worker", model_id]
    cmd += ["--runs", str(runs), "--languages", *languages]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        print(f"  {model_id} failed:\n    " + "\n    ".join(tail), file=sys.stderr)
        raise WorkerError(tail[-1] if tail else f"exit {proc.returncode}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def summarize(model_id: str, samples: list[dict]) -> dict:
    latencies = [latency for sample in samples for latency in sample["latencies"]]
    audio_seconds = sum(sum(sample["audio_seconds"]) for sample in samples)
    language = samples[0]["language"]
    return {
        "model": model_id,
        "language": language,
        "voice": samples[0]["voice"],
        "metric": METRIC[language],
        "first_call_s": statistics.median(sample["first_call_s"] for sample in samples),
        "latency_median_s": statistics.median(latencies),
        "latency_min_s": min(latencies),
        "latency_max_s": max(latencies),
        "rtf": sum(latencies) / audio_seconds,
        "peak_rss_mb": max(sample["peak_rss_mb"] for sample in samples),
        "rounds": len(samples),
        "runs": len(latencies),
        "clips": samples[-1]["clips"],
    }


def benchmark(
    model_ids: list[str], languages: list[Language], runs: int, rounds: int
) -> list[dict]:
    samples: dict[str, list[dict]] = {model_id: [] for model_id in model_ids}
    errors: dict[str, str] = {}
    for round_number in range(1, rounds + 1):
        for model_id in model_ids:
            if model_id in errors:
                continue
            print(f"round {round_number}/{rounds}: {model_id}", file=sys.stderr)
            try:
                samples[model_id] += run_isolated(model_id, languages, runs)
            except WorkerError as error:
                errors[model_id] = str(error)

    rows = []
    for model_id in model_ids:
        if model_id in errors:
            rows.append({"model": model_id, "error": errors[model_id]})
            continue
        by_language: dict[str, list[dict]] = {}
        for sample in samples[model_id]:
            by_language.setdefault(sample["language"], []).append(sample)
        rows += [summarize(model_id, group) for group in by_language.values()]
    return rows


def score(rows: list[dict]) -> None:
    from ttsbench.asr import error_rate, transcribe

    for row in rows:
        if "clips" not in row:
            continue
        audios = [
            Audio(*sf.read(REPO_ROOT / clip["path"], dtype="float32"))
            for clip in row["clips"]
        ]
        transcripts = transcribe(audios, row["language"])
        for clip, transcript in zip(row["clips"], transcripts):
            clip["transcript"] = transcript
            clip["error_rate"] = error_rate(clip["text"], transcript, row["language"])
        row["error_rate"] = statistics.mean(clip["error_rate"] for clip in row["clips"])


def markdown(rows: list[dict]) -> str:
    lines = [
        "| Model | Language | Voice | First call (s) | Median latency (s) | Latency range (s) | RTF | Peak RAM (MB) | ASR error |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        name = MODELS[row["model"]].name
        if "error" in row:
            lines.append(f"| {name} | | | failed: {row['error']} | | | | | |")
            continue
        error = row.get("error_rate")
        error_cell = "not scored" if error is None else f"{error:.1%} {row['metric']}"
        lines.append(
            f"| {name} | {LANGUAGE_NAMES[row['language']]} | {row['voice']} | {row['first_call_s']:.2f} "
            f"| {row['latency_median_s']:.3f} | {row['latency_min_s']:.3f} to {row['latency_max_s']:.3f} "
            f"| {row['rtf']:.3f} | {row['peak_rss_mb']:.0f} | {error_cell} |"
        )
    return "\n".join(lines) + "\n"


def save(rows: list[dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    (output / "results.md").write_text(markdown(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        help="default: every installed model",
    )
    parser.add_argument(
        "--languages", nargs="+", choices=["en", "bn"], default=["en", "bn"]
    )
    parser.add_argument(
        "--rounds", type=int, default=3, help="fresh subprocess per model per round"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="timed runs per sentence per round"
    )
    parser.add_argument(
        "--no-asr", action="store_true", help="skip the ASR round-trip accuracy check"
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(run_model(args.worker, args.languages, args.runs)))
        return

    model_ids = args.models or [
        model_id for model_id, spec in MODELS.items() if spec.available
    ]
    rows = benchmark(model_ids, args.languages, args.runs, args.rounds)
    save(rows, args.output)
    asr_installed = installed(*ASR_REQUIRES)
    if not args.no_asr and not asr_installed:
        print(
            "skipping ASR scoring: run `uv sync --extra asr` to enable it",
            file=sys.stderr,
        )
    if not args.no_asr and asr_installed:
        print("scoring with ASR", file=sys.stderr)
        score(rows)
        save(rows, args.output)
    print(markdown(rows))


if __name__ == "__main__":
    main()
