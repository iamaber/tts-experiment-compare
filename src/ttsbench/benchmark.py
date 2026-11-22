"""Benchmark every registered model: first-call time, latency, real-time factor, memory, and ASR error rate.

Each model runs in its own subprocess so load time and peak memory are not shared between models.
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

from ttsbench.core import LANGUAGE_NAMES, REPO_ROOT, Audio, Language
from ttsbench.registry import MODELS

TEXTS_PATH = REPO_ROOT / "benchmarks" / "texts.json"
RESULTS_DIR = REPO_ROOT / "benchmarks" / "results"
AUDIO_DIR = REPO_ROOT / "benchmarks" / "audio"


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
    rows = []
    for language in (lang for lang in spec.languages if lang in languages):
        voice = next(v for v in spec.voices if v.language == language)
        start = time.perf_counter()
        engine.synthesize(texts[language][0], voice, 1.0)
        first_call_s = time.perf_counter() - start + (construct_s if not rows else 0.0)
        latencies, audio_seconds, clips = [], 0.0, []
        for index, text in enumerate(texts[language]):
            for _ in range(runs):
                start = time.perf_counter()
                audio: Audio = engine.synthesize(text, voice, 1.0)
                latencies.append(time.perf_counter() - start)
                audio_seconds += audio.duration
            path = AUDIO_DIR / model_id / language / f"{index}.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, audio.samples, audio.sample_rate)
            clips.append({"text": text, "path": str(path.relative_to(REPO_ROOT))})
        rows.append(
            {
                "model": model_id,
                "language": language,
                "voice": voice.id,
                "first_call_s": first_call_s,
                "latency_median_s": statistics.median(latencies),
                "latency_min_s": min(latencies),
                "latency_max_s": max(latencies),
                "rtf": sum(latencies) / audio_seconds,
                "audio_s": audio_seconds / runs,
                "runs": len(latencies),
                "clips": clips,
            }
        )
    for row in rows:
        row["peak_rss_mb"] = peak_rss_mb()
    return rows


def run_isolated(model_id: str, languages: list[Language], runs: int) -> list[dict]:
    cmd = [
        sys.executable,
        "-m",
        "ttsbench.benchmark",
        "--worker",
        model_id,
        "--runs",
        str(runs),
    ]
    cmd += ["--languages", *languages]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = proc.stderr.strip().splitlines()[-5:]
        print(f"  {model_id} failed:\n    " + "\n    ".join(tail), file=sys.stderr)
        return [
            {
                "model": model_id,
                "error": tail[-1] if tail else f"exit {proc.returncode}",
            }
        ]
    return json.loads(proc.stdout.strip().splitlines()[-1])


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
        "| Model | Language | Voice | First call (s) | Median latency (s) | RTF | Peak RAM (MB) | ASR error |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        name = MODELS[row["model"]].name
        if "error" in row:
            lines.append(f"| {name} | | | failed: {row['error']} | | | | |")
            continue
        error = row.get("error_rate")
        metric = "WER" if row["language"] == "en" else "CER"
        error_cell = f"{error:.1%} {metric}" if error is not None else "not scored"
        lines.append(
            f"| {name} | {LANGUAGE_NAMES[row['language']]} | {row['voice']} | {row['first_call_s']:.2f} "
            f"| {row['latency_median_s']:.3f} | {row['rtf']:.3f} | {row['peak_rss_mb']:.0f} | {error_cell} |"
        )
    return "\n".join(lines) + "\n"


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
    parser.add_argument("--runs", type=int, default=3, help="timed runs per sentence")
    parser.add_argument(
        "--no-asr",
        action="store_true",
        help="skip the ASR round-trip accuracy check",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(json.dumps(run_model(args.worker, args.languages, args.runs)))
        return

    model_ids = args.models or [id for id, spec in MODELS.items() if spec.available]
    rows = []
    for model_id in model_ids:
        print(f"benchmarking {model_id}", file=sys.stderr)
        rows += run_isolated(model_id, args.languages, args.runs)
    if not args.no_asr:
        print("scoring with ASR", file=sys.stderr)
        score(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False)
    )
    table = markdown(rows)
    (args.output / "results.md").write_text(table)
    print(table)


if __name__ == "__main__":
    main()
