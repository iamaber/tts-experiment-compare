"""Download the weights for every installed model, plus the ASR models the benchmark uses.

Files land in the Hugging Face cache, the NeMo cache, or models/ (Kokoro). Runs that find a file already present skip it.
"""

import argparse
import importlib.util
import sys
import time

from ttsbench.registry import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS),
        help="default: every installed model",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="skip the ASR models used by the benchmark",
    )
    args = parser.parse_args()

    failures = []
    for spec in (MODELS[id] for id in args.models or MODELS):
        if not spec.available:
            print(f"skip {spec.id}: not installed, run `uv sync --extra {spec.extra}`")
            continue
        if spec.online:
            print(f"skip {spec.id}: cloud service, nothing to download")
            continue
        print(f"downloading {spec.id} ({len(spec.voices)} voices)", flush=True)
        start = time.perf_counter()
        try:
            spec.download()
        except Exception as error:
            failures.append(spec.id)
            print(f"  failed: {error}", file=sys.stderr)
            continue
        print(f"  done in {time.perf_counter() - start:.1f}s")

    if not args.skip_asr and all(map(importlib.util.find_spec, ("jiwer", "nemo"))):
        from ttsbench.asr import ASR_MODELS, load_asr

        for language, name in ASR_MODELS.items():
            print(f"downloading ASR model {name}", flush=True)
            load_asr(language)

    if failures:
        sys.exit(f"failed: {', '.join(failures)}")


if __name__ == "__main__":
    main()
