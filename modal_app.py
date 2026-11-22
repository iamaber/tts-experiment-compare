"""Run the ttsbench benchmark on a Modal GPU.

    uvx modal run modal_app.py --gpu L4

Results land in benchmarks/results/<gpu>/. Model weights persist in the ttsbench-cache volume between runs.
"""

import subprocess
from pathlib import Path

import modal

REPO = Path(__file__).parent
CACHE = "/cache"
RESULTS = Path("/tmp/results")

app = modal.App("ttsbench")
image = modal.Image.from_dockerfile(
    REPO / "docker" / "Dockerfile.gpu", context_dir=REPO
)
cache = modal.Volume.from_name("ttsbench-cache", create_if_missing=True)


@app.function(image=image, volumes={CACHE: cache}, cpu=8, memory=32768, timeout=3600)
def benchmark(args: list[str]) -> dict[str, str]:
    subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"],
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-c",
            "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see the GPU';"
            " print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name())",
        ],
        check=True,
    )
    subprocess.run(["ttsbench-download"], check=True)
    cache.commit()
    subprocess.run(["ttsbench-bench", "--output", str(RESULTS), *args], check=True)
    return {path.name: path.read_text() for path in RESULTS.iterdir()}


@app.local_entrypoint()
def main(gpu: str = "L4", rounds: int = 3, models: str = "", languages: str = "en,bn"):
    args = ["--rounds", str(rounds), "--languages", *languages.split(",")]
    if models:
        args += ["--models", *models.split(",")]
    files = benchmark.with_options(gpu=gpu).remote(args)
    output = REPO / "benchmarks" / "results" / gpu.lower()
    output.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (output / name).write_text(content)
    print(files["results.md"])
    print(f"saved to {output.relative_to(REPO)}")
