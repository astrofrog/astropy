# Zenithal plane-to-plane benchmark

Benchmarks the fast `astropy.wcs._accel` transform across array backends and all
25 zenithal projection pairs, and writes a single self-contained HTML report.

```bash
python bench.py                              # all backends, sizes 1M/4M/16M -> benchmark_results.html
python bench.py --sizes 1000000,9000000      # custom input sizes
python bench.py --backends numpy,jax-gpu,cupy,torch-gpu --out report.html
```

Backends: `numpy`, `jax-cpu-1core`, `jax-cpu-multi`, `jax-gpu`, `cupy`,
`torch-cpu`, `torch-gpu`, `torch-compile`, `mlx`, plus two reference paths this
work replaces — the `wcslib` sphere round trip and the high-level
`pixel_to_pixel`. Each runs in its own subprocess and is shown as N/A if its
library or device is missing, so the script is safe to run anywhere and only
fills the GPU/torch/mlx columns where those are installed.

The script first probes and prints which backends are available, then reports all
throughputs as a speedup relative to the `wcslib` path for each projection pair
(so `wcslib` is 1.0x and `pixel_to_pixel` shows the high-level overhead). The HTML
benchmarks each requested input size and gives buttons to switch between them.

`torch-compile` and the `jax-*` backends are compiled (kernel fusion); `numpy`,
`cupy`, `torch-cpu/gpu` and `mlx` run eagerly. MLX and torch-on-MPS are float32
(Apple GPUs have no float64), so run those with `--dtype float32`.

`torch-compile` needs a Triton-capable GPU (CUDA capability >= 7.0); on older
cards it falls back to compiling on CPU. `torch-gpu` and `cupy` also need wheels
built for your GPU architecture — older cards (e.g. Pascal / sm_61) may hit "no
kernel image" or an nvrtc `--gpu-architecture` error and show as N/A; that is an
install/hardware limitation, not a benchmark error. The run continues regardless.

To populate the GPU rows, install the matching wheels on that host, e.g.
`pip install "jax[cuda12]"` and/or `pip install cupy-cuda12x`. The numbers are
float64 (parity with the CPU baselines); pass `--dtype float32` to see the GPU's
float32 throughput.

`bench.py` loads the transform straight from the source tree (`--accel-dir`,
default resolved relative to this script), so it does not require the branch to
be installed — only a working `astropy` (for `WCS`) and the chosen backend.
