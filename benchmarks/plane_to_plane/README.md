# Zenithal plane-to-plane benchmark

Benchmarks the fast `astropy.wcs._accel` transform across array backends and all
25 zenithal projection pairs, and writes a single self-contained HTML report.

```bash
python bench.py                       # all backends, 4M points, -> benchmark_results.html
python bench.py --size 16000000       # past the CPU cache cliff (steady-state)
python bench.py --backends numpy,jax-gpu,cupy --out report.html
```

Backends: `numpy`, `jax-cpu-1core`, `jax-cpu-multi`, `jax-gpu`, `cupy`, plus two
reference paths this work replaces — the `wcslib` sphere round trip and the
high-level `pixel_to_pixel`. Each runs in its own subprocess and is shown as N/A
if its library or device is missing, so the script is safe to run anywhere and
only fills the GPU columns on a GPU host.

The script first probes and prints which backends are available, then reports all
throughputs as a speedup relative to the `wcslib` path for each projection pair
(so `wcslib` is 1.0x and `pixel_to_pixel` shows the high-level overhead).

To populate the GPU rows, install the matching wheels on that host, e.g.
`pip install "jax[cuda12]"` and/or `pip install cupy-cuda12x`. The numbers are
float64 (parity with the CPU baselines); pass `--dtype float32` to see the GPU's
float32 throughput.

`bench.py` loads the transform straight from the source tree (`--accel-dir`,
default resolved relative to this script), so it does not require the branch to
be installed — only a working `astropy` (for `WCS`) and the chosen backend.
