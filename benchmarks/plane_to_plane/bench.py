#!/usr/bin/env python
"""
Benchmark the fast zenithal plane-to-plane transform across array backends and
projection pairs, and write a single self-contained HTML report.

The same namespace-agnostic ``_accel`` module is driven through several
backends, each in an isolated subprocess (so CPU affinity pinning and GPU
initialisation do not interfere):

    numpy            single-threaded reference
    jax-cpu-1core    jax + jit, process pinned to one core   (fusion only)
    jax-cpu-multi    jax + jit, all cores                     (fusion x cores)
    jax-gpu          jax + jit on the default CUDA device
    cupy             CuPy on the default CUDA device
    wcslib           astropy core wcs_pix2world/world2pix round trip (baseline)

Backends whose libraries or devices are missing are skipped and shown as N/A.

Usage (driver):
    python bench.py [--sizes 1000000,4000000,16000000] [--reps R]
                    [--dtype float64] [--out report.html]
                    [--backends numpy,jax-cpu-multi,...]

Run it on a GPU machine to populate the jax-gpu and cupy columns.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ACCEL_DIR = os.path.normpath(
    os.path.join(HERE, "..", "..", "astropy", "wcs", "_accel")
)

PROJECTIONS = ["TAN", "SIN", "STG", "ARC", "ZEA"]

# backend -> (env overrides, pin_to_one_core)
BACKENDS = {
    "numpy": ({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, False),
    "jax-cpu-1core": (
        {"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1",
         "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false"},
        True,
    ),
    "jax-cpu-multi": ({"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "1"}, False),
    "jax-gpu": ({"JAX_PLATFORMS": "cuda", "JAX_ENABLE_X64": "1"}, False),
    "cupy": ({}, False),
    "torch-cpu": ({}, False),
    "torch-gpu": ({}, False),
    "torch-compile": ({}, False),
    "mlx": ({}, False),
    "wcslib": ({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, False),
    "pixel_to_pixel": ({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, False),
}

# backends benchmarked through a reference astropy code path (not the fast
# transform): the wcslib sphere round trip and the high-level pixel_to_pixel.
REFERENCE_BACKENDS = ("wcslib", "pixel_to_pixel")
# the per-pair baseline every speedup is reported against
BASELINE = "wcslib"


# --------------------------------------------------------------------------
# Shared helpers (used by workers)
# --------------------------------------------------------------------------
def load_accel(accel_dir):
    """Load the four _accel source files as a standalone package."""
    import importlib.util
    import types

    pkg = types.ModuleType("p2p_accel")
    pkg.__path__ = [accel_dir]
    pkg.__package__ = "p2p_accel"
    sys.modules["p2p_accel"] = pkg
    for sub in ("_projections", "_wcs", "_matrix"):
        spec = importlib.util.spec_from_file_location(
            f"p2p_accel.{sub}", os.path.join(accel_dir, f"{sub}.py")
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"p2p_accel.{sub}"] = mod
        spec.loader.exec_module(mod)
    return sys.modules["p2p_accel._matrix"]


def make_wcs(proj, crval, crpix, cdelt, rot_deg):
    from astropy.wcs import WCS

    w = WCS(naxis=2)
    w.wcs.ctype = [f"RA---{proj}", f"DEC--{proj}"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    c, s = np.cos(np.radians(rot_deg)), np.sin(np.radians(rot_deg))
    w.wcs.cd = np.array([[cdelt * c, -cdelt * s], [cdelt * s, cdelt * c]])
    w.wcs.set()
    return w


def wcs_pair(p1, p2, side):
    """Two overlapping WCS (shared frame) with offset, scale and rotation."""
    centre = (side / 2.0, side / 2.0)
    w1 = make_wcs(p1, (266.40, -29.00), centre, -0.0004, 0.0)
    w2 = make_wcs(p2, (266.45, -28.95), (side / 2 + 40, side / 2 - 30),
                  -0.00035, 12.0)
    return w1, w2


# --------------------------------------------------------------------------
# Worker: benchmark ONE backend, write JSON
# --------------------------------------------------------------------------
def _grids(side, dtype):
    """Full input pixel grid plus a small grid for the correctness check."""
    ax = np.arange(side, dtype=dtype)
    gx, gy = np.meshgrid(ax, ax)
    px = np.ascontiguousarray(gx.ravel())
    py = np.ascontiguousarray(gy.ravel())
    sg = np.linspace(0, side - 1, 64).astype(dtype)
    sgx, sgy = np.meshgrid(sg, sg)
    return px, py, sgx.ravel(), sgy.ravel()


def _bench_apply(side, dtype, reps, setup, compute_transform, apply_transform):
    """Benchmark the fast transform over all projection pairs at one size."""
    xp, to_backend, to_numpy, sync, _device, _ver, jit = setup
    px, py, sx, sy = _grids(side, dtype)
    n = px.size
    gx_b, gy_b = to_backend(px), to_backend(py)
    sx64, sy64 = sx.astype(np.float64), sy.astype(np.float64)

    def make_call(t):
        # jax/torch: compile the apply (header values bake in as constants,
        # projection codes are static) so the elementwise chain fuses.
        if jit is not None:
            fn = jit(lambda a, b: apply_transform(t, a, b, xp=xp))
            return lambda: fn(gx_b, gy_b)
        return lambda: apply_transform(t, gx_b, gy_b, xp=xp)

    results = {}
    for p1 in PROJECTIONS:
        for p2 in PROJECTIONS:
            w1, w2 = wcs_pair(p1, p2, side)
            t = compute_transform(w1, w2)
            ms, cores = _time(make_call(t), sync, reps)
            ref = apply_transform(t, sx64, sy64, xp=np)
            got = apply_transform(t, to_backend(sx), to_backend(sy), xp=xp)
            err = float(np.nanmax(np.abs(to_numpy(got[0]) - ref[0])))
            results[f"{p1}->{p2}"] = {
                "mpix": n / (ms / 1e3) / 1e6, "ms": ms, "cores": cores, "err": err,
            }
    return results


def run_worker(backend, sizes, reps, dtype_name, accel_dir, out_path):
    import warnings

    warnings.simplefilter("ignore")  # WCS/pixel_to_pixel emit frame warnings
    result = {"backend": backend, "dtype": dtype_name}
    try:
        _, pin = BACKENDS[backend]
        if pin and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})

        dtype = np.dtype(dtype_name)
        setup = _backend_setup(backend, dtype)
        result["device"] = setup[4]
        result["versions"] = setup[5]

        accel = load_accel(accel_dir)
        compute_transform = accel.compute_transform
        apply_transform = accel.apply_transform

        by_size = {}
        for size in sizes:
            side = int(np.sqrt(size))
            n = side * side
            r = reps if n <= 4_000_000 else max(3, reps - 2)
            if backend in REFERENCE_BACKENDS:
                res = _bench_reference(backend, side, r,
                                       compute_transform, apply_transform)
            else:
                res = _bench_apply(side, dtype, r, setup,
                                   compute_transform, apply_transform)
            by_size[str(n)] = res
        result["results_by_size"] = by_size
        result["status"] = "ok"
        result["sysinfo"] = _sysinfo()
    except Exception as exc:  # report, never crash the driver
        import traceback
        result["status"] = "unavailable"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
    with open(out_path, "w") as fh:
        json.dump(result, fh)


def _noop(_result):
    pass


class _MlxNamespace:
    """Adapt mlx.core to the handful of names the transform calls."""

    def __init__(self, mx):
        self._mx = mx

    def __getattr__(self, name):
        return getattr(self._mx, name)

    def asarray(self, a):  # mlx spells this `array`
        return self._mx.array(a)


def _torch_device(backend, torch):
    cuda = torch.cuda.is_available()
    mps = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    if backend == "torch-gpu":
        if cuda:
            return "cuda"
        if mps:
            return "mps"
        raise RuntimeError("no CUDA or MPS device for torch")
    if backend == "torch-compile":
        return "cuda" if cuda else "cpu"  # mps + compile is still flaky
    return "cpu"


def _backend_setup(backend, dtype):
    """Return (xp, to_backend, to_numpy, sync, device_str, versions, jit).

    ``jit`` wraps the per-pair apply for fusion (jax.jit / torch.compile),
    or None for eager backends.
    """
    if backend in ("numpy", *REFERENCE_BACKENDS):
        return (np, np.asarray, np.asarray, _noop, "cpu (numpy)",
                {"numpy": np.__version__}, None)
    if backend.startswith("jax"):
        import jax
        import jax.numpy as jnp

        try:
            devs = jax.devices()
        except Exception as exc:  # cuda requested but unavailable
            raise RuntimeError(
                f"jax device init failed ({type(exc).__name__}); no GPU?"
            ) from exc
        kind = devs[0].platform
        if backend == "jax-gpu" and kind not in ("gpu", "cuda", "rocm"):
            raise RuntimeError("no GPU device visible to jax")
        device = (f"gpu: {devs[0].device_kind}"
                  if kind in ("gpu", "cuda", "rocm") else "cpu")

        def sync(r):
            r[0].block_until_ready()
            r[1].block_until_ready()

        return (jnp, jnp.asarray, np.asarray, sync, device,
                {"jax": jax.__version__}, jax.jit)
    if backend == "cupy":
        import cupy

        props = cupy.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        name = name.decode() if isinstance(name, bytes) else name

        def sync(r):
            cupy.cuda.runtime.deviceSynchronize()

        return (cupy, cupy.asarray, cupy.asnumpy, sync,
                f"gpu: {name}", {"cupy": cupy.__version__}, None)
    if backend.startswith("torch"):
        import torch

        torch.set_grad_enabled(False)
        dev = _torch_device(backend, torch)
        if dev == "mps" and dtype == np.dtype("float64"):
            raise RuntimeError("torch MPS has no float64; rerun with --dtype float32")
        if dev == "cuda":
            device = f"gpu: {torch.cuda.get_device_name(0)}"
        elif dev == "mps":
            device = "gpu: Apple MPS"
        else:
            device = "cpu"

        def to_backend(a):
            return torch.asarray(a, device=dev)

        def to_numpy(a):
            return a.detach().to("cpu").numpy()

        def sync(r):
            if dev == "cuda":
                torch.cuda.synchronize()
            elif dev == "mps":
                torch.mps.synchronize()

        jit = torch.compile if backend == "torch-compile" else None
        return (torch, to_backend, to_numpy, sync, device,
                {"torch": torch.__version__}, jit)
    if backend == "mlx":
        import mlx.core as mx

        def to_numpy(a):
            mx.eval(a)
            return np.array(a)

        def sync(r):
            mx.eval(r[0], r[1])

        return (_MlxNamespace(mx), mx.array, to_numpy, sync,
                "gpu: Apple MLX", {"mlx": mx.__version__}, None)
    raise ValueError(backend)


def _time(call, sync, reps):
    # warmup (triggers jit compile / first allocation)
    for _ in range(2):
        sync(call())
    best_w = np.inf
    cores = 1.0
    for _ in range(reps):
        w0, c0 = time.perf_counter(), time.process_time()
        sync(call())
        w = time.perf_counter() - w0
        if w < best_w:
            best_w = w
            cpu = time.process_time() - c0
            cores = cpu / w if w > 0 else 1.0
    return best_w * 1e3, cores  # ms, effective host cores


def _reference_call(kind, w1, w2, px, py):
    """Return a 0-arg callable for a reference (non-fast) transform path."""
    if kind == "wcslib":
        def call():
            world = w1.wcs_pix2world(px, py, 0)
            return w2.wcs_world2pix(world[0], world[1], 0)
        return call
    from astropy.wcs.utils import pixel_to_pixel

    def call():
        return pixel_to_pixel(w1, w2, px, py)
    return call


def _bench_reference(kind, side, reps, compute_transform, apply_transform):
    """Benchmark a reference path (wcslib round trip or high-level pixel_to_pixel)."""
    px, py, sx, sy = _grids(side, np.dtype("float64"))
    n = px.size
    results = {}
    for p1 in PROJECTIONS:
        for p2 in PROJECTIONS:
            w1, w2 = wcs_pair(p1, p2, side)
            ms, cores = _time(_reference_call(kind, w1, w2, px, py), _noop, reps)
            # correctness: reference path vs fast transform on the small grid
            t = compute_transform(w1, w2)
            ref = apply_transform(t, sx.astype(np.float64), sy.astype(np.float64), xp=np)
            got0 = _reference_call(kind, w1, w2, sx, sy)()[0]
            err = float(np.nanmax(np.abs(got0 - ref[0])))
            results[f"{p1}->{p2}"] = {
                "mpix": n / (ms / 1e3) / 1e6, "ms": ms,
                "cores": cores, "err": err,
            }
    return results


def _sysinfo():
    info = {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


# --------------------------------------------------------------------------
# Probe: quick availability check for ONE backend, write JSON
# --------------------------------------------------------------------------
def run_probe(backend, dtype_name, out_path):
    result = {"backend": backend}
    try:
        import astropy  # every backend needs WCS to build the headers

        setup = _backend_setup(backend, np.dtype(dtype_name))  # import / check device
        versions = dict(setup[5])
        versions["astropy"] = astropy.__version__
        if backend == "pixel_to_pixel":
            from astropy.wcs.utils import pixel_to_pixel
            if pixel_to_pixel is None:
                raise RuntimeError("pixel_to_pixel not importable")
        result.update(status="ok", device=setup[4], versions=versions)
    except Exception as exc:
        result.update(status="unavailable", reason=f"{type(exc).__name__}: {exc}")
    with open(out_path, "w") as fh:
        json.dump(result, fh)


# --------------------------------------------------------------------------
# Driver: probe, spawn workers, render HTML
# --------------------------------------------------------------------------
def _spawn(args_list, env_over):
    env = dict(os.environ, **env_over)
    return subprocess.run([sys.executable, os.path.abspath(__file__), *args_list],
                          env=env, capture_output=True, text=True, check=False)


def _read_json(path, fallback):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return fallback


def run_driver(args):
    backends = [b for b in (args.backends.split(",") if args.backends
                            else list(BACKENDS)) if b in BACKENDS]
    sizes = [int(s) for s in args.sizes.split(",")]
    tmpdir = os.path.join(HERE, ".bench_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    # 1. probe availability and report up front
    print("Backend availability:")
    probes, available = {}, []
    for b in backends:
        out = os.path.join(tmpdir, f"probe_{b}.json")
        _spawn(["--probe", b, "--dtype", args.dtype, "--out", out], BACKENDS[b][0])
        d = _read_json(out, {"status": "unavailable", "reason": "probe crashed"})
        probes[b] = d
        if d.get("status") == "ok":
            available.append(b)
            ver = ", ".join(f"{k} {v}" for k, v in d.get("versions", {}).items())
            print(f"  {b:16} available    {d.get('device', '')}  [{ver}]")
        else:
            print(f"  {b:16} unavailable  ({d.get('reason', '')})")
    print()

    # 2. benchmark the available backends
    collected = {}
    for b in backends:
        if b not in available:
            collected[b] = probes[b]
            continue
        out = os.path.join(tmpdir, f"{b}.json")
        print(f"running {b:<16} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        proc = _spawn(["--worker", b, "--sizes", args.sizes, "--reps",
                       str(args.reps), "--dtype", args.dtype,
                       "--accel-dir", args.accel_dir, "--out", out], BACKENDS[b][0])
        data = _read_json(out, {"backend": b, "status": "unavailable",
                                "reason": f"worker exited {proc.returncode}",
                                "traceback": proc.stderr[-2000:]})
        collected[b] = data
        tag = data.get("status")
        extra = "" if tag == "ok" else f"({data.get('reason', '')})"
        print(f"{tag} {extra}  [{time.perf_counter() - t0:.1f}s]")

    _print_console_summary(collected, sizes)
    html = render_html(collected, args)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"\nwrote {args.out}")


def _print_console_summary(collected, sizes):
    keys = [str(int(np.sqrt(s)) ** 2) for s in sizes]
    print(f"\nMedian speedup vs {BASELINE}  (columns = {', '.join(_fmt_n(int(k)) for k in keys)}):")
    for b in BACKENDS:
        d = collected.get(b)
        if not d or d.get("status") != "ok":
            continue
        cols = []
        for k in keys:
            res = d.get("results_by_size", {}).get(k, {})
            wbase = collected.get(BASELINE, {}).get("results_by_size", {}).get(k)
            if res and wbase:
                cols.append(f"{_median_ratio(res, wbase):6.1f}x")
            elif res:
                cols.append(f"{_median(res):5.0f}M")
            else:
                cols.append(f"{'-':>6}")
        print(f"  {b:16} " + " ".join(cols))


def _median(results):
    vals = [c["mpix"] for c in results.values()]
    return float(np.median(vals)) if vals else 0.0


def _median_ratio(results, wbase):
    rs = [results[p]["mpix"] / wbase[p]["mpix"]
          for p in results if p in wbase and wbase[p]["mpix"] > 0]
    return float(np.median(rs)) if rs else 0.0


def _heat_ratio(ratio, gmax):
    """Diverging colour: green if faster than the baseline, red if slower."""
    if ratio <= 0 or gmax <= 0:
        return "#f7f7f7", "#000"
    f = max(-1.0, min(1.0, np.log10(ratio) / gmax))
    if f >= 0:  # faster -> green (#2e8b57)
        r, g, b = (int(255 - f * (255 - c)) for c in (46, 139, 87))
    else:       # slower -> red (#c0392b)
        a = -f
        r, g, b = (int(255 - a * (255 - c)) for c in (192, 57, 43))
    return f"rgb({r},{g},{b})", ("#fff" if abs(f) > 0.6 else "#000")


def _matrix_table(results, wbase):
    """5x5 table of speedup vs the wcslib baseline (per projection pair)."""
    ratios = {}
    for p1 in PROJECTIONS:
        for p2 in PROJECTIONS:
            k = f"{p1}->{p2}"
            if k in results and wbase.get(k, {}).get("mpix", 0) > 0:
                ratios[k] = results[k]["mpix"] / wbase[k]["mpix"]
    if not ratios:
        return "<p>no data</p>"
    gmax = max(abs(np.log10(r)) for r in ratios.values()) or 1.0
    out = ['<table class="mat"><tr><th>in \\ out</th>']
    out += [f"<th>{p}</th>" for p in PROJECTIONS]
    out.append("</tr>")
    for p1 in PROJECTIONS:
        out.append(f"<tr><th>{p1}</th>")
        for p2 in PROJECTIONS:
            k = f"{p1}->{p2}"
            cell = results.get(k)
            if cell is None or k not in ratios:
                out.append("<td>-</td>")
                continue
            bg, fg = _heat_ratio(ratios[k], gmax)
            tip = (f"{cell['mpix']:.0f} Mpix/s | {cell['ms']:.2f} ms | "
                   f"err {cell['err']:.1e} | cores~{cell['cores']:.1f}")
            out.append(f'<td style="background:{bg};color:{fg}" title="{tip}">'
                       f'{ratios[k]:.1f}x</td>')
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _fmt_n(n):
    if n >= 1_000_000:
        return f"{n / 1e6:.0f}M"
    if n >= 1000:
        return f"{n // 1000}k"
    return str(n)


def _bar_svg(chart_items, is_ratio):
    if not chart_items:
        return "<p>no successful backends</p>"
    cmax = max(v for _, v in chart_items) or 1.0
    unit = "x" if is_ratio else " Mpix/s"
    bw, gap, x0, top = 480, 24, 160, 12
    bars = []
    for i, (b, v) in enumerate(chart_items):
        y = top + i * (22 + gap)
        w = max(2, bw * v / cmax)
        fill = "#888" if b in REFERENCE_BACKENDS else "#2e8b57"
        bars.append(
            f'<text x="150" y="{y+15}" text-anchor="end" class="bl">{b}</text>'
            f'<rect x="{x0}" y="{y}" width="{w:.0f}" height="22" fill="{fill}"/>'
            f'<text x="{x0+w+6:.0f}" y="{y+15}" class="bv">{v:.1f}{unit}</text>')
    h = len(chart_items) * (22 + gap) + top
    return f'<svg width="800" height="{h}" role="img">' + "".join(bars) + "</svg>"


def _render_size(collected, size_key, device_by_backend):
    """One switchable block: summary + chart + heatmaps for a single size."""
    def res(d):
        return (d or {}).get("results_by_size", {}).get(size_key, {})

    wbase = res(collected.get(BASELINE)) or None

    def med(d):
        r = res(d)
        return _median_ratio(r, wbase) if wbase else _median(r)

    rows = []
    for b in BACKENDS:
        d = collected.get(b)
        if d is None:
            continue
        if d.get("status") != "ok":
            rows.append(f"<tr><td>{b}</td><td>-</td><td colspan='3' class='na'>"
                        f"N/A &mdash; {d.get('reason', 'unavailable')}</td></tr>")
            continue
        r = res(d)
        if not r:
            continue
        val = med(d)
        disp = f"{val:.1f}x" if wbase else f"{val:.0f} Mpix/s"
        cores = np.median([c["cores"] for c in r.values()])
        maxerr = max(c["err"] for c in r.values())
        rows.append(f"<tr><td>{b}</td><td>{device_by_backend[b]}</td>"
                    f"<td class='num'>{disp}</td><td class='num'>{cores:.1f}</td>"
                    f"<td class='num'>{maxerr:.0e}</td></tr>")

    chart = _bar_svg([(b, med(collected[b])) for b in BACKENDS
                      if res(collected.get(b))], bool(wbase))

    heatmaps = []
    for b in BACKENDS:
        d = collected.get(b)
        if not res(d) or not wbase:
            continue
        heatmaps.append(
            f"<h4>{b} <span class='dev'>&mdash; {device_by_backend[b]}</span></h4>"
            f"{_matrix_table(res(d), wbase)}")

    n = int(size_key)
    return (
        f"<div class='sizeblock' data-size='{size_key}'>"
        f"<h2>{n:,} pixels</h2>"
        f"<table class='sum'><tr><th>backend</th><th>device</th>"
        f"<th>median vs {BASELINE}</th><th>host cores~</th><th>max err</th></tr>"
        f"{''.join(rows)}</table>"
        f"<h3>median speedup vs {BASELINE}</h3>{chart}"
        f"<h3>by projection pair</h3>"
        f"<p class='cap'>rows = input proj, cols = output proj; green = faster than "
        f"{BASELINE}, red = slower. hover a cell for Mpix/s / ms / err / cores.</p>"
        f"{''.join(heatmaps)}</div>")


def render_html(collected, args):
    device_by_backend = {b: d.get("device", "-") for b, d in collected.items()}
    sysinfo = next((d["sysinfo"] for d in collected.values() if d.get("sysinfo")), {})

    size_keys = sorted({k for d in collected.values() if d.get("status") == "ok"
                        for k in d.get("results_by_size", {})}, key=int)
    buttons = "".join(
        f'<button class="sizebtn" data-size="{s}" '
        f"onclick=\"showSize('{s}')\">{_fmt_n(int(s))}</button>"
        for s in size_keys)
    blocks = "".join(_render_size(collected, s, device_by_backend) for s in size_keys)

    sys_html = "".join(f"<li><b>{k}</b>: {v}</li>" for k, v in sysinfo.items())
    gpu_devs = sorted({d for b, d in device_by_backend.items()
                       if isinstance(d, str) and d.startswith("gpu")})
    sys_html += "".join(f"<li><b>device</b>: {g}</li>" for g in gpu_devs)

    return _HTML.format(
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        sizes=", ".join(_fmt_n(int(s)) for s in size_keys) or "-",
        dtype=args.dtype, reps=args.reps, sysinfo=sys_html,
        switch=buttons, blocks=blocks,
        default_size=size_keys[-1] if size_keys else "",
    )


_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>plane-to-plane benchmark</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;color:#222;max-width:1000px}}
h1{{margin-bottom:.2rem}} h3{{margin-top:1.8rem}}
.meta{{color:#666;font-size:.9rem}} .dev{{color:#888;font-weight:normal;font-size:.85em}}
table{{border-collapse:collapse;margin:.4rem 0}}
td,th{{border:1px solid #ddd;padding:.35rem .6rem;text-align:center}}
.num{{text-align:right;font-variant-numeric:tabular-nums}}
table.sum th{{background:#f0f0f0}} table.sum td:first-child{{text-align:left;font-weight:600}}
table.mat td{{font-variant-numeric:tabular-nums;min-width:3.2rem}}
table.mat th{{background:#f5f5f5}}
.na{{color:#b00;text-align:left}} .cap{{color:#777;font-size:.85rem;margin:.2rem 0}}
ul{{line-height:1.5}} .bl{{font-size:13px;fill:#333}} .bv{{font-size:12px;fill:#555}}
.note{{background:#f7f9fb;border-left:3px solid #2e8b57;padding:.6rem 1rem;font-size:.9rem}}
.switch{{margin:1rem 0;font-size:.9rem;color:#555}}
.sizebtn{{font:inherit;margin-right:.3rem;padding:.3rem .8rem;border:1px solid #bbb;
  border-radius:5px;background:#fff;cursor:pointer}}
.sizebtn.active{{background:#2e8b57;color:#fff;border-color:#2e8b57}}
.sizeblock{{display:none}} h4{{margin:1.1rem 0 .2rem}}
</style></head><body>
<h1>Zenithal plane-to-pixel benchmark</h1>
<p class="meta">{when} &middot; sizes {sizes} &middot; dtype {dtype} &middot; {reps} timed reps (min wall)</p>
<h2>System</h2><ul>{sysinfo}</ul>
<div class="switch">input size: {switch}</div>
{blocks}
<h2>Notes</h2>
<div class="note">
<p><b>Relative to wcslib.</b> All throughputs are reported as a speedup over the
<code>wcslib</code> sphere round trip for the same projection pair (1.0x = wcslib).
The two grey rows, <code>wcslib</code> and <code>pixel_to_pixel</code>, are the
reference paths this work replaces; <code>pixel_to_pixel</code> is the high-level
path and its &lt;1.0x values show its per-projection overhead over raw wcslib.</p>
<p><b>Single vs multi core.</b> <code>jax-cpu-1core</code> pins the process to one
core, isolating XLA kernel <i>fusion</i> from parallelism; <code>jax-cpu-multi</code>
lets XLA use every core. Compare the <i>host cores~</i> column (CPU time / wall
time) to see how many cores each row actually used &mdash; GPU rows show ~1 host
core because the work runs on the device.</p>
<p><b>float64.</b> All numbers are float64 for parity with the numpy and wcslib
baselines; GPU backends would be substantially faster at float32. Max-error
columns are vs the numpy float64 result (wcslib row is vs the fast transform).</p>
<p><b>jit/compile vs eager.</b> <code>jax-*</code> (via <code>jax.jit</code>) and
<code>torch-compile</code> (via <code>torch.compile</code>) fuse the whole
elementwise chain into one kernel &mdash; the real win over eager execution.
<code>numpy</code>, <code>cupy</code>, <code>torch-cpu/gpu</code> and
<code>mlx</code> run eagerly (one kernel launch per op), so an eager-vs-compiled
gap on the same device reflects fusion, not hardware. <code>torch-compile</code>
runs on CUDA if present, else CPU.</p>
<p><b>wcslib</b> is astropy's core <code>wcs_pix2world</code>&rarr;
<code>wcs_world2pix</code> sphere round trip &mdash; the baseline this work
replaces. It is single-threaded C.</p>
</div>
<script>
function showSize(n){{
  document.querySelectorAll('.sizeblock').forEach(function(e){{
    e.style.display = (e.dataset.size === n) ? 'block' : 'none';
  }});
  document.querySelectorAll('.sizebtn').forEach(function(b){{
    b.classList.toggle('active', b.dataset.size === n);
  }});
}}
showSize('{default_size}');
</script>
</body></html>"""


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--probe", help=argparse.SUPPRESS)
    ap.add_argument("--sizes", default="1000000,4000000,16000000",
                    help="comma list of pixel counts (default 1M,4M,16M)")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--accel-dir", default=DEFAULT_ACCEL_DIR)
    ap.add_argument("--backends", default="",
                    help="comma list; default all: " + ",".join(BACKENDS))
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_results.html"))
    args = ap.parse_args()

    if args.probe:
        run_probe(args.probe, args.dtype, args.out)
    elif args.worker:
        run_worker(args.worker, [int(s) for s in args.sizes.split(",")],
                   args.reps, args.dtype, args.accel_dir, args.out)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
