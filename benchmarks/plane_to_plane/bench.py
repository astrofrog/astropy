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
    python bench.py [--size N] [--reps R] [--dtype float64] [--out report.html]
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
    "wcslib": ({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}, False),
}


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
def run_worker(backend, size, reps, dtype_name, accel_dir, out_path):
    result = {"backend": backend, "size": size, "dtype": dtype_name}
    try:
        _, pin = BACKENDS[backend]
        if pin and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]})

        side = int(np.sqrt(size))
        size = side * side
        result["size"] = size
        dtype = np.dtype(dtype_name)

        ax = np.arange(side, dtype=dtype)
        gx, gy = np.meshgrid(ax, ax)
        px_np = np.ascontiguousarray(gx.ravel())
        py_np = np.ascontiguousarray(gy.ravel())
        # small grid for a correctness check against numpy float64
        sg = np.linspace(0, side - 1, 64)
        sx, sy = (a.ravel() for a in np.meshgrid(sg, sg))

        xp, to_backend, to_numpy, sync, device, versions, jit = _backend_setup(backend)
        result["device"] = device
        result["versions"] = versions

        accel = load_accel(accel_dir)
        compute_transform = accel.compute_transform
        apply_transform = accel.apply_transform

        if backend == "wcslib":
            results = _bench_wcslib(px_np, py_np, sx, sy, reps,
                                    compute_transform, apply_transform)
        else:
            gx_b, gy_b = to_backend(px_np), to_backend(py_np)

            def make_call(t):
                # jax: jit the apply (numpy header values bake in as constants,
                # projection codes are static) so XLA fuses the elementwise
                # chain. Eager backends call the apply directly.
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
                    # correctness vs numpy float64
                    ref = apply_transform(t, sx.astype(np.float64),
                                          sy.astype(np.float64), xp=np)
                    got = apply_transform(t, to_backend(sx), to_backend(sy), xp=xp)
                    err = float(np.nanmax(np.abs(to_numpy(got[0]) - ref[0])))
                    results[f"{p1}->{p2}"] = {
                        "mpix": size / (ms / 1e3) / 1e6, "ms": ms,
                        "cores": cores, "err": err,
                    }
        result["results"] = results
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


def _backend_setup(backend):
    """Return (xp, to_backend, to_numpy, sync, device_str, versions, jit).

    ``jit`` is the compiler to wrap the per-pair apply in (jax.jit for jax,
    None for eager backends).
    """
    if backend in ("numpy", "wcslib"):
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


def _bench_wcslib(px, py, sx, sy, reps, compute_transform, apply_transform):
    results = {}
    side = int(np.sqrt(px.size))
    for p1 in PROJECTIONS:
        for p2 in PROJECTIONS:
            w1, w2 = wcs_pair(p1, p2, side)

            def call():
                world = w1.wcs_pix2world(px, py, 0)
                return w2.wcs_world2pix(world[0], world[1], 0)

            ms, cores = _time(call, lambda r: None, reps)
            # correctness: wcslib round trip vs fast transform on small grid
            t = compute_transform(w1, w2)
            ref = apply_transform(t, sx.astype(np.float64), sy.astype(np.float64), xp=np)
            world = w1.wcs_pix2world(sx, sy, 0)
            got = w2.wcs_world2pix(world[0], world[1], 0)
            err = float(np.nanmax(np.abs(got[0] - ref[0])))
            results[f"{p1}->{p2}"] = {
                "mpix": px.size / (ms / 1e3) / 1e6, "ms": ms,
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
# Driver: spawn workers, render HTML
# --------------------------------------------------------------------------
def run_driver(args):
    backends = args.backends.split(",") if args.backends else list(BACKENDS)
    tmpdir = os.path.join(HERE, ".bench_tmp")
    os.makedirs(tmpdir, exist_ok=True)
    collected = {}
    for backend in backends:
        if backend not in BACKENDS:
            print(f"  skip unknown backend {backend!r}")
            continue
        env_over, _ = BACKENDS[backend]
        out = os.path.join(tmpdir, f"{backend}.json")
        env = dict(os.environ, **env_over)
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", backend,
               "--size", str(args.size), "--reps", str(args.reps),
               "--dtype", args.dtype, "--accel-dir", args.accel_dir, "--out", out]
        print(f"running {backend:<14} ...", end=" ", flush=True)
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               check=False)
        data = None
        if os.path.exists(out):
            with open(out) as fh:
                data = json.load(fh)
        if data is None:
            data = {"backend": backend, "status": "unavailable",
                    "reason": f"worker exited {proc.returncode}",
                    "traceback": proc.stderr[-2000:]}
        collected[backend] = data
        dt = time.perf_counter() - t0
        tag = data.get("status")
        extra = "" if tag == "ok" else f"({data.get('reason', '')})"
        print(f"{tag} {extra}  [{dt:.1f}s]")

    html = render_html(collected, args)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"\nwrote {args.out}")


def _heat(value, lo, hi):
    """Green background intensity for a value on a log scale in [lo, hi]."""
    if value <= 0 or hi <= lo:
        return "#f7f7f7", "#000"
    f = (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    f = max(0.0, min(1.0, f))
    # light -> dark green
    r = int(247 - f * (247 - 0))
    g = int(247 - f * (247 - 109))
    b = int(247 - f * (247 - 44))
    text = "#fff" if f > 0.6 else "#000"
    return f"rgb({r},{g},{b})", text


def _matrix_table(results):
    cells = [results[f"{p1}->{p2}"]["mpix"]
             for p1 in PROJECTIONS for p2 in PROJECTIONS if f"{p1}->{p2}" in results]
    if not cells:
        return "<p>no data</p>"
    lo, hi = min(cells), max(cells)
    out = ['<table class="mat"><tr><th>in \\ out</th>']
    out += [f"<th>{p}</th>" for p in PROJECTIONS]
    out.append("</tr>")
    for p1 in PROJECTIONS:
        out.append(f"<tr><th>{p1}</th>")
        for p2 in PROJECTIONS:
            cell = results.get(f"{p1}->{p2}")
            if cell is None:
                out.append('<td>-</td>')
                continue
            bg, fg = _heat(cell["mpix"], lo, hi)
            tip = f"{cell['ms']:.2f} ms | err {cell['err']:.1e} | cores~{cell['cores']:.1f}"
            out.append(
                f'<td style="background:{bg};color:{fg}" title="{tip}">'
                f'{cell["mpix"]:.0f}</td>'
            )
        out.append("</tr>")
    out.append("</table>")
    return "".join(out)


def _median(results):
    vals = [c["mpix"] for c in results.values()]
    return float(np.median(vals)) if vals else 0.0


def render_html(collected, args):
    ok = {k: v for k, v in collected.items() if v.get("status") == "ok"}
    base = _median(ok["numpy"]["results"]) if "numpy" in ok else 0.0

    sysinfo = {}
    device_by_backend = {}
    for b, d in collected.items():
        device_by_backend[b] = d.get("device", "-")
        if d.get("sysinfo"):
            sysinfo = d["sysinfo"]

    # summary rows
    rows = []
    for b in BACKENDS:
        d = collected.get(b)
        if d is None:
            continue
        if d.get("status") != "ok":
            rows.append(
                f"<tr><td>{b}</td><td>-</td><td colspan='4' class='na'>"
                f"N/A &mdash; {d.get('reason', 'unavailable')}</td></tr>")
            continue
        med = _median(d["results"])
        speedup = med / base if base else 0.0
        cores = np.median([c["cores"] for c in d["results"].values()])
        maxerr = max(c["err"] for c in d["results"].values())
        rows.append(
            f"<tr><td>{b}</td><td>{device_by_backend[b]}</td>"
            f"<td class='num'>{med:.1f}</td><td class='num'>{speedup:.1f}x</td>"
            f"<td class='num'>{cores:.1f}</td><td class='num'>{maxerr:.0e}</td></tr>")

    # svg bar chart of median Mpix/s (log)
    bars = []
    chart_items = [(b, _median(collected[b]["results"]))
                   for b in BACKENDS
                   if collected.get(b, {}).get("status") == "ok"]
    if chart_items:
        cmax = max(v for _, v in chart_items)
        bw, gap, x0, top = 520, 26, 130, 12
        h = len(chart_items) * (22 + gap)
        for i, (b, v) in enumerate(chart_items):
            y = top + i * (22 + gap)
            w = max(2, bw * v / cmax)
            bars.append(
                f'<text x="120" y="{y+15}" text-anchor="end" class="bl">{b}</text>'
                f'<rect x="{x0}" y="{y}" width="{w:.0f}" height="22" fill="#2e8b57"/>'
                f'<text x="{x0+w+6:.0f}" y="{y+15}" class="bv">{v:.0f} Mpix/s</text>')
        svg = (f'<svg width="760" height="{h+top}" role="img">' + "".join(bars)
               + "</svg>")
    else:
        svg = "<p>no successful backends</p>"

    # per-backend heatmaps
    heatmaps = []
    for b in BACKENDS:
        d = collected.get(b)
        if not d or d.get("status") != "ok":
            continue
        heatmaps.append(
            f"<h3>{b} <span class='dev'>&mdash; {device_by_backend[b]}</span></h3>"
            f"<p class='cap'>throughput in Mpix/s (hover a cell for ms / error / "
            f"cores). darker = faster.</p>{_matrix_table(d['results'])}")

    sys_html = "".join(
        f"<li><b>{k}</b>: {v}</li>" for k, v in sysinfo.items())
    gpu_devs = sorted({device_by_backend[b] for b in ("jax-gpu", "cupy")
                       if device_by_backend.get(b, "-") not in ("-", None)})
    if gpu_devs:
        sys_html += "".join(f"<li><b>gpu</b>: {g}</li>" for g in gpu_devs)

    return _HTML.format(
        when=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        size=f"{args.size:,}", dtype=args.dtype, reps=args.reps,
        sysinfo=sys_html, summary="".join(rows), chart=svg,
        heatmaps="".join(heatmaps),
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
</style></head><body>
<h1>Zenithal plane-to-pixel benchmark</h1>
<p class="meta">{when} &middot; {size} pixels &middot; dtype {dtype} &middot; {reps} timed reps (min wall)</p>
<h2>System</h2><ul>{sysinfo}</ul>
<h2>Summary</h2>
<table class="sum"><tr><th>backend</th><th>device</th><th>median Mpix/s</th>
<th>vs numpy</th><th>host cores~</th><th>max err</th></tr>{summary}</table>
<h3>median throughput</h3>{chart}
<h2>Per-backend throughput by projection pair</h2>
<p class="cap">rows = input projection, columns = output projection. TAN&rarr;TAN uses
the single-matrix fast path; all other cells use the general five-step path.</p>
{heatmaps}
<h2>Notes</h2>
<div class="note">
<p><b>Single vs multi core.</b> <code>jax-cpu-1core</code> pins the process to one
core, isolating XLA kernel <i>fusion</i> from parallelism; <code>jax-cpu-multi</code>
lets XLA use every core. Compare the <i>host cores~</i> column (CPU time / wall
time) to see how many cores each row actually used &mdash; GPU rows show ~1 host
core because the work runs on the device.</p>
<p><b>float64.</b> All numbers are float64 for parity with the numpy and wcslib
baselines; GPU backends would be substantially faster at float32. Max-error
columns are vs the numpy float64 result (wcslib row is vs the fast transform).</p>
<p><b>jit vs eager.</b> The jax backends are <code>jax.jit</code>-compiled, so XLA
fuses the whole elementwise chain into one kernel (the real win over eager
numpy). CuPy runs eagerly (one kernel launch per op, no fusion), so a jax-gpu vs
cupy gap partly reflects fusion rather than hardware. numpy is also eager.</p>
<p><b>wcslib</b> is astropy's core <code>wcs_pix2world</code>&rarr;
<code>wcs_world2pix</code> sphere round trip &mdash; the baseline this work
replaces. It is single-threaded C.</p>
</div></body></html>"""


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--size", type=int, default=4_000_000)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--accel-dir", default=DEFAULT_ACCEL_DIR)
    ap.add_argument("--backends", default="",
                    help="comma list; default all: " + ",".join(BACKENDS))
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_results.html"))
    args = ap.parse_args()

    if args.worker:
        run_worker(args.worker, args.size, args.reps, args.dtype,
                   args.accel_dir, args.out)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
