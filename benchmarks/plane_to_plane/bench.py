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
def run_worker(backend, size, reps, dtype_name, accel_dir, out_path):
    import warnings

    warnings.simplefilter("ignore")  # WCS/pixel_to_pixel emit frame warnings
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

        if backend in REFERENCE_BACKENDS:
            results = _bench_reference(backend, px_np, py_np, sx, sy, reps,
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


def _bench_reference(kind, px, py, sx, sy, reps, compute_transform, apply_transform):
    """Benchmark a reference path (wcslib round trip or high-level pixel_to_pixel)."""
    results = {}
    side = int(np.sqrt(px.size))
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
# Probe: quick availability check for ONE backend, write JSON
# --------------------------------------------------------------------------
def run_probe(backend, out_path):
    result = {"backend": backend}
    try:
        import astropy  # every backend needs WCS to build the headers

        setup = _backend_setup(backend)  # imports the backend / checks device
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
    tmpdir = os.path.join(HERE, ".bench_tmp")
    os.makedirs(tmpdir, exist_ok=True)

    # 1. probe availability and report up front
    print("Backend availability:")
    probes, available = {}, []
    for b in backends:
        out = os.path.join(tmpdir, f"probe_{b}.json")
        _spawn(["--probe", b, "--out", out], BACKENDS[b][0])
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
        proc = _spawn(["--worker", b, "--size", str(args.size), "--reps",
                       str(args.reps), "--dtype", args.dtype,
                       "--accel-dir", args.accel_dir, "--out", out], BACKENDS[b][0])
        data = _read_json(out, {"backend": b, "status": "unavailable",
                                "reason": f"worker exited {proc.returncode}",
                                "traceback": proc.stderr[-2000:]})
        collected[b] = data
        tag = data.get("status")
        extra = "" if tag == "ok" else f"({data.get('reason', '')})"
        print(f"{tag} {extra}  [{time.perf_counter() - t0:.1f}s]")

    _print_console_summary(collected)
    html = render_html(collected, args)
    with open(args.out, "w") as fh:
        fh.write(html)
    print(f"\nwrote {args.out}")


def _print_console_summary(collected):
    wbase = collected.get(BASELINE, {}).get("results")
    head = f"vs {BASELINE}" if wbase else "Mpix/s"
    print(f"\nMedian throughput ({head}):")
    for b in BACKENDS:
        d = collected.get(b)
        if not d or d.get("status") != "ok":
            continue
        med = _median(d["results"])
        if wbase:
            print(f"  {b:16} {_median_ratio(d['results'], wbase):6.1f}x"
                  f"   ({med:.0f} Mpix/s)")
        else:
            print(f"  {b:16} {med:.0f} Mpix/s")


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


def render_html(collected, args):
    ok = {k: v for k, v in collected.items() if v.get("status") == "ok"}
    wbase = ok.get(BASELINE, {}).get("results")

    sysinfo = {}
    device_by_backend = {}
    for b, d in collected.items():
        device_by_backend[b] = d.get("device", "-")
        if d.get("sysinfo"):
            sysinfo = d["sysinfo"]

    def med_ratio(d):
        return _median_ratio(d["results"], wbase) if wbase else _median(d["results"])

    # summary rows
    rows = []
    for b in BACKENDS:
        d = collected.get(b)
        if d is None:
            continue
        if d.get("status") != "ok":
            rows.append(
                f"<tr><td>{b}</td><td>-</td><td colspan='3' class='na'>"
                f"N/A &mdash; {d.get('reason', 'unavailable')}</td></tr>")
            continue
        val = med_ratio(d)
        disp = f"{val:.1f}x" if wbase else f"{val:.0f} Mpix/s"
        cores = np.median([c["cores"] for c in d["results"].values()])
        maxerr = max(c["err"] for c in d["results"].values())
        rows.append(
            f"<tr><td>{b}</td><td>{device_by_backend[b]}</td>"
            f"<td class='num'>{disp}</td>"
            f"<td class='num'>{cores:.1f}</td><td class='num'>{maxerr:.0e}</td></tr>")

    # svg bar chart of median speedup vs baseline
    chart_items = [(b, med_ratio(collected[b]))
                   for b in BACKENDS
                   if collected.get(b, {}).get("status") == "ok"]
    if chart_items:
        cmax = max(v for _, v in chart_items) or 1.0
        unit = "x" if wbase else " Mpix/s"
        bw, gap, x0, top = 520, 26, 150, 12
        h = len(chart_items) * (22 + gap)
        bars = []
        for i, (b, v) in enumerate(chart_items):
            y = top + i * (22 + gap)
            w = max(2, bw * v / cmax)
            fill = "#888" if b in REFERENCE_BACKENDS else "#2e8b57"
            bars.append(
                f'<text x="140" y="{y+15}" text-anchor="end" class="bl">{b}</text>'
                f'<rect x="{x0}" y="{y}" width="{w:.0f}" height="22" fill="{fill}"/>'
                f'<text x="{x0+w+6:.0f}" y="{y+15}" class="bv">{v:.1f}{unit}</text>')
        svg = (f'<svg width="780" height="{h+top}" role="img">' + "".join(bars)
               + "</svg>")
    else:
        svg = "<p>no successful backends</p>"

    # per-backend heatmaps
    heatmaps = []
    for b in BACKENDS:
        d = collected.get(b)
        if not d or d.get("status") != "ok" or not wbase:
            continue
        heatmaps.append(
            f"<h3>{b} <span class='dev'>&mdash; {device_by_backend[b]}</span></h3>"
            f"<p class='cap'>speedup vs {BASELINE} (hover a cell for Mpix/s / ms / "
            f"error / cores). green = faster than {BASELINE}, red = slower.</p>"
            f"{_matrix_table(d['results'], wbase)}")

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
<table class="sum"><tr><th>backend</th><th>device</th><th>median vs wcslib</th>
<th>host cores~</th><th>max err</th></tr>{summary}</table>
<h3>median speedup vs wcslib</h3>{chart}
<h2>Per-backend speedup vs wcslib by projection pair</h2>
<p class="cap">rows = input projection, columns = output projection. Each cell is the
throughput relative to the wcslib sphere round trip for that same pair (so wcslib
is 1.0x everywhere). TAN&rarr;TAN uses the single-matrix fast path; all other
cells use the general five-step path.</p>
{heatmaps}
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
    ap.add_argument("--probe", help=argparse.SUPPRESS)
    ap.add_argument("--size", type=int, default=4_000_000)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--accel-dir", default=DEFAULT_ACCEL_DIR)
    ap.add_argument("--backends", default="",
                    help="comma list; default all: " + ",".join(BACKENDS))
    ap.add_argument("--out", default=os.path.join(HERE, "benchmark_results.html"))
    args = ap.parse_args()

    if args.probe:
        run_probe(args.probe, args.out)
    elif args.worker:
        run_worker(args.worker, args.size, args.reps, args.dtype,
                   args.accel_dir, args.out)
    else:
        run_driver(args)


if __name__ == "__main__":
    main()
