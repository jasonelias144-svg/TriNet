"""Ours. Writing and reading a hologram, simulated: light of one colour, as
a scalar field, with numpy alone.

    python experiments/holography.py              # all three parts, about a minute
    python experiments/holography.py --part c     # one part

  a  Write a hologram of three points at different depths; read it back
     and refocus at each depth. Then the same plate recorded without a
     reference beam (intensity alone, as film without holography).
  b  Cut the plate to one quarter and read it again.
  c  Many memories in one hologram: keys at different angles, and keys of
     random phase; against matrix memories (the plain sum of outer
     products, and the delta rule) holding the same count of numbers.

The optics is set out in docs/NEURON-CATALOG.md ("Writing and reading a
hologram"), which proposed this simulation; the founder agreed to it on
2026-11-23 (STATUS.md, Decided). Results are in experiments/README.md.
"""
import argparse

import numpy as np

LAM = 0.633  # wavelength, micrometres (a helium-neon laser)
DX = 1.0  # pixel pitch on the plate, micrometres


def grid(n, dx=DX):
    """Positions on the plate (centred) and squared spatial frequencies."""
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    f = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(f, f, indexing="xy")
    return X, Y, FX**2 + FY**2


def propagate(u, z, f2, lam=LAM):
    """The angular spectrum method: the field is a sum of plane waves, one
    per spatial frequency, and each moves on by z with only a change of
    phase, exp(i kz z), kz = 2 pi sqrt(1/lam^2 - f^2). Waves finer than a
    wavelength (evanescent) die away; they are dropped."""
    arg = 1 / lam**2 - f2
    h = np.where(arg > 0, np.exp(2j * np.pi * np.sqrt(np.maximum(arg, 0)) * z), 0)
    return np.fft.ifft2(np.fft.fft2(u) * h)


def point_wave(X, Y, px, py, pz, lam=LAM):
    """The field on the plate (z = 0) of a point of light at (px, py, -pz)."""
    r = np.sqrt((X - px) ** 2 + (Y - py) ** 2 + pz**2)
    return np.exp(2j * np.pi / lam * r) * pz / r


def view(u, f2, radius):
    """Look only along the object's directions: keep the spatial
    frequencies within `radius` (an aperture in a lens's focal plane)."""
    return np.fft.ifft2(np.fft.fft2(u) * (f2 <= radius**2))


def fwhm(row, i, dx=DX):
    """Full width at half maximum of the peak at index i of a 1-D profile."""
    half, lo, hi = row[i] / 2, i, i
    while lo > 0 and row[lo] > half:
        lo -= 1
    while hi < len(row) - 1 and row[hi] > half:
        hi += 1
    return (hi - lo) * dx


def scene(n):
    """Three points at different depths. Depths and offsets scale with the
    plate's width, so a smaller plate sees them under the same angles."""
    s = n / 1024
    return [(-150.0 * s, 0.0, 20000.0 * s), (0.0, 120.0 * s, 30000.0 * s), (150.0 * s, -100.0 * s, 40000.0 * s)]


def write_and_read(n=1024, sin_ref=0.15, quarter=False, reference=True):
    """Record the plate, read it with the reference, and refocus at each
    point's depth. Returns {depth: intensity image} and the points."""
    X, Y, f2 = grid(n)
    points = scene(n)
    obj = sum(point_wave(X, Y, *p) for p in points)
    obj /= np.abs(obj).mean()
    ref = np.exp(2j * np.pi / LAM * sin_ref * X)  # a plane wave at an angle
    carrier = sin_ref / LAM
    # Writing: the plate records intensity only.
    plate = np.abs(obj + ref) ** 2 if reference else np.abs(obj) ** 2
    if quarter:
        plate = plate * ((X < 0) & (Y < 0))
    # Reading: light it with the reference again, look along the object's way.
    seen = view(ref * plate, f2, carrier / 2)
    return {pz: np.abs(propagate(seen, -pz, f2)) ** 2 for _, _, pz in points}, points


def index(n, px, py):
    return int(round(py / DX + n / 2)), int(round(px / DX + n / 2))


def part_a(n=1024):
    """Brightness at each point's place, in each refocused plane, as a share
    of that plane's brightest value: with and without a reference beam."""
    out = {}
    for reference in (True, False):
        images, points = write_and_read(n, reference=reference)
        out[reference] = np.array([[images[plane][index(n, px, py)] / images[plane].max()
                                    for px, py, _ in points] for _, _, plane in points])
    return out, points


def part_b(n=1024):
    """Spot widths (FWHM, micrometres) from the whole plate and from one
    quarter, the widths diffraction gives for a square window of width D
    (0.886 lam z / D), and how bright each point still is in the
    quarter's image."""
    whole, points = write_and_read(n)
    cut, _ = write_and_read(n, quarter=True)
    rows = []
    for px, py, pz in points:
        iy, ix = index(n, px, py)
        rows.append((pz, fwhm(whole[pz][iy], ix), fwhm(cut[pz][iy], ix), 0.886 * LAM * pz / (n * DX),
                     0.886 * LAM * pz / (n * DX / 2), cut[pz][iy, ix] / cut[pz].max()))
    return rows


def unit_phase(rng, shape):
    return np.exp(2j * np.pi * rng.random(shape))


def match(stored, out):
    """How well a read-out matches each stored memory: |<v, out>| / (|v| |out|)."""
    return np.abs(stored.conj() @ out) / (np.linalg.norm(stored, axis=1) * np.linalg.norm(out))


def random_keys(rng, n, d, trials=20):
    """n memories of d unit phases each, in one hologram of d numbers
    (point-by-point product: value times the key's conjugate, summed).
    Read with one key: the mean match with the memory asked for, and how
    often it is the best match among all stored."""
    score, found = [], []
    for _ in range(trials):
        vals, keys = unit_phase(rng, (n, d)), unit_phase(rng, (n, d))
        holo = (vals * keys.conj()).sum(0)
        j = rng.integers(n)
        s = match(vals, holo * keys[j])
        score.append(s[j])
        found.append(np.argmax(s) == j)
    return float(np.mean(score)), float(np.mean(found))


def write_matrix(keys, vals, rule):
    """An m x m memory of unit-phase keys and values, read as k F / m."""
    m = keys.shape[1]
    F = np.zeros((m, m), complex)
    for k, v in zip(keys, vals):
        err = v if rule == "sum" else v - k @ F / m
        F += np.outer(k.conj(), err)
    return F


def matrix_memory(rng, n, m, rule, trials=20):
    """n memories of m unit phases in an m x m matrix (m^2 numbers), written
    as the plain sum of outer products, or by the delta rule (each write
    stores only what the matrix does not yet return for its key)."""
    score, found = [], []
    for _ in range(trials):
        vals, keys = unit_phase(rng, (n, m)), unit_phase(rng, (n, m))
        F = write_matrix(keys, vals, rule)
        j = rng.integers(n)
        s = match(vals, keys[j] @ F / m)
        score.append(s[j])
        found.append(np.argmax(s) == j)
    return float(np.mean(score)), float(np.mean(found))


def angle_keys(rng, m=256, side=3, band=0.04):
    """side x side pages, each a smooth random image whose spatial
    frequencies lie within `band` (cycles per pixel), stored with keys that
    are plane waves at different angles (carriers on the grid's own
    frequencies), and read back through a viewing aperture. Returns the
    largest relative read-out error and the share of the plate's spatial
    frequencies each page uses."""
    xs = np.arange(m)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    f = np.fft.fftfreq(m)
    FX, FY = np.meshgrid(f, f, indexing="xy")
    lowpass = (FX**2 + FY**2) <= band**2
    pages = [np.fft.ifft2(np.fft.fft2(rng.standard_normal((m, m))) * lowpass) for _ in range(side**2)]
    step = round(0.3 * m) / m
    keys = [np.exp(2j * np.pi * ((a - side // 2) * step * XX + (b - side // 2) * step * YY))
            for a in range(side) for b in range(side)]
    holo = sum(p * k.conj() for p, k in zip(pages, keys))
    errs = [np.linalg.norm(np.fft.ifft2(np.fft.fft2(holo * k) * lowpass) - p) / np.linalg.norm(p)
            for p, k in zip(pages, keys)]
    return max(errs), float(lowpass.mean())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--part", nargs="+", choices=["a", "b", "c"], default=["a", "b", "c"])
    p.add_argument("--n", type=int, default=1024, help="plate width in pixels of 1 micrometre (parts a, b)")
    args = p.parse_args(argv)
    if "a" in args.part:
        out, points = part_a(args.n)
        print("A. Refocused at each depth: brightness at each point's place, as a share of the plane's brightest")
        print("   " + "plane".ljust(26) + "".join(f"point at {pz / 1000:4.1f} mm  " for *_, pz in points))
        for label, ref in (("with a reference beam", True), ("without one", False)):
            print(f"   {label}:")
            for (_, _, plane), row in zip(points, out[ref]):
                print(f"     {plane / 1000:5.1f} mm" + " " * 16 + "".join(f"{v:12.2f}    " for v in row))
    if "b" in args.part:
        print("\nB. The whole plate against one quarter: spot width at half maximum, micrometres")
        print("   (diffraction's 0.886 lam z / D in brackets; pixels are 1 micrometre)")
        for pz, whole, cut, ew, ec, shown in part_b(args.n):
            print(f"   point at {pz / 1000:4.1f} mm: whole {whole:3.0f} ({ew:.0f}), quarter {cut:3.0f} ({ec:.0f}),"
                  f" {cut / whole:.1f} times as wide; in the quarter's image, {shown:.2f} of the brightest")
    if "c" in args.part:
        rng = np.random.default_rng(0)
        print("\nC. Many memories in one store, read one at a time (20 trials each)")
        print("   score: mean match of the read-out with the memory asked for; found: how often that memory")
        print("   is the best match among all stored")
        print("   memories  hologram, d = 4,096         sum of outer products, 64 x 64   delta rule, 64 x 64")
        for n in (1, 2, 8, 32, 64, 128, 512):
            h = random_keys(rng, n, 4096)
            s = matrix_memory(rng, n, 64, "sum")
            dlt = matrix_memory(rng, n, 64, "delta")
            print(f"   {n:8d}  score {h[0]:.3f}, found {h[1]:4.0%}       score {s[0]:.3f}, found {s[1]:4.0%}"
                  f"        score {dlt[0]:.3f}, found {dlt[1]:4.0%}")
        print("   (each holds 4,096 numbers; 1/sqrt(n) for the hologram's score: "
              + ", ".join(f"{1 / np.sqrt(n):.3f}" for n in (1, 2, 8, 32, 64, 128, 512)) + ")")
        err, share = angle_keys(rng)
        print(f"\n   Keys at different angles: 9 pages, each using {share:.1%} of the plate's spatial frequencies,")
        print(f"   read back with a largest relative error of {err:.1e}")


if __name__ == "__main__":
    main()
