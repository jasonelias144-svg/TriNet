"""Ours. Writing and reading a hologram, simulated: light of one colour, as
a scalar field, with numpy alone.

    python experiments/holography.py              # all three parts, about 90 seconds
    python experiments/holography.py --part c     # one part

  a  Write a hologram of three points at different depths; read it back
     and refocus at each depth. Then a control: the same light recorded
     without a reference beam (intensity alone, as film without
     holography), lit straight on and refocused the same way.
  b  Cut the plate to one quarter and read it again.
  c  Many memories in one store: keys of random phase, and keys at
     different angles, each both as the complex sum of the term a
     hologram is read for (value times conjugate key) and as a plate
     recorded in intensity; against matrix memories holding the same
     count of numbers.

The optics is set out in docs/NEURON-CATALOG.md ("Writing and reading a
hologram"), which proposed this simulation; the founder agreed to it on
2026-11-23 (STATUS.md, Decided). Results are in experiments/README.md.
"""
import argparse

import numpy as np

LAM = 0.633  # wavelength, micrometres (a helium-neon laser)
DX = 1.0  # pixel pitch on the plate, micrometres


def freqs2(n, dx=DX):
    f = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(f, f, indexing="xy")
    return FX**2 + FY**2


def grid(n, dx=DX):
    """Positions on the plate (centred) and squared spatial frequencies."""
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing="xy")
    return X, Y, freqs2(n, dx)


def propagate(u, z, f2, lam=LAM):
    """The angular spectrum method: the field is a sum of plane waves, one
    per spatial frequency, and each moves on by z with only a change of
    phase, exp(i kz z), kz = 2 pi sqrt(1/lam^2 - f^2). Waves finer than a
    wavelength (evanescent) die away and are dropped; at 1 micrometre
    pixels and 633 nm there are none on the grid."""
    arg = 1 / lam**2 - f2
    h = np.where(arg > 0, np.exp(2j * np.pi * np.sqrt(np.maximum(arg, 0)) * z), 0)
    return np.fft.ifft2(np.fft.fft2(u) * h)


def propagate_padded(u, z):
    """propagate() on a grid twice as wide, the field set in its middle
    and the middle cut out again, so that light leaving the plate's edge
    does not wrap round to the other side (the FFT's grid is periodic)."""
    n = u.shape[0]
    big = np.zeros((2 * n, 2 * n), complex)
    big[n // 2: n // 2 + n, n // 2: n // 2 + n] = u
    return propagate(big, z, freqs2(2 * n))[n // 2: n // 2 + n, n // 2: n // 2 + n]


def point_wave(X, Y, px, py, pz, lam=LAM):
    """The field on the plate (z = 0) of a point of light at (px, py, -pz)."""
    r = np.sqrt((X - px) ** 2 + (Y - py) ** 2 + pz**2)
    return np.exp(2j * np.pi / lam * r) * pz / r


def view(u, f2, radius):
    """Look only along the object's directions: keep the spatial
    frequencies within `radius` (an aperture in a lens's focal plane)."""
    return np.fft.ifft2(np.fft.fft2(u) * (f2 <= radius**2))


def fwhm(row, i, dx=DX):
    """Full width at half maximum of the peak at index i of a 1-D profile,
    with the two half-maximum crossings placed by linear interpolation."""
    half, lo, hi = row[i] / 2, i, i
    while lo > 0 and row[lo] > half:
        lo -= 1
    while hi < len(row) - 1 and row[hi] > half:
        hi += 1
    left = lo + (half - row[lo]) / (row[lo + 1] - row[lo])
    right = hi - (half - row[hi]) / (row[hi - 1] - row[hi])
    return (right - left) * dx


def scene(n):
    """Three points at different depths. Depths and offsets scale with the
    plate's width, so a smaller plate sees them under the same angles."""
    s = n / 1024
    return [(-150.0 * s, 0.0, 20000.0 * s), (0.0, 120.0 * s, 30000.0 * s), (150.0 * s, -100.0 * s, 40000.0 * s)]


def write_and_read(n=1024, sin_ref=0.15, quarter=False, reference=True):
    """Record the plate and read it, then refocus at each point's depth.
    With the reference: the plate records |O + R|^2, is lit with R again
    and looked at through an aperture around the object's direction (the
    image term; the zero order and the twin travel elsewhere). Without:
    the plate records |O|^2 and is lit straight on, through the same
    aperture, now around the axis. Returns {depth: intensity image}."""
    X, Y, f2 = grid(n)
    points = scene(n)
    obj = sum(point_wave(X, Y, *p) for p in points)
    obj /= np.abs(obj).mean()
    ref = np.exp(2j * np.pi / LAM * sin_ref * X)  # a plane wave at an angle
    radius = sin_ref / LAM / 2
    plate = np.abs(obj + ref) ** 2 if reference else np.abs(obj) ** 2
    if quarter:
        plate = plate * ((X < 0) & (Y < 0))
    seen = view(ref * plate if reference else plate, f2, radius)
    return {pz: np.abs(propagate_padded(seen, -pz)) ** 2 for _, _, pz in points}, points


def index(n, px, py):
    return int(round(py / DX + n / 2)), int(round(px / DX + n / 2))


def part_a(n=1024):
    """Brightness at each point's place, in each refocused plane, as a share
    of that plane's brightest value: with a reference beam, and the control."""
    out = {}
    for reference in (True, False):
        images, points = write_and_read(n, reference=reference)
        out[reference] = np.array([[images[plane][index(n, px, py)] / images[plane].max()
                                    for px, py, _ in points] for _, _, plane in points])
    return out, points


def part_b(n=1024):
    """For each point: spot widths at half maximum along x and y (micrometres)
    from the whole plate and from one quarter; diffraction's 0.886 lam z / D
    for each window (D its width); how bright the quarter's spot is beside
    the whole plate's; and how far (pixels) each spot's peak, the
    brightest pixel within 8 pixels of the point, lies from it, for the
    whole plate and for the quarter."""
    whole, points = write_and_read(n)
    cut, _ = write_and_read(n, quarter=True)
    rows = []
    for px, py, pz in points:
        iy, ix = index(n, px, py)
        widths = [(fwhm(img[pz][iy, :], ix), fwhm(img[pz][:, ix], iy)) for img in (whole, cut)]
        offset = []
        for img in (whole, cut):
            win = img[pz][iy - 8: iy + 9, ix - 8: ix + 9]
            dy, dx = np.unravel_index(np.argmax(win), win.shape)
            offset.append(int(max(abs(dy - 8), abs(dx - 8))))
        rows.append(dict(depth=pz, whole=widths[0], quarter=widths[1],
                         expect_whole=0.886 * LAM * pz / (n * DX), expect_quarter=0.886 * LAM * pz / (n * DX / 2),
                         brightness=cut[pz][iy, ix] / whole[pz][iy, ix], offset=tuple(offset)))
    return rows


def unit_phase(rng, shape):
    return np.exp(2j * np.pi * rng.random(shape))


def match(stored, out):
    """How well a read-out matches each stored memory: |<v, out>| / (|v| |out|)."""
    return np.abs(stored.conj() @ out) / (np.linalg.norm(stored, axis=1) * np.linalg.norm(out))


def hologram_read(vals, keys, j, recorded):
    """A hologram of n memories (unit phases, one value and one key each),
    read with key j. recorded=False: the complex sum of the term a
    hologram is read for, value times the key's conjugate. recorded=True:
    a plate exposed once per memory, recording intensity |v + k|^2 = 2 +
    v k* + v* k; each exposure's mean (2) is taken off, so what remains
    also holds the twin terms v* k."""
    if recorded:
        holo = (np.abs(vals + keys) ** 2 - 2).sum(0)
    else:
        holo = (vals * keys.conj()).sum(0)
    return holo * keys[j]


def random_keys(rng, n, d, trials=100, recorded=False):
    """n memories of d unit phases each, in one hologram of d numbers, read
    one at a time with its key: the mean match with the memory asked for,
    and how often it is the best match among those stored (picking from a
    known list, not recalling it)."""
    score, found = [], []
    for _ in range(trials):
        vals, keys = unit_phase(rng, (n, d)), unit_phase(rng, (n, d))
        j = rng.integers(n)
        s = match(vals, hologram_read(vals, keys, j, recorded))
        score.append(s[j])
        found.append(np.argmax(s) == j)
    return float(np.mean(score)), float(np.mean(found))


def write_matrix(keys, vals, rule):
    """An m x m memory of unit-phase keys and values, read as k F / m.
    "sum": the plain sum of outer products. "delta": the delta rule in one
    pass at full step, each write storing only what F does not yet return
    for its key, with no decay and no clipping (TriNet's fast memory adds
    both, and learned gates). "lstsq": the matrix fitted to all memories
    at once by least squares, a bound on what any matrix can do."""
    m = keys.shape[1]
    if rule == "lstsq":
        return np.linalg.lstsq(keys / m, vals, rcond=None)[0]
    F = np.zeros((m, m), complex)
    for k, v in zip(keys, vals):
        err = v if rule == "sum" else v - k @ F / m
        F += np.outer(k.conj(), err)
    return F


def matrix_memory(rng, n, m, rule, trials=100):
    """As random_keys, for n memories of m unit phases in an m x m matrix."""
    score, found = [], []
    for _ in range(trials):
        vals, keys = unit_phase(rng, (n, m)), unit_phase(rng, (n, m))
        F = write_matrix(keys, vals, rule)
        j = rng.integers(n)
        s = match(vals, keys[j] @ F / m)
        score.append(s[j])
        found.append(np.argmax(s) == j)
    return float(np.mean(score)), float(np.mean(found))


def delta_by_age(rng, n, m=64, trials=50):
    """The delta rule's match for each quarter of the memories, oldest
    written first to newest written last."""
    by_quarter = np.zeros(4)
    for _ in range(trials):
        vals, keys = unit_phase(rng, (n, m)), unit_phase(rng, (n, m))
        F = write_matrix(keys, vals, "delta")
        s = np.array([match(vals[j:j + 1], keys[j] @ F / m)[0] for j in range(n)])
        by_quarter += [part.mean() for part in np.array_split(s, 4)]
    return by_quarter / trials


def angle_keys(rng, m=256, band=0.02, recorded=False):
    """Nine pages, each a smooth random image whose spatial frequencies lie
    within `band` (cycles per pixel), stored with keys that are plane
    waves at different angles, read back with each key through an aperture
    of radius `band` around the axis. The carriers sit on the grid's own
    frequencies, all on one side (x from 26/256 to 51/256), and apart by
    more than twice the band, so that on a recorded plate the zero order
    and the twins travel elsewhere too. recorded=False: the complex sum of
    page times conjugate key. recorded=True: the plate exposed once per
    page, recording |page + key|^2. Returns the largest relative read-out
    error and the share of the plate's frequencies each page uses."""
    xs = np.arange(m)
    XX, YY = np.meshgrid(xs, xs, indexing="xy")
    f = np.fft.fftfreq(m)
    FX, FY = np.meshgrid(f, f, indexing="xy")
    lowpass = (FX**2 + FY**2) <= band**2
    pages = [np.fft.ifft2(np.fft.fft2(rng.standard_normal((m, m))) * lowpass) for _ in range(9)]
    pages = [p / np.abs(p).max() / 3 for p in pages]
    carriers = [(cx / m, cy / m) for cx in (26, 38, 51) for cy in (-13, 0, 13)]
    keys = [np.exp(2j * np.pi * (cx * XX + cy * YY)) for cx, cy in carriers]
    if recorded:
        plate = sum(np.abs(p + k) ** 2 for p, k in zip(pages, keys))
    else:
        plate = sum(p * k.conj() for p, k in zip(pages, keys))
    errs = [np.linalg.norm(np.fft.ifft2(np.fft.fft2(plate * k) * lowpass) - p) / np.linalg.norm(p)
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
        for label, ref in (("with a reference beam", True), ("control, without one", False)):
            print(f"   {label}:")
            for (_, _, plane), row in zip(points, out[ref]):
                print(f"     {plane / 1000:5.1f} mm" + " " * 16 + "".join(f"{v:12.2f}    " for v in row))
    if "b" in args.part:
        print("\nB. The whole plate against one quarter: spot width at half maximum, x / y, micrometres")
        print("   (diffraction's 0.886 lam z / D in brackets)")
        for r in part_b(args.n):
            (wx, wy), (qx, qy) = r["whole"], r["quarter"]
            print(f"   point at {r['depth'] / 1000:4.1f} mm: whole {wx:4.1f} / {wy:4.1f} ({r['expect_whole']:.1f}),"
                  f" quarter {qx:4.1f} / {qy:4.1f} ({r['expect_quarter']:.1f}), {qx / wx:.2f} / {qy / wy:.2f} times"
                  f" as wide;\n     brightness {r['brightness']:.3f} of the whole plate's; peak"
                  f" {r['offset'][0]} px (whole) and {r['offset'][1]} px (quarter) from the point")
    if "c" in args.part:
        rng = np.random.default_rng(0)
        ns = (1, 2, 8, 32, 64, 128, 512)
        print("\nC. Many memories in one store, read one at a time (100 trials each)")
        print("   score: mean match of the read-out with the memory asked for; found: how often that memory")
        print("   is the best match among those stored. Every store holds 4,096 complex numbers.")
        print("   memories  hologram, d = 4,096:       matrix, 64 x 64:")
        print("             complex sum  recorded      sum          delta (1 pass) least squares")
        for n in ns:
            cells = [random_keys(rng, n, 4096), random_keys(rng, n, 4096, recorded=True),
                     matrix_memory(rng, n, 64, "sum"), matrix_memory(rng, n, 64, "delta"),
                     matrix_memory(rng, n, 64, "lstsq")]
            print(f"   {n:8d}  " + "  ".join(f"{s:.3f} {f:4.0%}" for s, f in cells))
        print("   (1/sqrt(n): " + ", ".join(f"{1 / np.sqrt(n):.3f}" for n in ns) + ")")
        print("\n   The delta rule by age, oldest quarter of the memories to newest (64 x 64):")
        for n in (32, 64, 128):
            print(f"   {n:8d}  " + ", ".join(f"{v:.2f}" for v in delta_by_age(rng, n)))
        for recorded in (False, True):
            err, share = angle_keys(rng, recorded=recorded)
            label = "recorded as intensity" if recorded else "complex sum"
            print(f"\n   Keys at different angles, {label}: 9 pages, each using {share:.2%} of the plate's"
                  f" spatial frequencies,\n   read back with a largest relative error of {err:.1e}")


if __name__ == "__main__":
    main()
