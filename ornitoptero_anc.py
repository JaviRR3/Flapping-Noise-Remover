#!/usr/bin/env python3
"""
Ornitoptero - limpieza del audio ambiente eliminando el ruido del aleteo.

Que hace (y por que asi, segun los datos de sensors_000078):
  * La RPM del CSV llega a ~2 muestras/s y alterna valores altos/bajos (aliasing de la
    oscilacion de RPM dentro de cada aleteo). Su MEDIA movil si es fiable: media/33.6
    coincide con la frecuencia de aleteo medida en el audio (<1.5 %).
  * El mic de referencia casi no tiene coherencia con el ambiente (~0.05-0.14), asi que
    NO se usa para restar forma de onda; solo se informa de la coherencia.
  * El ruido dominante es de banda ancha y se repite en cada ciclo de aleteo (~5.7 Hz):
    se estima su plantilla por ciclo (mediana de ciclos vecinos) y se resta.
  * Puerta de RPM: donde el motor esta parado la salida es EXACTAMENTE el original.

Uso:  python ornitoptero_anc.py  [--ambiente A.wav] [--ref R.wav] [--csv C.csv] [--out DIR]
Requiere: numpy, scipy, pandas
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import signal, ndimage
from scipy.io import wavfile

# ------------------------------------------------------------------ tren de engranajes
Z1, Z2, Z3, Z4 = 15, 72, 8, 56
I1, I2 = Z2 / Z1, Z4 / Z3           # 4.8 y 7
ITOT = I1 * I2                      # 33.6
RATIOS = {                          # frecuencia = ratio * f_motor
    "eje motor": 1.0, "eje intermedio": 1 / I1, "eje salida (aleteo)": 1 / ITOT,
    "engrane Z1-Z2": float(Z1), "engrane Z3-Z4": Z3 / I1,
}


# ------------------------------------------------------------------ E/S
def read_wav(path):
    fs, x = wavfile.read(path)
    if x.dtype == np.int16:   x = x.astype(np.float32) / 32768
    elif x.dtype == np.int32: x = x.astype(np.float32) / 2**31
    elif x.dtype == np.uint8: x = (x.astype(np.float32) - 128) / 128
    else:                     x = x.astype(np.float32)
    return fs, x


def write_wav(path, fs, x):
    wavfile.write(path, fs, (np.clip(x, -1, 1) * 32767).astype(np.int16))


# ------------------------------------------------------------------ RPM
class Rpm:
    """RPM en rejilla de 100 Hz: cruda (para la puerta) y media movil (para el aleteo)."""

    def __init__(self, csv, tcol, rcol, offset, dur, smooth_s, rpm_min, margin_s):
        df = pd.read_csv(csv)
        t = df[tcol].to_numpy(float) + offset
        r = df[rcol].to_numpy(float)
        ok = ~np.isnan(t) & ~np.isnan(r)
        tu, iu = np.unique(t[ok], return_index=True)      # el CSV repite marcas de tiempo
        ru = r[ok][iu]
        self.hz = 100.0
        self.tg = np.arange(0, dur, 1 / self.hz)
        raw = np.interp(self.tg, tu, ru, left=0, right=0)
        on = raw > rpm_min
        on = ndimage.binary_closing(on, np.ones(int(0.5 * self.hz)))
        on = ndimage.binary_dilation(on, np.ones(int(2 * margin_s * self.hz) + 1))
        self.on = on
        k = np.ones(int(smooth_s * self.hz))
        num = np.convolve(raw * (raw > rpm_min), k, "same")
        den = np.convolve((raw > rpm_min).astype(float), k, "same")
        self.rpm = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)   # media movil solo con motor

    def fout(self, t):  # Hz de aleteo
        return np.interp(t, self.tg, self.rpm) / 60 / ITOT

    def fm(self, t):
        return np.interp(t, self.tg, self.rpm) / 60

    def segments(self):
        d = np.diff(np.r_[0, self.on.astype(int), 0])
        return [(a / self.hz, b / self.hz) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


# ------------------------------------------------------------------ diagnosticos
def coherence_report(x, ref, fs, t0, t1):
    a, b = int(t0 * fs), int(t1 * fs)
    xs, ys = x[a:b], ref[a:b]
    f, C = signal.coherence(xs, ys, fs, nperseg=4096)
    bands = [(50, 500), (500, 2000), (2000, 5000), (5000, 10000)]
    vals = [float(C[(f >= lo) & (f < hi)].mean()) for lo, hi in bands]
    print("Coherencia ambiente-referencia (sin alinear):",
          ", ".join(f"{lo}-{hi} Hz: {v:.2f}" for (lo, hi), v in zip(bands, vals)))
    if max(vals) < 0.3:
        print("  -> coherencia baja: la referencia no puede restar ruido de banda ancha; no se usa.")


def tone_check(x, fs, rpm, t0, t1):
    """Pico sobre fondo en los armonicos de cada fuente, siguiendo la RPM media."""
    nper = 8192
    f, tt, S = signal.spectrogram(x, fs, nperseg=nper, noverlap=nper * 3 // 4)
    P = 10 * np.log10(S + 1e-20)
    df = f[1] - f[0]
    sel = np.where((tt > t0) & (tt < t1))[0][::4]
    print("Prueba de tonos de engranaje (pico - fondo en dB; ~5 dB = solo ruido):")
    for name, ratio in RATIOS.items():
        row = []
        for h in (1, 2, 3):
            dev = []
            for j in sel:
                f0 = ratio * h * rpm.fm(tt[j])
                i0 = int(round(f0 / df))
                if f0 < 60 or f0 > 20000 or i0 < 35:
                    continue
                dev.append(P[i0 - 2:i0 + 3, j].max() - np.r_[P[i0 - 30:i0 - 8, j], P[i0 + 8:i0 + 30, j]].mean())
            if dev:
                row.append(f"{h}x: {np.mean(dev):4.1f}")
        print(f"   {name:22s}" + "  ".join(row))


# ------------------------------------------------------------------ cancelacion sincrona
def _bp(x, lo, hi, fs, order=2):
    return signal.sosfiltfilt(signal.butter(order, [lo, hi], "band", fs=fs, output="sos"), x)


def cycle_marks(xs, fs, fout_guide):
    """Marcas de ciclo a partir de la fase de la fundamental de la envolvente."""
    dec = fs // 1000
    env = np.abs(_bp(xs, 1000, 8000, fs))
    env = signal.sosfiltfilt(signal.butter(2, 60, "low", fs=fs, output="sos"), env)[::dec]
    env -= env.mean()
    fe = fs / dec
    band = _bp(env, 0.7 * fout_guide, 1.3 * fout_guide, fe)
    ph = np.unwrap(np.angle(signal.hilbert(band))) / (2 * np.pi)
    ph = np.maximum.accumulate(ph)
    edge = int(1.5 * fe)
    lo, hi = ph[edge], ph[-edge]
    best, best_score = None, -np.inf
    for p0 in np.linspace(0, 1, 16, endpoint=False):
        cyc = np.arange(np.ceil(lo - p0), np.floor(hi - p0) + 1) + p0
        idx = np.interp(cyc, ph, np.arange(len(ph)))
        score = np.interp(idx, np.arange(len(env)), env).mean()
        if score > best_score:
            best, best_score = idx, score
    return np.round(best * dec).astype(int)


def sync_cancel(x, fs, rpm, ta, tb, K=15, shrink=2.0, depth=1.0):
    """Resta el ruido periodico por ciclo de aleteo en el tramo [ta, tb] (s). Devuelve (y, estimacion)."""
    ia, ib = int(ta * fs), int(tb * fs)
    xs = x[ia:ib].astype(np.float64)
    fg = float(np.median(rpm.fout(np.linspace(ta, tb, 200))))
    marks = cycle_marks(xs, fs, fg)
    Tm = int(np.median(np.diff(marks)))
    h = Tm
    Lw = 2 * h
    print(f"  tramo {ta:.1f}-{tb:.1f} s: f_aleteo guia {fg:.2f} Hz, {len(marks)} ciclos")

    bands = [(100, 500), (100, 1200), (200, 3000), (300, 6000), (300, 6000), (300, 6000)]
    S0, S1 = int(0.004 * fs), int(0.0008 * fs)
    ar = np.arange(-h, h)
    for q, (lo, hi) in enumerate(bands):                     # de banda estrecha a ancha
        xb = _bp(xs, lo, hi, fs)
        S = S0 if q < 4 else S1
        marks = marks[(marks - h - S >= 0) & (marks + h + S <= len(xs))]
        W = xb[marks[:, None] + ar[None, :]].T                # (Lw, nm)
        new = marks.copy()
        for k in range(len(marks)):
            cols = list(range(max(0, k - K), k)) + list(range(k + 1, min(len(marks), k + K + 1)))
            Tk = np.median(W[:, cols], axis=1)
            big = xb[marks[k] - h - S: marks[k] + h + S]
            c = signal.correlate(big, Tk, mode="valid", method="fft")
            new[k] = marks[k] + int(np.argmax(c)) - S
        marks = new
    keep = np.r_[True, np.diff(marks) > 0.5 * Tm]             # marcas ordenadas y separadas
    marks = marks[keep & (marks - h >= 0) & (marks + h <= len(xs))]

    w = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(Lw) / Lw)   # Hann periodica (COLA a 50 %)
    W = xs[marks[:, None] + ar[None, :]].T
    acc, nrm, gains = np.zeros(len(xs)), np.zeros(len(xs)), []
    for k in range(len(marks)):
        cols = list(range(max(0, k - K), k)) + list(range(k + 1, min(len(marks), k + K + 1)))
        Wn = W[:, cols]
        Tk = np.median(Wn, axis=1)
        se = 1.253 * 1.4826 * np.median(np.abs(Wn - Tk[:, None]), axis=1) / np.sqrt(len(cols))
        Tk = np.sign(Tk) * np.maximum(np.abs(Tk) - shrink * se, 0)       # limpia la plantilla
        ak = float(np.clip(np.sum(w * W[:, k] * Tk) / (np.sum(w * Tk * Tk) + 1e-12), 0, 1))
        gains.append(ak)
        sl = slice(marks[k] - h, marks[k] + h)
        acc[sl] += w * ak * Tk
        nrm[sl] += w
    est = np.where(nrm > 1e-3, acc / np.maximum(nrm, 1e-3), 0.0)
    print(f"  ganancia media por ciclo: {np.mean(gains):.2f}  (0 = sin parecido con la plantilla, 1 = identico)")
    y = x.copy()
    y[ia:ib] = xs - depth * est
    e = np.zeros_like(x)
    e[ia:ib] = est
    return y, e


# ------------------------------------------------------------------ notches opcionales
def tone_notches(x, fs, rpm, ta, tb, depth=0.8, bw=20.0, fmax_frac=0.45):
    """Muescas gaussianas en las frecuencias de eje/engrane y sus armonicos (STFT). OPCIONAL."""
    ia, ib = int(ta * fs), int(tb * fs)
    nper = 4096
    f, tt, Z = signal.stft(x[ia:ib], fs, nperseg=nper, noverlap=nper * 3 // 4)
    df, sig = f[1] - f[0], max(bw / 2.355, f[1] - f[0])
    fm = rpm.fm(ta + tt)
    orders = []
    for name, r in RATIOS.items():
        nh = 10 if r < 2 else int(fmax_frac * fs / (r * max(fm.min(), 50)))
        orders += [r * h for h in range(1, min(nh, 300) + 1)]
    orders = np.array(sorted(set(np.round(orders, 6))))
    M = np.ones(Z.shape)
    for j in range(Z.shape[1]):
        f0 = orders * fm[j]
        f0 = f0[(f0 > 0) & (f0 < fmax_frac * fs)]
        M[:, j] = 1 - depth * np.exp(-0.5 * ((f[:, None] - f0[None, :]) / sig) ** 2).max(axis=1)
    _, y = signal.istft(Z * M, fs, nperseg=nper, noverlap=nper * 3 // 4)
    out = x.copy()
    n = min(len(y), ib - ia)
    out[ia:ia + n] = y[:n]
    return out


# ------------------------------------------------------------------ main
def main():
    U = "/mnt/user-data/uploads/sensors_000078_20260929T142640Z"
    ap = argparse.ArgumentParser()
    ap.add_argument("--ambiente", default=U + "_2.wav")
    ap.add_argument("--ref", default=U + "_0.wav")
    ap.add_argument("--csv", default=U + ".csv")
    ap.add_argument("--out", default="/mnt/user-data/outputs")
    ap.add_argument("--time-col", default="mavlink_motor_rpm_monotonic_s")
    ap.add_argument("--rpm-col", default="mavlink_motor_rpm")
    ap.add_argument("--time-offset", type=float, default=0.0, help="s: reloj CSV -> reloj audio")
    ap.add_argument("--rpm-min", type=float, default=300, help="rpm: por debajo no se procesa nada")
    ap.add_argument("--rpm-smooth", type=float, default=2.0, help="s: media movil de la RPM")
    ap.add_argument("--depth", type=float, default=1.0, help="0..1 fraccion de plantilla que se resta")
    ap.add_argument("--k", type=int, default=15, help="ciclos vecinos por lado para la mediana")
    ap.add_argument("--notches", action="store_true", help="ademas, muescas en tonos de engranaje")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    fs, x = read_wav(a.ambiente)
    x = x[:, 0] if x.ndim > 1 else x
    dur = len(x) / fs
    rpm = Rpm(a.csv, a.time_col, a.rpm_col, a.time_offset, dur, a.rpm_smooth, a.rpm_min, margin_s=0.3)
    segs = [(max(s + 1.0, 0), min(e - 1.0, dur)) for s, e in rpm.segments() if e - s > 6]
    print(f"Audio {dur:.1f} s @ {fs} Hz | tramos con motor: {[(round(s,1), round(e,1)) for s,e in segs]}")
    print(f"Relacion total {ITOT:.1f}:1 | RPM media con motor {rpm.rpm[rpm.on].mean():.0f} -> "
          f"aleteo {rpm.rpm[rpm.on].mean()/60/ITOT:.2f} Hz")
    if not segs:
        raise SystemExit("No hay tramo con motor girando: revisa --time-offset / columnas del CSV")

    if os.path.exists(a.ref):
        fr, r = read_wav(a.ref)
        r = r.mean(axis=1) if r.ndim > 1 else r
        n = min(len(r), len(x))
        coherence_report(x[:n], r[:n], fs, segs[0][0] + 5, segs[0][1] - 5)
    tone_check(x, fs, rpm, segs[0][0] + 5, segs[0][1] - 5)

    y, est = x.copy(), np.zeros_like(x)
    for ta, tb in segs:
        y, e = sync_cancel(y, fs, rpm, ta, tb, K=a.k, depth=a.depth)
        est += e
        if a.notches:
            y = tone_notches(y, fs, rpm, ta, tb)

    # puerta de RPM: fuera del motor -> original; transicion de 100 ms
    gate = np.interp(np.arange(len(x)) / fs, rpm.tg, rpm.on.astype(float))
    gate = ndimage.uniform_filter1d(gate, int(0.1 * fs))
    y = (gate * y + (1 - gate) * x).astype(np.float32)

    # metricas en el tramo con motor
    A, B = int(segs[0][0] * fs), int(segs[0][1] * fs)
    print("\nReduccion de potencia con motor (dB) por banda:")
    for lo, hi in [(100, 1000), (1000, 3000), (3000, 8000), (8000, 20000)]:
        px = np.mean(_bp(x[A:B], lo, hi, fs) ** 2)
        py = np.mean(_bp(y[A:B], lo, hi, fs) ** 2)
        print(f"   {lo:5d}-{hi:5d} Hz: {10*np.log10(px/py):5.1f} dB")

    # salidas: nivel original conservado + version normalizada para escuchar
    peak = np.abs(y).max()
    write_wav(f"{a.out}/ambiente_limpio.wav", fs, y if peak < 0.99 else y * 0.99 / peak)
    p999 = np.percentile(np.abs(y), 99.9)
    write_wav(f"{a.out}/ambiente_limpio_fuerte.wav", fs, y / (p999 + 1e-9) * 0.8)
    write_wav(f"{a.out}/ruido_aleteo_estimado.wav", fs, est / (np.abs(est).max() + 1e-9) * 0.9)

    # figura comparativa (10 s con aleteo)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t0 = segs[0][0] + 20
    a0, b0 = int(t0 * fs), int((t0 + 10) * fs)
    fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for axi, sig_, ttl in zip(ax, (x[a0:b0], y[a0:b0]), ("Original", "Limpio")):
        f, tt, S = signal.spectrogram(sig_, fs, nperseg=2048, noverlap=1536)
        m = f < 12000
        axi.pcolormesh(tt + t0, f[m], 10 * np.log10(S[m] + 1e-14), shading="auto", vmin=-120, vmax=-50)
        axi.set_title(ttl); axi.set_ylabel("Hz")
    ax[1].set_xlabel("t (s)")
    plt.tight_layout(); plt.savefig(f"{a.out}/comparacion_espectrograma.png", dpi=80)
    print(f"\nListo. Archivos en {a.out}")


if __name__ == "__main__":
    main()
