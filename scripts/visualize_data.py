"""Visualize a MOABB EEG dataset -> PNG figures (no GUI needed).

Produces, into --out:
  1. raw_traces.png     stacked channel traces, one example trial per class
  2. psd_by_class.png   class-mean power spectrum (where the signal lives)
  3. bandpower_heatmap.png   channel x class log band-power (per band)
  4. topomap_<band>.png      scalp maps of band-power per class (needs mne)
  5. lateralization.png      C3/C4 mu+beta power per class (the MI signal)
  6. embedding.png      PCA(+optional t-SNE) of band-power features, colored by class
  7. spectrogram.png    time-frequency of one trial

Run:
  python scripts/visualize_data.py --subjects 1 --out figs
  python scripts/visualize_data.py --subjects 1 --bands mu beta --tsne
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from tseegjepa.data.moabb_eeg import load_moabb

BANDS = {"delta": (1, 4), "theta": (4, 8), "mu": (8, 13),
         "beta": (13, 30), "low_gamma": (30, 45)}


def _bandpower(X, sfreq, band):
    from scipy.signal import welch
    lo, hi = band
    f, pxx = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(sfreq)), axis=-1)
    return pxx[..., (f >= lo) & (f <= hi)].mean(-1)        # (n, C)


def _mne_info(ch_names, sfreq):
    import mne
    info = mne.create_info(list(ch_names), sfreq, ch_types="eeg")
    info.set_montage("standard_1020", match_case=False, on_missing="ignore")
    return info


# ------------------------------- plots ------------------------------------
def raw_traces(X, y, ch_names, sfreq, label_names, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(label_names), figsize=(4 * len(label_names), 6),
                             sharey=True)
    axes = np.atleast_1d(axes)
    t = np.arange(X.shape[-1]) / sfreq
    off = 6
    for ax, ci in zip(axes, range(len(label_names))):
        idx = np.where(y == ci)[0][0]
        x = X[idx]
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-9)
        for c in range(x.shape[0]):
            ax.plot(t, x[c] + c * off, lw=0.5)
        ax.set_title(label_names[ci]); ax.set_xlabel("s")
        ax.set_yticks([c * off for c in range(len(ch_names))])
        if ci == 0:
            ax.set_yticklabels(ch_names, fontsize=6)
    fig.suptitle("raw traces (one trial/class, per-channel z-scored)")
    fig.tight_layout(); _save(fig, out, "raw_traces.png")


def psd_by_class(X, y, sfreq, label_names, out):
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    f, pxx = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(sfreq)), axis=-1)
    pxx = pxx.mean(1)                                       # avg channels
    fig, ax = plt.subplots(figsize=(7, 4))
    for ci, name in enumerate(label_names):
        ax.semilogy(f, pxx[y == ci].mean(0), label=name)
    for b, (lo, hi) in BANDS.items():
        ax.axvspan(lo, hi, alpha=0.05, color="k")
    ax.set_xlim(0, 45); ax.set_xlabel("Hz"); ax.set_ylabel("PSD")
    ax.legend(); ax.set_title("class-mean PSD (channel-averaged)")
    fig.tight_layout(); _save(fig, out, "psd_by_class.png")


def bandpower_heatmap(X, y, ch_names, sfreq, label_names, bands, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(bands), figsize=(3.2 * len(bands), 6))
    axes = np.atleast_1d(axes)
    for ax, b in zip(axes, bands):
        bp = np.log(_bandpower(X, sfreq, BANDS[b]) + 1e-12)    # (n,C)
        M = np.stack([bp[y == ci].mean(0) for ci in range(len(label_names))])  # (cls,C)
        M = M - M.mean(0, keepdims=True)                       # contrast vs mean class
        im = ax.imshow(M, aspect="auto", cmap="RdBu_r")
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names, rotation=90, fontsize=6)
        ax.set_yticks(range(len(label_names))); ax.set_yticklabels(label_names)
        ax.set_title(f"{b} log-power (class - mean)")
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.suptitle("per-channel band-power contrast by class (red=high)")
    fig.tight_layout(); _save(fig, out, "bandpower_heatmap.png")


def topomaps(X, y, ch_names, sfreq, label_names, bands, out):
    try:
        import matplotlib.pyplot as plt
        import mne
    except Exception:
        print("mne/matplotlib missing -> skip topomaps"); return
    info = _mne_info(ch_names, sfreq)
    picks = [i for i, ch in enumerate(info.ch_names)
             if info.get_montage() and ch in info.get_montage().ch_names]
    for b in bands:
        bp = np.log(_bandpower(X, sfreq, BANDS[b]) + 1e-12)
        fig, axes = plt.subplots(1, len(label_names),
                                 figsize=(3 * len(label_names), 3))
        axes = np.atleast_1d(axes)
        vals = [bp[y == ci].mean(0) for ci in range(len(label_names))]
        vmin = min(v.min() for v in vals); vmax = max(v.max() for v in vals)
        for ax, ci, name in zip(axes, range(len(label_names)), label_names):
            try:
                mne.viz.plot_topomap(vals[ci], info, axes=ax, show=False,
                                     vlim=(vmin, vmax), contours=4)
            except Exception as e:
                ax.set_title(f"fail: {e}"); continue
            ax.set_title(name)
        fig.suptitle(f"{b} band-power topography")
        fig.tight_layout(); _save(fig, out, f"topomap_{b}.png")


def lateralization(X, y, ch_names, sfreq, label_names, out):
    import matplotlib.pyplot as plt
    u = [c.upper() for c in ch_names]
    if "C3" not in u or "C4" not in u:
        print("no C3/C4 -> skip lateralization"); return
    c3, c4 = u.index("C3"), u.index("C4")
    bp = np.log(_bandpower(X, sfreq, (8, 30)) + 1e-12)
    v3 = [bp[y == ci, c3].mean() for ci in range(len(label_names))]
    v4 = [bp[y == ci, c4].mean() for ci in range(len(label_names))]
    x = np.arange(len(label_names)); w = 0.38
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, v3, w, label="C3 (left cortex)")
    ax.bar(x + w / 2, v4, w, label="C4 (right cortex)")
    ax.set_xticks(x); ax.set_xticklabels(label_names)
    ax.set_ylabel("log mu+beta power"); ax.legend()
    ax.set_title("lateralization: hands should flip C3 vs C4")
    fig.tight_layout(); _save(fig, out, "lateralization.png")


def embedding(X, y, sfreq, label_names, bands, out, tsne):
    import matplotlib.pyplot as plt
    F = np.concatenate([np.log(_bandpower(X, sfreq, BANDS[b]) + 1e-12)
                        for b in bands], axis=1)
    F = (F - F.mean(0)) / (F.std(0) + 1e-9)
    from sklearn.decomposition import PCA
    methods = [("PCA", PCA(n_components=2).fit_transform(F))]
    if tsne:
        try:
            from sklearn.manifold import TSNE
            methods.append(("t-SNE", TSNE(n_components=2, perplexity=30,
                                          init="pca").fit_transform(F)))
        except Exception as e:
            print("t-SNE skip:", e)
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))
    axes = np.atleast_1d(axes)
    for ax, (name, Z) in zip(axes, methods):
        for ci, lab in enumerate(label_names):
            m = y == ci
            ax.scatter(Z[m, 0], Z[m, 1], s=10, alpha=0.6, label=lab)
        ax.set_title(f"{name} of log-bandpower"); ax.legend(fontsize=8)
    fig.suptitle("trial embedding (separated clusters = learnable)")
    fig.tight_layout(); _save(fig, out, "embedding.png")


def spectrogram(X, y, ch_names, sfreq, label_names, out):
    import matplotlib.pyplot as plt
    from scipy.signal import spectrogram as spec
    u = [c.upper() for c in ch_names]
    ch = u.index("C3") if "C3" in u else 0
    fig, axes = plt.subplots(1, len(label_names),
                             figsize=(4 * len(label_names), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, ci, name in zip(axes, range(len(label_names)), label_names):
        idx = np.where(y == ci)[0][0]
        f, t, Sxx = spec(X[idx, ch], fs=sfreq, nperseg=int(sfreq * 0.5))
        ax.pcolormesh(t, f, np.log(Sxx + 1e-12), shading="auto")
        ax.set_ylim(0, 45); ax.set_title(f"{name} @ {ch_names[ch]}")
        ax.set_xlabel("s")
    axes[0].set_ylabel("Hz")
    fig.suptitle(f"spectrogram, channel {ch_names[ch]}")
    fig.tight_layout(); _save(fig, out, "spectrogram.png")


def _save(fig, out, name):
    import matplotlib.pyplot as plt
    p = os.path.join(out, name)
    fig.savefig(p, dpi=120); plt.close(fig)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--paradigm", default="MotorImagery")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--sample-rate", type=int, default=128)
    ap.add_argument("--bands", nargs="+", default=["mu", "beta"], choices=list(BANDS))
    ap.add_argument("--out", default="figs")
    ap.add_argument("--tsne", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")

    print(f"loading {a.dataset} subjects={a.subjects} ...")
    X, y, meta, ch_names, label_names = load_moabb(
        a.dataset, subjects=a.subjects, paradigm_name=a.paradigm,
        sample_rate=a.sample_rate,
    )
    sr = a.sample_rate
    print(f"X={X.shape}  classes={label_names}")

    raw_traces(X, y, ch_names, sr, label_names, a.out)
    psd_by_class(X, y, sr, label_names, a.out)
    bandpower_heatmap(X, y, ch_names, sr, label_names, a.bands, a.out)
    topomaps(X, y, ch_names, sr, label_names, a.bands, a.out)
    lateralization(X, y, ch_names, sr, label_names, a.out)
    embedding(X, y, sr, label_names, a.bands, a.out, a.tsne)
    spectrogram(X, y, ch_names, sr, label_names, a.out)
    print(f"\nall figures -> {a.out}/")


if __name__ == "__main__":
    main()
