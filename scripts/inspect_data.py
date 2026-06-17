"""Exploratory analysis of a MOABB EEG dataset -- understand it before training.

No model involved. Answers:
  * what's in it: trials, channels, classes, per-subject/session balance
  * signal sanity: amplitude scale, dead/noisy channels, NaNs
  * where the signal is: band-power per class, mu/beta lateralization (C3 vs C4)
  * how hard it is: classic-pipeline accuracy ceilings
      - log-bandpower + logistic regression (within-subject CV)
      - CSP + LDA (canonical MI baseline, left vs right hand)

Run:
  python scripts/inspect_data.py --subjects 1 2 3 --bands mu beta
  python scripts/inspect_data.py --subjects 1 --plots out/        # save figures
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from tseegjepa.data.moabb_eeg import load_moabb

BANDS = {
    "delta": (1, 4), "theta": (4, 8), "mu": (8, 13),
    "beta": (13, 30), "low_gamma": (30, 45),
}


# ----------------------------- helpers ------------------------------------
def _welch_bandpower(X, sfreq, band):
    """X (n, C, T) -> (n, C) mean power in [lo,hi] Hz via Welch PSD."""
    from scipy.signal import welch
    lo, hi = band
    nper = min(X.shape[-1], int(sfreq))
    f, pxx = welch(X, fs=sfreq, nperseg=nper, axis=-1)
    m = (f >= lo) & (f <= hi)
    return pxx[..., m].mean(-1)


def _section(title):
    print("\n" + "=" * 68 + f"\n{title}\n" + "=" * 68)


# ----------------------------- analyses -----------------------------------
def summarize(X, y, meta, ch_names, label_names, sfreq):
    _section("DATASET SUMMARY")
    n, C, T = X.shape
    print(f"trials={n}  channels={C}  samples/trial={T}  sfreq={sfreq}Hz  "
          f"({T/sfreq:.2f}s)")
    print(f"classes ({len(label_names)}): {label_names}")
    print(f"channels: {ch_names}")

    print("\nclass balance:")
    cnt = Counter(int(v) for v in y)
    for i, name in enumerate(label_names):
        print(f"  {name:12s} {cnt.get(i,0):4d}  ({cnt.get(i,0)/n:.1%})")

    subs = Counter(m["subject"] for m in meta)
    sess = Counter(m["session"] for m in meta)
    print(f"\nsubjects: {dict(sorted(subs.items()))}")
    print(f"sessions: {dict(sorted(sess.items()))}")

    if any("sex" in m for m in meta):
        print("\ndemographics (per subject):")
        seen = {}
        for m in meta:
            seen.setdefault(m["subject"],
                            (m.get("sex", "?"), m.get("age", "?"), m.get("hand", "?")))
        print(f"  {'subj':>4s} {'sex':>4s} {'age':>4s} {'hand':>5s}")
        for s in sorted(seen):
            sx, ag, hd = seen[s]
            print(f"  {s:>4d} {str(sx):>4s} {str(ag):>4s} {str(hd):>5s}")
        sx_cnt = Counter(v[0] for v in seen.values())
        print(f"  sex balance across subjects: {dict(sx_cnt)}")


def signal_sanity(X, ch_names):
    _section("SIGNAL SANITY")
    print(f"global: mean={X.mean():.3g} std={X.std():.3g} "
          f"min={X.min():.3g} max={X.max():.3g}  NaNs={np.isnan(X).sum()}")
    ch_std = X.std(axis=(0, 2))                      # per channel
    order = np.argsort(ch_std)
    print("\nquietest 3 channels (possible dead):")
    for i in order[:3]:
        print(f"  {ch_names[i]:6s} std={ch_std[i]:.3g}")
    print("loudest 3 channels (possible noisy):")
    for i in order[-3:][::-1]:
        print(f"  {ch_names[i]:6s} std={ch_std[i]:.3g}")


def bandpower_by_class(X, y, label_names, sfreq, bands):
    _section("LOG BAND-POWER BY CLASS  (mean over channels+trials)")
    hdr = "band       " + "".join(f"{n[:10]:>12s}" for n in label_names)
    print(hdr)
    for b in bands:
        bp = np.log(_welch_bandpower(X, sfreq, BANDS[b]) + 1e-12)  # (n,C)
        row = f"{b:10s} "
        for ci in range(len(label_names)):
            row += f"{bp[y == ci].mean():12.3f}"
        print(row)
    print("\n(differences across columns within a row = class-discriminative band)")


def lateralization(X, y, meta, ch_names, label_names, sfreq):
    """Contralateral mu/beta ERD: right-hand MI suppresses LEFT-cortex (C3)."""
    _section("LATERALIZATION  (mu+beta power, C3 vs C4)")
    names_u = [c.upper() for c in ch_names]
    if "C3" not in names_u or "C4" not in names_u:
        print("C3/C4 not in montage -> skip"); return
    c3, c4 = names_u.index("C3"), names_u.index("C4")
    bp = np.log(_welch_bandpower(X, sfreq, (8, 30)) + 1e-12)   # (n,C)
    print(f"{'class':12s}{'C3(left)':>12s}{'C4(right)':>12s}{'C3-C4':>10s}")
    for ci, name in enumerate(label_names):
        m = y == ci
        v3, v4 = bp[m, c3].mean(), bp[m, c4].mean()
        print(f"{name:12s}{v3:12.3f}{v4:12.3f}{v3-v4:10.3f}")
    print("\nexpect left_hand: C3>C4 (right-cortex ERD) ; right_hand: C3<C4."
          "\nif C3-C4 flips sign between hands -> real MI signal present.")


def separability(X, y, meta, label_names, sfreq, bands):
    _section("DIFFICULTY CEILING  (classic pipelines, within-subject)")
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        print("sklearn unavailable -> skip"); return

    # feature: log band-power per channel, concatenated over chosen bands
    feats = [np.log(_welch_bandpower(X, sfreq, BANDS[b]) + 1e-12) for b in bands]
    F = np.concatenate(feats, axis=1)                # (n, C*len(bands))
    subj = np.array([m["subject"] for m in meta])

    chance = 1.0 / len(label_names)
    print(f"feature = log-bandpower [{','.join(bands)}], {F.shape[1]} dims | "
          f"chance={chance:.3f}")
    print(f"\n{'subject':>8s}{'logreg 5-fold acc':>20s}")
    accs = []
    for s in sorted(set(subj)):
        m = subj == s
        if m.sum() < 10:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=500, C=0.5))
        sc = cross_val_score(clf, F[m], y[m], cv=5)
        accs.append(sc.mean())
        print(f"{s:8d}{sc.mean():14.3f} +-{sc.std():.3f}")
    if accs:
        print(f"\nmean within-subject ceiling: {np.mean(accs):.3f} "
              f"(chance {chance:.3f})")
        print("-> THIS is roughly the best a frozen linear probe can hope for.")
        print("   if your JEPA probe is far below this, the encoder is the issue;")
        print("   if it's near this, the task/montage is just this hard.")


def csp_lda(X, y, meta, label_names, sfreq):
    _section("CSP + LDA  (canonical MI baseline, left vs right hand)")
    try:
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import make_pipeline
    except Exception:
        print("mne/sklearn unavailable -> skip"); return
    lut = {n: i for i, n in enumerate(label_names)}
    if "left_hand" not in lut or "right_hand" not in lut:
        print("needs left_hand/right_hand classes -> skip"); return
    lh, rh = lut["left_hand"], lut["right_hand"]
    subj = np.array([m["subject"] for m in meta])
    print(f"{'subject':>8s}{'CSP+LDA acc':>16s}  (chance 0.500)")
    accs = []
    for s in sorted(set(subj)):
        m = (subj == s) & ((y == lh) | (y == rh))
        if m.sum() < 20:
            continue
        yy = (y[m] == rh).astype(int)
        clf = make_pipeline(CSP(n_components=6, log=True),
                            LinearDiscriminantAnalysis())
        try:
            sc = cross_val_score(clf, X[m].astype(np.float64), yy, cv=5)
        except Exception as e:
            print(f"{s:8d}   failed: {e}"); continue
        accs.append(sc.mean())
        print(f"{s:8d}{sc.mean():14.3f}")
    if accs:
        print(f"\nmean CSP+LDA (binary): {np.mean(accs):.3f}  "
              "-> the strong classical reference for left/right MI.")


def plots(X, y, label_names, sfreq, outdir):
    try:
        import os
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.signal import welch
    except Exception:
        print("matplotlib/scipy unavailable -> skip plots"); return
    os.makedirs(outdir, exist_ok=True)
    f, pxx = welch(X, fs=sfreq, nperseg=min(X.shape[-1], int(sfreq)), axis=-1)
    pxx = pxx.mean(1)                                # avg channels -> (n, F)
    plt.figure(figsize=(7, 4))
    for ci, name in enumerate(label_names):
        plt.semilogy(f, pxx[y == ci].mean(0), label=name)
    plt.xlim(0, 45); plt.xlabel("Hz"); plt.ylabel("PSD"); plt.legend()
    plt.title("class-mean PSD"); plt.tight_layout()
    p = os.path.join(outdir, "psd_by_class.png"); plt.savefig(p, dpi=120)
    print(f"saved {p}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="BNCI2014_001")
    ap.add_argument("--paradigm", default="MotorImagery")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1])
    ap.add_argument("--sample-rate", type=int, default=128)
    ap.add_argument("--bands", nargs="+", default=["mu", "beta"],
                    choices=list(BANDS))
    ap.add_argument("--plots", default=None, help="dir to save figures")
    ap.add_argument("--no-csp", action="store_true")
    a = ap.parse_args()

    print(f"loading {a.dataset} subjects={a.subjects} ...")
    X, y, meta, ch_names, label_names = load_moabb(
        a.dataset, subjects=a.subjects, paradigm_name=a.paradigm,
        sample_rate=a.sample_rate,
    )
    sfreq = a.sample_rate

    summarize(X, y, meta, ch_names, label_names, sfreq)
    signal_sanity(X, ch_names)
    bandpower_by_class(X, y, label_names, sfreq, a.bands)
    lateralization(X, y, meta, ch_names, label_names, sfreq)
    separability(X, y, meta, label_names, sfreq, a.bands)
    if not a.no_csp:
        csp_lda(X, y, meta, label_names, sfreq)
    if a.plots:
        plots(X, y, label_names, sfreq, a.plots)
    print("\ndone.")


if __name__ == "__main__":
    main()
