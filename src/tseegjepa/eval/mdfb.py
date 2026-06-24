"""Post-hoc MDFB-style analysis for motor-imagery representations.

The shared decoder remains zero-shot: test-subject labels are never used to fit
the encoder, decoder, preprocessing, or checkpoints. Labels from the first two
acquisition runs are used only after evaluation to compute an MDFB-like
diagnostic and compare it with label-free decoder saliency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import collate_variable_montage
from ..train.pretrain import move


def _channel_index(ch_names, name):
    lookup = {ch.upper(): i for i, ch in enumerate(ch_names)}
    return lookup.get(name.upper())


def _laplacian(X, ch_names, center, neighbours):
    center_idx = _channel_index(ch_names, center)
    neighbour_idx = [_channel_index(ch_names, name) for name in neighbours]
    if center_idx is None:
        return None
    valid = [idx for idx in neighbour_idx if idx is not None]
    if not valid:
        return X[:, center_idx]
    return len(valid) * X[:, center_idx] - X[:, valid].sum(1)


def _corr_by_column(features, labels):
    x = features - features.mean(0, keepdims=True)
    y = labels.astype(np.float64) - labels.mean()
    denom = np.sqrt((x * x).sum(0) * (y * y).sum())
    return np.divide(
        (x * y[:, None]).sum(0),
        denom,
        out=np.zeros(features.shape[1], dtype=np.float64),
        where=denom > 0,
    )


def _contiguous_band(freqs, score, threshold_fraction):
    peak_idx = int(np.argmax(score))
    threshold = float(score[peak_idx]) * threshold_fraction
    left = right = peak_idx
    while left > 0 and score[left - 1] >= threshold:
        left -= 1
    while right + 1 < len(score) and score[right + 1] >= threshold:
        right += 1
    step = float(np.median(np.diff(freqs))) if len(freqs) > 1 else 0.5
    return (
        float(max(0.0, freqs[left] - step / 2)),
        float(freqs[right] + step / 2),
        float(freqs[peak_idx]),
    )


def mdfb_like(
    X,
    y,
    ch_names,
    sample_rate,
    fmin=5.0,
    fmax=35.0,
    bin_width=0.25,
):
    """Approximate the published MDFB heuristic on calibration trials."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    left = _laplacian(X, ch_names, "C3", ("FC3", "CP3", "C5", "C1"))
    right = _laplacian(X, ch_names, "C4", ("FC4", "CP4", "C6", "C2"))
    signals = [signal for signal in (left, right) if signal is not None]
    if not signals:
        raise ValueError("MDFB analysis requires C3 and/or C4")

    fft_freqs = np.fft.rfftfreq(X.shape[-1], 1.0 / sample_rate)
    centers = np.arange(fmin, fmax + bin_width / 2, bin_width)
    channel_scores = []
    for signal in signals:
        power = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
        features = []
        for center in centers:
            mask = np.abs(fft_freqs - center) <= bin_width / 2
            if not mask.any():
                mask[np.argmin(np.abs(fft_freqs - center))] = True
            features.append(np.log(power[:, mask].mean(-1) + 1e-12))
        channel_scores.append(_corr_by_column(np.stack(features, 1), y))
    channel_scores = np.stack(channel_scores)
    reference = int(np.argmax(channel_scores.sum(0)))
    signs = np.where(channel_scores[:, reference] >= 0, 1.0, -1.0)
    score = (channel_scores * signs[:, None]).sum(0)
    low, high, peak = _contiguous_band(centers, score, 0.05)
    return {
        "low_hz": low,
        "high_hz": high,
        "peak_hz": peak,
        "frequencies_hz": centers.tolist(),
        "score": (score / max(float(score.max()), 1e-12)).tolist(),
    }


def decoder_spectral_saliency(
    model,
    dataset,
    decoder_artifact,
    device,
    pool,
    sample_rate,
    fmin=5.0,
    fmax=35.0,
    bin_width=0.5,
    max_trials=80,
):
    """Label-free frequency sensitivity of the frozen encoder/shared decoder."""
    loader = DataLoader(
        dataset,
        batch_size=min(16, max_trials),
        shuffle=False,
        collate_fn=collate_variable_montage,
    )
    mu = decoder_artifact["feature_mean"].to(device)
    sd = decoder_artifact["feature_std"].to(device)
    comp = decoder_artifact["pca_components"]
    comp = comp.to(device) if comp is not None else None
    weight = decoder_artifact["weight"].to(device)
    bias = decoder_artifact["bias"].to(device)
    accumulated = None
    used = 0
    model.eval()
    for batch in loader:
        if used >= max_trials:
            break
        b = move(batch, device)
        remaining = max_trials - used
        if b["signal"].shape[0] > remaining:
            for key, value in list(b.items()):
                matches_batch = (
                    isinstance(value, torch.Tensor)
                    and value.ndim > 0
                    and value.shape[0] == b["signal"].shape[0]
                )
                if matches_batch:
                    b[key] = value[:remaining]
        signal = b["signal"].detach().requires_grad_(True)
        b["signal"] = signal
        features = model.encode_features(b, pool)
        z = (features - mu) / sd
        if comp is not None:
            z = z @ comp.t()
        logits = F.linear(z, weight, bias)
        top = logits.topk(min(2, logits.shape[-1]), dim=-1).values
        objective = (
            (top[:, 0] - top[:, 1]).sum()
            if top.shape[-1] == 2 else top[:, 0].sum()
        )
        grad = torch.autograd.grad(objective, signal)[0]
        xfft = torch.fft.rfft(signal.detach().float(), dim=-1)
        gfft = torch.fft.rfft(grad.detach().float(), dim=-1)
        contribution = (xfft * gfft.conj()).abs().mean((0, 1)).cpu()
        accumulated = (
            contribution if accumulated is None else accumulated + contribution
        )
        used += signal.shape[0]

    if accumulated is None:
        raise ValueError("no trials available for spectral saliency")
    fft_freqs = torch.fft.rfftfreq(
        dataset[0]["signal"].shape[-1], 1.0 / sample_rate
    )
    centers = torch.arange(fmin, fmax + bin_width / 2, bin_width)
    score = []
    for center in centers:
        mask = (fft_freqs - center).abs() <= bin_width / 2
        if not mask.any():
            mask[(fft_freqs - center).abs().argmin()] = True
        score.append(accumulated[mask].mean())
    score = torch.stack(score).numpy()
    freqs = centers.numpy()
    low, high, peak = _contiguous_band(freqs, score, 0.5)
    return {
        "low_hz": low,
        "high_hz": high,
        "peak_hz": peak,
        "frequencies_hz": freqs.tolist(),
        "score": (score / max(float(score.max()), 1e-12)).tolist(),
        "n_trials": used,
    }


def compare_bands(mdfb, learned):
    intersection = max(
        0.0,
        min(mdfb["high_hz"], learned["high_hz"])
        - max(mdfb["low_hz"], learned["low_hz"]),
    )
    union = max(mdfb["high_hz"], learned["high_hz"]) - min(
        mdfb["low_hz"], learned["low_hz"]
    )
    peak = learned["peak_hz"]
    distance = (
        mdfb["low_hz"] - peak
        if peak < mdfb["low_hz"]
        else peak - mdfb["high_hz"]
        if peak > mdfb["high_hz"]
        else 0.0
    )
    return {
        "learned_peak_in_mdfb": mdfb["low_hz"] <= peak <= mdfb["high_hz"],
        "peak_distance_hz": float(distance),
        "band_iou": float(intersection / union) if union > 0 else 0.0,
    }
