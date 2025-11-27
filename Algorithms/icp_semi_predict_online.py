import numpy as np
import json
import os
import argparse
import vitaldb

"""
Online semi-Markov predictor with streaming exponential-decay reestimation.

This script loads a saved model JSON (from `icp_model_semi.py`) and processes a .vital
file sample-by-sample (or truncated via --max-samples). It updates transition counts and
sojourn weighted histograms online using a decay factor and reports prediction diagnostics.

Usage:
  python icp_semi_predict_online.py --vital <file.vital> --model-file <model.json> --decay-factor 0.999
"""


def discretize_icp(icp_values, thresholds=None):
    if thresholds is None:
        t1, t2 = 15.0, 20.0
    else:
        t1, t2 = float(thresholds[0]), float(thresholds[1])
    icp = np.asarray(icp_values, dtype=float).ravel()
    states = np.zeros_like(icp, dtype=int)
    states[icp < t1] = 0
    states[(icp >= t1) & (icp < t2)] = 1
    states[icp >= t2] = 2
    return states


def weighted_hist_update(hist: dict, duration: float, decay: float, bin_width: float = 1.0):
    """Decay existing histogram weights and add new observation at rounded bin."""
    # decay existing weights
    if decay < 1.0:
        for k in list(hist.keys()):
            hist[k] *= decay
            if hist[k] <= 1e-12:
                del hist[k]
    # add new
    b = int(round(duration / bin_width))
    hist[b] = hist.get(b, 0.0) + 1.0


def empirical_p_leave_from_hist(hist: dict, t: float, dt: float = 1.0, bin_width: float = 1.0):
    """Compute empirical conditional leave probability in interval (t, t+dt] using weighted hist bins."""
    if not hist:
        return 0.0
    # total weight
    total = sum(hist.values())
    if total <= 0:
        return 0.0
    # compute cumulative up to t and up to t+dt
    bin_t = int(round(t / bin_width))
    bin_t_dt = int(round((t + dt) / bin_width))
    c_t = sum(w for b, w in hist.items() if b <= bin_t)
    c_t_dt = sum(w for b, w in hist.items() if b <= bin_t_dt)
    denom = total - c_t
    if denom <= 0:
        return 0.0
    p_leave = (c_t_dt - c_t) / denom
    return float(np.clip(p_leave, 0.0, 1.0))


def run_online(vital_path, model_file, decay_factor=0.999, bin_width=1.0, max_samples=None):
    vf = vitaldb.VitalFile(vital_path)
    # load model
    if not os.path.exists(model_file):
        raise FileNotFoundError(model_file)
    with open(model_file, 'r') as fh:
        model = json.load(fh)
    P = np.asarray(model.get('P', []), dtype=float)
    thresholds = model.get('thresholds', [15.0, 20.0])
    counts = np.array(model.get('counts', np.zeros_like(P)), dtype=float)
    # sojourn histograms: keyed by state index string
    sojourns_hist = {}
    for k, lst in (model.get('sojourns') or {}).items():
        try:
            sojourns_hist[str(k)] = {int(round(d / bin_width)): 1.0 for d in lst}
        except Exception:
            sojourns_hist[str(k)] = {}

    # read icp
    res = vf.to_numpy('ICP', 0, return_timestamp=True)
    if isinstance(res, tuple) and len(res) >= 2:
        ts, icp = np.asarray(res[0]).ravel().astype(float), np.asarray(res[1]).ravel().astype(float)
    else:
        arr = np.asarray(res)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            ts, icp = arr[:, 0].astype(float), arr[:, 1].astype(float)
        else:
            icp = arr.ravel().astype(float)
            ts = np.arange(len(icp), dtype=float)

    # clean
    mask = (~np.isnan(icp)) & (icp >= 0.0) & (icp <= 50.0)
    icp = icp[mask]
    ts = ts[mask]
    if max_samples is not None and icp.size > max_samples:
        icp = icp[:max_samples]
        ts = ts[:max_samples]

    states = discretize_icp(icp, thresholds=thresholds, hysteresis=0.5)

    n_states = P.shape[0]
    correct = 0
    total = 0
    change_correct = 0
    change_total = 0

    run_start = 0
    for i in range(len(states) - 1):
        cur = int(states[i])
        nxt = int(states[i + 1])
        # compute duration so far in run
        if i > 0 and states[i] != states[i - 1]:
            run_start = i
        duration = float(ts[i] - ts[run_start]) if ts is not None else float(i - run_start + 1)

        # compute p_leave from hist
        hist = sojourns_hist.get(str(cur), {})
        p_leave = empirical_p_leave_from_hist(hist, duration, dt=1.0, bin_width=bin_width)
        stay = 1.0 - p_leave
        row = counts[cur, :].copy()
        off_total = np.sum(row) - row[cur]
        if off_total <= 0:
            # uniform distribution among others if no off-diagonal information
            probs = np.ones(n_states) / float(n_states)
        else:
            probs = np.zeros(n_states, dtype=float)
            probs[cur] = stay
            for j in range(n_states):
                if j == cur:
                    continue
                probs[j] = p_leave * (row[j] / off_total)

        pred = int(np.argmax(probs))
        if pred == nxt:
            correct += 1
            if nxt != cur:
                change_correct += 1
        if nxt != cur:
            change_total += 1
        total += 1

        # update counts with decay and observation if a transition happened
        # apply decay to all counts
        counts *= decay_factor
        if nxt != cur:
            counts[cur, nxt] += 1.0
            # update sojourn histogram for this state
            weighted_hist_update(sojourns_hist.setdefault(str(cur), {}), duration, decay_factor, bin_width=bin_width)

    overall = (correct / total * 100.0) if total > 0 else 0.0
    change_acc = (change_correct / change_total * 100.0) if change_total > 0 else 0.0
    print(f"Online predictor results: overall={overall:.3f}% change_points={change_acc:.3f}% ({change_correct}/{change_total})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vital', type=str, required=True)
    parser.add_argument('--model-file', type=str, required=True)
    parser.add_argument('--decay-factor', type=float, default=0.999, help='Multiplicative decay applied each observation to historical counts')
    parser.add_argument('--bin-width', type=float, default=1.0, help='Histogram bin width for sojourn durations')
    parser.add_argument('--max-samples', type=int, default=None, help='Limit samples for diagnostics')
    args = parser.parse_args()
    run_online(args.vital, args.model_file, decay_factor=args.decay_factor, bin_width=args.bin_width, max_samples=args.max_samples)
