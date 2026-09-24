#!/usr/bin/env python3
"""C-spike v3: calibrated per-query confidence from cheap features.

Given per_query outputs from headtohead matrices, train a lightweight
regression from cheap-at-query-time features to per-query nDCG@10. The
goal isn't to beat Qdrant on retrieval — it's to prove ANNex can *tell
you which of its own answers are trustworthy*, which is the primitive
under RecallGuard / Search-SLO / adaptive-escalation.

Feature set:
  top_score, top_minus_second, top_score * top_minus_second, log(top_score),
  log(top_minus_second), returned (count of hits returned)

Model: isotonic regression on a single scalar signal + linear regression
on the full feature bag. Isotonic gives us the calibrated 0..1 confidence
number; linear gives us the multivariate baseline we compare against.

Evaluation: leave-one-corpus-out. Train on two corpora, test on the third.
Report per-fold R², spearman rank corr, and calibration curve buckets. If
those hold cross-corpus, the primitive is portable and real.

Usage:
    .venv/bin/python benchmark/confidence_calibration.py \\
        benchmark/reports/headtohead-fiqa-v3-matrix.json \\
        benchmark/reports/headtohead-scifact-v3-matrix.json \\
        benchmark/reports/headtohead-nfcorpus-v3-matrix.json \\
        --candidates 250
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_rows(matrix_path: Path, candidates: int):
    data = json.loads(matrix_path.read_text())
    dataset = data.get("dataset", str(matrix_path))
    key = f"annex_multivector_c{candidates}"
    sys = data["systems"].get(key)
    if sys is None:
        # fallback: any annex system if only one exists
        for name, s in data["systems"].items():
            if name.startswith("annex"):
                sys = s
                key = name
                break
    if sys is None:
        raise RuntimeError(f"no annex system in {matrix_path}")
    return dataset, key, sys.get("per_query", [])


def rank_percentile(x):
    x = np.asarray(x, dtype=np.float64)
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def build_features(rows, quantile: bool = False, quantile_label: bool = False):
    """Return (X, y) where X is (n, num_features) and y is (n,) per-query nDCG.

    If `quantile=True`, raw features are replaced by their rank-percentile
    within this batch (0..1). Quantile features are dimensionless and
    corpus-invariant, so a model trained on one corpus can be applied to
    another without recalibration — at the cost of losing absolute
    calibration.

    If `quantile_label=True`, the label (nDCG) is ALSO replaced by its
    rank-percentile within the batch. This drops the absolute-nDCG target
    entirely and instead predicts 'is this query in the top-X% of quality
    for this batch?' — which is what a relative confidence output actually
    needs. Both feature and label distributions become identical [0,1]
    uniform across all corpora, removing the label-distribution shift.
    """
    xs = []
    ys = []
    for r in rows:
        top = r.get("top_score", 0.0)
        margin = r.get("top_minus_second", 0.0)
        returned = r.get("returned", 0)
        features = [
            top,
            margin,
            top * margin,
            math.log(max(top, 1e-6)),
            math.log(max(margin, 1e-6)),
            returned,
        ]
        xs.append(features)
        ys.append(r.get("ndcg@10", 0.0))
    X = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if quantile:
        X_q = np.empty_like(X)
        for j in range(X.shape[1]):
            X_q[:, j] = rank_percentile(X[:, j])
        X = X_q
    if quantile_label:
        y = rank_percentile(y)
    return X, y


def pearson(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    if len(a) < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def linear_fit(X, y):
    """Closed-form OLS with intercept. Returns (coefs including intercept, predict fn)."""
    X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
    # Guard against singular via lstsq
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)

    def predict(X_new):
        X_aug_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
        return X_aug_new @ beta

    return beta, predict


def isotonic_fit(x, y):
    """Pool-Adjacent-Violators isotonic regression on a single scalar. Returns
    a piecewise-constant nondecreasing fn — predicts y from x by interpolating
    the fitted step curve. Small self-contained implementation so we don't
    pull in scikit-learn just for this."""
    order = np.argsort(x)
    x_s = x[order]
    y_s = y[order]
    n = len(x_s)
    weights = np.ones(n)
    values = y_s.copy()
    i = 0
    while i < n - 1:
        if values[i] > values[i + 1]:
            # Merge blocks i and i+1
            total_w = weights[i] + weights[i + 1]
            merged = (values[i] * weights[i] + values[i + 1] * weights[i + 1]) / total_w
            values[i] = merged
            weights[i] = total_w
            # Delete i+1
            values = np.delete(values, i + 1)
            weights = np.delete(weights, i + 1)
            x_s = np.delete(x_s, i + 1)
            n -= 1
            # Rewind so we can re-check
            if i > 0:
                i -= 1
        else:
            i += 1

    def predict(x_new):
        # For each x_new, find the block index by searching sorted x_s.
        # Blocks are represented by their upper x boundary (x_s[i]).
        # y = values[i] where x_s[i-1] < x_new <= x_s[i], with clamps at edges.
        idx = np.searchsorted(x_s, x_new, side="right") - 1
        idx = np.clip(idx, 0, len(values) - 1)
        return values[idx]

    return predict


def calibration_curve(y_true, y_pred, bins=10):
    """Bucket predictions into `bins` deciles, report mean predicted vs mean actual per bucket."""
    order = np.argsort(y_pred)
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    n = len(order)
    rows = []
    for b in range(bins):
        lo = (b * n) // bins
        hi = ((b + 1) * n) // bins
        if hi <= lo:
            continue
        rows.append({
            "bin": b + 1,
            "n": hi - lo,
            "pred_mean": float(np.mean(y_pred_sorted[lo:hi])),
            "actual_mean": float(np.mean(y_true_sorted[lo:hi])),
            "pred_range": (float(y_pred_sorted[lo]), float(y_pred_sorted[hi - 1])),
        })
    return rows


def evaluate_split(name, X_train, y_train, X_test, y_test):
    print(f"\n### {name}")
    print(f"  train n={len(y_train)}  test n={len(y_test)}")

    # Linear model on full feature bag
    _, predict_lin = linear_fit(X_train, y_train)
    y_pred_lin = np.clip(predict_lin(X_test), 0.0, 1.0)
    r2_lin = 1.0 - np.sum((y_test - y_pred_lin) ** 2) / max(
        np.sum((y_test - np.mean(y_test)) ** 2), 1e-12
    )
    print(
        f"  linear (6 features): pearson={pearson(y_pred_lin, y_test):+.3f}  "
        f"spearman={spearman(y_pred_lin, y_test):+.3f}  R^2={r2_lin:+.3f}"
    )

    # Isotonic on top-vs-2nd margin alone (strongest single signal on scifact)
    predict_iso_margin = isotonic_fit(X_train[:, 1], y_train)
    y_pred_iso_margin = predict_iso_margin(X_test[:, 1])
    r2_im = 1.0 - np.sum((y_test - y_pred_iso_margin) ** 2) / max(
        np.sum((y_test - np.mean(y_test)) ** 2), 1e-12
    )
    print(
        f"  isotonic (margin only): pearson={pearson(y_pred_iso_margin, y_test):+.3f}  "
        f"spearman={spearman(y_pred_iso_margin, y_test):+.3f}  R^2={r2_im:+.3f}"
    )

    # Isotonic on top_score alone (strongest single signal on nfcorpus)
    predict_iso_top = isotonic_fit(X_train[:, 0], y_train)
    y_pred_iso_top = predict_iso_top(X_test[:, 0])
    r2_it = 1.0 - np.sum((y_test - y_pred_iso_top) ** 2) / max(
        np.sum((y_test - np.mean(y_test)) ** 2), 1e-12
    )
    print(
        f"  isotonic (top_score):   pearson={pearson(y_pred_iso_top, y_test):+.3f}  "
        f"spearman={spearman(y_pred_iso_top, y_test):+.3f}  R^2={r2_it:+.3f}"
    )

    # Show calibration curve from the linear model — 10 buckets of prediction
    print(f"  calibration (linear pred → actual nDCG, 10 buckets):")
    print(f"  {'bin':>4} {'n':>5} {'pred_mean':>10} {'actual_mean':>12} {'pred_range':>20}")
    for row in calibration_curve(y_test, y_pred_lin):
        lo, hi = row["pred_range"]
        print(
            f"  {row['bin']:>4} {row['n']:>5} {row['pred_mean']:>10.4f} "
            f"{row['actual_mean']:>12.4f}  [{lo:>6.3f},{hi:>6.3f}]"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("matrix_paths", nargs="+", type=Path)
    p.add_argument("--candidates", type=int, default=250)
    args = p.parse_args()

    modes = [
        ("RAW FEATURES + ABSOLUTE nDCG LABEL", False, False),
        ("QUANTILE FEATURES + ABSOLUTE nDCG LABEL", True, False),
        ("QUANTILE FEATURES + QUANTILE nDCG LABEL (relative rank)", True, True),
    ]
    for header, quantile_mode, quantile_label in modes:
        print(f"\n{'=' * 78}\n{header}\n{'=' * 78}")

        corpora = []
        for path in args.matrix_paths:
            dataset, key, rows = load_rows(path, args.candidates)
            X, y = build_features(rows, quantile=quantile_mode, quantile_label=quantile_label)
            corpora.append({"path": path, "dataset": dataset, "key": key, "X": X, "y": y})
            if not quantile_mode and not quantile_label:
                print(f"loaded {dataset}: n={len(y)}, key={key}")

        if len(corpora) < 2:
            print("need at least 2 corpora for leave-one-out")
            return

        # Baseline: within-corpus fit + predict (upper bound of what's possible)
        print("\n## Within-corpus (upper bound)")
        for c in corpora:
            evaluate_split(f"{c['dataset']} → {c['dataset']}", c["X"], c["y"], c["X"], c["y"])

        # Cross-corpus leave-one-out: this is the real generalization test.
        print("\n## Leave-one-corpus-out (generalization)")
        for i, held_out in enumerate(corpora):
            train_corpora = [c for j, c in enumerate(corpora) if j != i]
            X_train = np.vstack([c["X"] for c in train_corpora])
            y_train = np.concatenate([c["y"] for c in train_corpora])
            train_names = "+".join(c["dataset"].split("/")[-2] for c in train_corpora)
            held_name = held_out["dataset"].split("/")[-2]
            evaluate_split(
                f"train={train_names}  test={held_name}",
                X_train, y_train, held_out["X"], held_out["y"],
            )


if __name__ == "__main__":
    main()
