"""Train, compare and save the model.   Run:  python train.py

Flow
 1. Split the data (stratified). The TEST set is locked away until the very end.
 2. Compare 3 models with 5-fold cross-validation on the TRAIN data only.
 3. Pick the best model (if scores are practically tied, pick the simpler one).
 4. Choose a decision threshold so that dangerous (malignant) cases are rarely missed.
 5. Evaluate ONCE on the test set, with bootstrap confidence intervals.
 6. Save model, metadata, metrics and plots.
"""
import argparse, json, os
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MALIGNANT, BENIGN = 0, 1          # label meaning in this dataset
TIE_TOLERANCE = 0.005             # CV scores closer than this count as a tie
SIMPLICITY_ORDER = ["logistic_regression", "random_forest", "neural_network"]  # simplest first


def build_candidates(seed):
    """3 models. The scaler sits INSIDE each pipeline, so cross-validation fits it on the
    training folds only. That prevents data leakage."""
    def pipe(estimator):
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])
    return {
        "logistic_regression": pipe(LogisticRegression(max_iter=2000)),
        "random_forest": pipe(RandomForestClassifier(n_estimators=300, random_state=seed)),
        "neural_network": pipe(MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=2000, random_state=seed)),
    }


def select_model(cv_scores, tol=TIE_TOLERANCE):
    """Best CV score wins. If several models are within `tol` of the best, take the simplest."""
    best = max(cv_scores.values())
    return next(n for n in SIMPLICITY_ORDER if n in cv_scores and cv_scores[n] >= best - tol)


def choose_threshold(y, p_mal, target_recall=0.98):
    """Largest threshold t (flag 'malignant' when P(malignant) >= t) that still catches at
    least `target_recall` of the malignant cases. Never stricter than the default 0.5."""
    y, p_mal = np.asarray(y), np.asarray(p_mal)
    n_mal = (y == MALIGNANT).sum()
    for t in np.unique(p_mal)[::-1]:                      # try high thresholds first
        if ((p_mal >= t) & (y == MALIGNANT)).sum() / n_mal >= target_recall:
            return float(np.floor(min(t, 0.5) * 1e4) / 1e4)  # round DOWN so recall never drops below target
    return 0.0


def evaluate(y, p_mal, t):
    """Metrics when we flag 'malignant' for P(malignant) >= t."""
    y, p_mal = np.asarray(y), np.asarray(p_mal)
    pred_mal, is_mal = p_mal >= t, y == MALIGNANT
    tp, fn = int((pred_mal & is_mal).sum()), int((~pred_mal & is_mal).sum())
    fp, tn = int((pred_mal & ~is_mal).sum()), int((~pred_mal & ~is_mal).sum())
    return {"accuracy": (tp + tn) / len(y),
            "malignant_recall": tp / (tp + fn) if tp + fn else float("nan"),
            "malignant_precision": tp / (tp + fp) if tp + fp else float("nan"),
            "confusion": {"tp_malignant_caught": tp, "fn_malignant_missed": fn,
                          "fp_false_alarm": fp, "tn_benign_ok": tn}}


def bootstrap_ci(y, p_mal, t, n=2000, seed=0):
    """95% interval: re-draw the test set with replacement many times. Shows how shaky a small test set is."""
    rng, y, p_mal = np.random.default_rng(seed), np.asarray(y), np.asarray(p_mal)
    acc, rec = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        m = evaluate(y[i], p_mal[i], t)
        acc.append(m["accuracy"]); rec.append(m["malignant_recall"])
    lo_hi = lambda v: [round(float(x), 4) for x in np.nanpercentile(v, [2.5, 97.5])]
    return {"accuracy": lo_hi(acc), "malignant_recall": lo_hi(rec)}


def _round(d):
    def r(v):
        if isinstance(v, float): return round(v, 4)
        if isinstance(v, dict): return _round(v)
        if isinstance(v, list): return [r(x) for x in v]
        return v
    return {k: r(v) for k, v in d.items()}


def main(seed=0, target_recall=0.98, out="results", model_dir="models"):
    os.makedirs(out, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    data = load_breast_cancer()
    X, y = data.data, data.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 2-3) compare models with cross-validation on TRAIN data only (ROC-AUC does not depend on a threshold)
    candidates = build_candidates(seed)
    cv_auc = {n: float(cross_val_score(p, Xtr, ytr, cv=cv, scoring="roc_auc").mean()) for n, p in candidates.items()}
    best_name = select_model(cv_auc)

    # 4) pick the threshold from out-of-fold predictions (each train sample is predicted by a model that never saw it)
    oof_p_mal = cross_val_predict(clone(candidates[best_name]), Xtr, ytr, cv=cv, method="predict_proba")[:, MALIGNANT]
    threshold = choose_threshold(ytr, oof_p_mal, target_recall)

    # 5) final fit, then the test set is used exactly once
    model = clone(candidates[best_name]).fit(Xtr, ytr)
    assert list(model.classes_) == [MALIGNANT, BENIGN]
    p_mal = model.predict_proba(Xte)[:, MALIGNANT]
    metrics = _round({
        "selected_model": best_name,
        "selection_rule": f"highest 5-fold CV ROC-AUC; scores within {TIE_TOLERANCE} count as a tie -> simpler model",
        "cv_roc_auc": cv_auc,
        "target_malignant_recall": target_recall,
        "threshold": threshold,
        "test_roc_auc": float(roc_auc_score(yte == MALIGNANT, p_mal)),
        "test_at_default_threshold_0.5": evaluate(yte, p_mal, 0.5),
        "test_at_tuned_threshold": evaluate(yte, p_mal, threshold),
        "test_bootstrap_95ci_at_tuned_threshold": bootstrap_ci(yte, p_mal, threshold, seed=seed),
        "p_malignant_of_cases_missed_at_0.5": sorted(float(x) for x in p_mal[(yte == MALIGNANT) & (p_mal < 0.5)]),
        "train_test_sizes": [len(ytr), len(yte)],
        "seed": seed,
        "sklearn_version": sklearn.__version__,
    })
    json.dump(metrics, open(f"{out}/metrics.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))

    # 6) save what the API needs
    joblib.dump(model, f"{model_dir}/model.joblib")
    meta = {"model_name": best_name, "threshold": threshold, "feature_names": list(data.feature_names),
            "classes": {"0": "malignant", "1": "benign"}, "sklearn_version": sklearn.__version__, "metrics": metrics}
    json.dump(meta, open(f"{model_dir}/meta.json", "w"), indent=2)
    make_plots(out, yte, p_mal, threshold, ytr, oof_p_mal)
    return metrics


def make_plots(out, yte, p_mal, threshold, ytr, oof_p_mal):
    m = evaluate(yte, p_mal, threshold)["confusion"]
    cm = np.array([[m["tp_malignant_caught"], m["fn_malignant_missed"]], [m["fp_false_alarm"], m["tn_benign_ok"]]])
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set(title=f"Test set, threshold {threshold}", xlabel="model says", ylabel="truth",
           xticks=[0, 1], yticks=[0, 1], xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(f"{out}/confusion_matrix.png", dpi=120); plt.close(fig)

    ts = np.linspace(0.01, 0.99, 99)
    ev = [evaluate(ytr, oof_p_mal, t) for t in ts]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.plot(ts, [e["malignant_recall"] for e in ev], label="malignant recall (cases caught)")
    ax.plot(ts, [e["malignant_precision"] for e in ev], label="malignant precision (alarms that are real)")
    ax.axvline(0.5, color="grey", ls=":", label="default 0.5")
    ax.axvline(threshold, color="red", ls="--", label=f"chosen {threshold}")
    ax.set(xlabel="threshold on P(malignant)", ylim=(0.8, 1.01), title="Threshold trade-off (train, out-of-fold)")
    ax.legend(fontsize=8, loc="lower center"); fig.tight_layout()
    fig.savefig(f"{out}/threshold_tradeoff.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-recall", type=float, default=0.98, help="share of malignant cases we insist on catching")
    a = ap.parse_args()
    main(a.seed, a.target_recall)
