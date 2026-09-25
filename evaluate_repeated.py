"""One test split can be lucky or unlucky. This repeats the whole procedure on many random
splits and averages, so we can judge the tuned threshold fairly.   Run:  python evaluate_repeated.py"""
import argparse, json, os, warnings
import numpy as np
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from train import MALIGNANT, build_candidates, choose_threshold, evaluate


def run(model_name="logistic_regression", n_splits=10, target_recall=0.98, out="results"):
    warnings.filterwarnings("ignore")
    X, y = load_breast_cancer(return_X_y=True)
    rows = []
    for seed in range(n_splits):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        pipe = build_candidates(seed)[model_name]
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = cross_val_predict(clone(pipe), Xtr, ytr, cv=cv, method="predict_proba")[:, MALIGNANT]
        t = choose_threshold(ytr, oof, target_recall)
        p = clone(pipe).fit(Xtr, ytr).predict_proba(Xte)[:, MALIGNANT]
        d, u = evaluate(yte, p, 0.5)["confusion"], evaluate(yte, p, t)["confusion"]
        rows.append({"seed": seed, "threshold": t,
                     "default_missed": d["fn_malignant_missed"], "default_false_alarms": d["fp_false_alarm"],
                     "tuned_missed": u["fn_malignant_missed"], "tuned_false_alarms": u["fp_false_alarm"]})
    avg = {k: round(float(np.mean([r[k] for r in rows])), 2) for k in rows[0] if k != "seed"}
    result = {"model": model_name, "n_splits": n_splits, "test_size_per_split": 114,
              "average_per_split": avg, "per_split": rows}
    os.makedirs(out, exist_ok=True)
    json.dump(result, open(f"{out}/repeated_splits.json", "w"), indent=2)
    print(json.dumps(result["average_per_split"], indent=2))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="logistic_regression")
    ap.add_argument("--splits", type=int, default=10)
    a = ap.parse_args()
    run(a.model, a.splits)
