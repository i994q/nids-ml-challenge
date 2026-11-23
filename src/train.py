"""Train and compare the baseline and advanced intrusion detection models.

Optimisation metric is macro F1: attacks are a small share of traffic and the rare
classes (SQL Injection, Brute Force -Web/-XSS) have dozens of rows, so accuracy and
even weighted F1 hide exactly the failures that matter.

    python src/train.py                      # both models, default settings
    python src/train.py --leaf 20            # forest leaf size (see README on size)
    python src/train.py --skip-mlp           # baseline only, for a quick loop

Measured on this laptop (15 cores): the forest costs about 23 s per million rows, and
min_samples_leaf=20 shrinks it ~11x with no time penalty. The MLP costs about 8 s per
iteration per 4.7M rows.
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models")
SEED = 42
BENIGN = "Benign"


class LabelDecodingClassifier:
    """Wraps the MLP so .predict() returns the original label strings.

    The dashboard only ever calls .predict(), and it must get 'Benign' /
    'DDOS attack-HOIC' back, not the integers the MLP was trained on.
    """

    def __init__(self, pipeline, encoder):
        self.pipeline = pipeline
        self.encoder = encoder
        self.classes_ = encoder.classes_

    def predict(self, X):
        return self.encoder.inverse_transform(self.pipeline.predict(X))

    def predict_proba(self, X):
        return self.pipeline.predict_proba(X)


def scores(name, y_true, pred, classes, seconds):
    macro = f1_score(y_true, pred, average="macro")
    per_class = f1_score(y_true, pred, average=None, labels=classes)
    # binary view: anything not Benign is an attack -- this is what a SOC cares about
    attack_recall = recall_score(y_true != BENIGN, pred != BENIGN, pos_label=True)
    tn = int(((y_true == BENIGN) & (pred == BENIGN)).sum())
    fp = int(((y_true == BENIGN) & (pred != BENIGN)).sum())
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    print(f"\n=== {name} ===  (fit {seconds:.1f}s)")
    print(f"macro F1            : {macro:.4f}")
    print(f"attack recall       : {attack_recall:.4f}   (any non-Benign class)")
    print(f"false-positive rate : {fpr:.4f}   ({fp:,} of {fp + tn:,} benign flows)")
    worst = sorted(zip(classes, per_class), key=lambda kv: kv[1])[:3]
    print("weakest three classes by F1:")
    for cls, f1 in worst:
        n = int((y_true == cls).sum())
        rec = recall_score(y_true == cls, pred == cls, pos_label=True) if n else float("nan")
        print(f"  {cls:<28} F1 {f1:.3f}   recall {rec:.3f}   test rows {n:,}")
    return {"macro_f1": float(macro), "attack_recall": float(attack_recall),
            "fpr": float(fpr), "fit_seconds": round(seconds, 1),
            "per_class_f1": {c: float(v) for c, v in zip(classes, per_class)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaf", type=int, default=20,
                    help="min_samples_leaf for the forest; 20 keeps the model ~11x smaller")
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--mlp-iter", type=int, default=60)
    ap.add_argument("--skip-mlp", action="store_true")
    args = ap.parse_args()

    bundle = joblib.load(os.path.join(MODELS, "clean.joblib"))
    table = bundle["table"]
    label_col = bundle["label_col"]
    print(f"table: {len(table):,} rows   features {table.shape[1] - 1}   "
          f"classes {table[label_col].nunique()}")
    print(table[label_col].value_counts().to_string())

    X = table.drop(columns=[label_col]).to_numpy(dtype="float32")
    y = table[label_col].to_numpy()
    features = [c for c in table.columns if c != label_col]
    del table

    # a class with a single row cannot be stratified or learned; report and drop it
    counts = pd.Series(y).value_counts()
    too_rare = counts[counts < 10].index.tolist()
    if too_rare:
        print(f"\nclasses with under 10 rows, dropped: {too_rare}")
        keep = ~np.isin(y, too_rare)
        X, y = X[keep], y[keep]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y,
                                              random_state=SEED)
    classes = sorted(set(y))
    print(f"\ntrain {len(X_tr):,}   test {len(X_te):,}")

    results, models = {}, {}

    # ---- baseline: random forest on the raw features -------------------------
    rf = RandomForestClassifier(n_estimators=args.trees, class_weight="balanced",
                                n_jobs=-1, random_state=SEED,
                                min_samples_leaf=args.leaf)
    t0 = time.time()
    rf.fit(X_tr, y_tr)
    results["random_forest"] = scores(
        f"random forest (trees={args.trees}, min_samples_leaf={args.leaf})",
        y_te, rf.predict(X_te), classes, time.time() - t0)
    nodes = sum(e.tree_.node_count for e in rf.estimators_)
    print(f"forest size: {nodes:,} nodes (~{nodes * 80 / 1e6:.0f} MB in memory)")
    results["random_forest"]["forest_nodes"] = int(nodes)
    models["random_forest"] = rf

    # ---- advanced: MLP, which needs scaled inputs ----------------------------
    if not args.skip_mlp:
        # sklearn 1.6's MLPClassifier crashes with early_stopping on string labels
        # (it calls np.isnan on the predicted strings), so the labels are encoded to
        # integers here and the wrapper maps them back.
        encoder = LabelEncoder().fit(y_tr)
        mlp = make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=args.mlp_iter,
                          early_stopping=True, n_iter_no_change=5, random_state=SEED),
        )
        t0 = time.time()
        mlp.fit(X_tr, encoder.transform(y_tr))
        pred = encoder.inverse_transform(mlp.predict(X_te))
        results["mlp"] = scores(f"MLP 128-64 (max_iter={args.mlp_iter}, early stopping)",
                                y_te, pred, classes, time.time() - t0)
        results["mlp"]["iterations_run"] = int(mlp[-1].n_iter_)
        models["mlp"] = LabelDecodingClassifier(mlp, encoder)

    # ---- keep the better macro F1 -------------------------------------------
    best_name = max(results, key=lambda k: results[k]["macro_f1"])
    best = models[best_name]
    print(f"\n=== best by macro F1: {best_name} ({results[best_name]['macro_f1']:.4f}) ===")

    joblib.dump({"model": best, "features": features, "classes": classes,
                 "model_name": best_name}, os.path.join(MODELS, "model.joblib"), compress=3)
    size_mb = os.path.getsize(os.path.join(MODELS, "model.joblib")) / 1e6
    print(f"wrote models/model.joblib ({size_mb:.1f} MB)")

    print("\nfull classification report for the saved model:")
    print(classification_report(y_te, best.predict(X_te), zero_division=0))

    cm = confusion_matrix(y_te, best.predict(X_te), labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        os.path.join(MODELS, "confusion_matrix.csv"))

    with open(os.path.join(MODELS, "metrics.json"), "w") as fh:
        json.dump({"best": best_name, "model_file_mb": round(size_mb, 1),
                   "n_train": len(X_tr), "n_test": len(X_te),
                   "benign_cap": bundle.get("benign_cap"), "results": results},
                  fh, indent=1)
    print("wrote models/metrics.json and models/confusion_matrix.csv")


if __name__ == "__main__":
    main()
