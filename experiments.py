"""
Oil classification experiments.

Runs three studies and writes their metrics to results/:

  A. Leakage demonstration — how much of the reported accuracy comes from the
     label being a deterministic function of a feature.
  B. Honest baseline      — the same models with the leaking columns removed.
  C. Feature ablation     — which of the remaining measurements carry signal.

All preprocessing happens inside a Pipeline so that imputation and scaling are
fitted on training folds only.
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUT = "results"

LEAKY = ["API", "density"]
HONEST_FEATURES = [
    "viscosity", "pour_point", "flash_point",
    "saturates", "aromatics", "resins", "asphaltenes", "sulfur",
]


def api_from_density(density):
    """Standard API-gravity conversion. Specific gravity ~ density in g/mL."""
    return 141.5 / density - 131.5


def log_viscosity(X):
    """Viscosity spans seven orders of magnitude; model it on a log scale."""
    X = X.copy()
    if "viscosity" in X:
        X["viscosity"] = np.log10(X["viscosity"].clip(lower=1e-3))
    return X


def make_pipeline(model, scale=True):
    steps = [
        ("log", FunctionTransformer(log_viscosity, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def models():
    return {
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(random_state=RANDOM_STATE), False),
        "KNN": (KNeighborsClassifier(n_neighbors=5), True),
    }


def evaluate(X, y, label, report):
    """Stratified 5-fold CV plus a held-out test split, for each model."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    rows = []
    baseline = make_pipeline(DummyClassifier(strategy="most_frequent"), scale=False)
    baseline.fit(X_tr, y_tr)
    rows.append({
        "Model": "Majority-class baseline",
        "CV_accuracy_mean": cross_val_score(baseline, X, y, cv=cv, scoring="accuracy").mean(),
        "CV_accuracy_std": cross_val_score(baseline, X, y, cv=cv, scoring="accuracy").std(),
        "Test_accuracy": accuracy_score(y_te, baseline.predict(X_te)),
        "Test_macro_F1": f1_score(y_te, baseline.predict(X_te), average="macro"),
    })

    for name, (model, scale) in models().items():
        pipe = make_pipeline(model, scale=scale)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        rows.append({
            "Model": name,
            "CV_accuracy_mean": scores.mean(),
            "CV_accuracy_std": scores.std(),
            "Test_accuracy": accuracy_score(y_te, pred),
            "Test_macro_F1": f1_score(y_te, pred, average="macro"),
        })
        report[f"{label} :: {name}"] = classification_report(y_te, pred, output_dict=True)

    df = pd.DataFrame(rows).sort_values("Test_accuracy", ascending=False)
    return df, (X_tr, X_te, y_tr, y_te)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv("dataset.csv")
    y = df["class"]
    report = {}

    print("=" * 72)
    print("A. LEAKAGE DEMONSTRATION")
    print("=" * 72)

    # The label is classify_oil(API). API is a closed-form function of density,
    # so a two-line rule over density alone should reproduce the label exactly.
    api_hat = api_from_density(df["density"])
    rule = pd.cut(api_hat, [-np.inf, 22, 31, np.inf], labels=["heavy", "medium", "light"])
    rule_acc = accuracy_score(y, rule.astype(str))
    print(f"Rule 'API = 141.5/density - 131.5' + thresholds, no learning at all:")
    print(f"  accuracy against the label = {rule_acc:.4f}")
    print(f"  correlation(density, API)  = {df['density'].corr(df['API']):.4f}")

    leaky_df, _ = evaluate(df[["density"] + HONEST_FEATURES], y, "leaky", report)
    print("\nModels trained WITH density (the original setup):")
    print(leaky_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 72)
    print("B. HONEST BASELINE — density and API removed")
    print("=" * 72)
    honest_df, split = evaluate(df[HONEST_FEATURES], y, "honest", report)
    print(honest_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print()
    print("=" * 72)
    print("C. FEATURE ABLATION (Random Forest, CV accuracy)")
    print("=" * 72)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ablation = []
    for feat in HONEST_FEATURES:
        pipe = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE), scale=False)
        score = cross_val_score(pipe, df[[feat]], y, cv=cv, scoring="accuracy").mean()
        ablation.append({
            "Feature": feat,
            "Alone_CV_accuracy": score,
            "Coverage": 1 - df[feat].isna().mean(),
        })
    ablation_df = pd.DataFrame(ablation).sort_values("Alone_CV_accuracy", ascending=False)
    print(ablation_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Confusion matrix for the best honest model
    X_tr, X_te, y_tr, y_te = split
    best = make_pipeline(RandomForestClassifier(random_state=RANDOM_STATE), scale=False)
    best.fit(X_tr, y_tr)
    cm = confusion_matrix(y_te, best.predict(X_te), labels=["heavy", "medium", "light"])
    print("\nHonest Random Forest — confusion matrix (rows = true, cols = pred)")
    print(pd.DataFrame(cm, index=["heavy", "medium", "light"],
                       columns=["heavy", "medium", "light"]).to_string())
    print("\nPer-class report:")
    print(classification_report(y_te, best.predict(X_te)))

    leaky_df.to_csv(f"{OUT}/benchmark_with_density_leak.csv", index=False)
    honest_df.to_csv(f"{OUT}/benchmark_honest.csv", index=False)
    ablation_df.to_csv(f"{OUT}/feature_ablation.csv", index=False)
    pd.DataFrame(cm, index=["heavy", "medium", "light"],
                 columns=["heavy", "medium", "light"]).to_csv(f"{OUT}/confusion_matrix_honest.csv")
    with open(f"{OUT}/summary.json", "w") as f:
        json.dump({
            "n_assays": len(df),
            "class_balance": y.value_counts().to_dict(),
            "coverage": (1 - df.isna().mean()).round(4).to_dict(),
            "leakage_rule_accuracy": round(rule_acc, 4),
            "density_api_correlation": round(float(df["density"].corr(df["API"])), 4),
            "random_state": RANDOM_STATE,
        }, f, indent=2)
    print(f"\nWritten to {OUT}/")


if __name__ == "__main__":
    main()
