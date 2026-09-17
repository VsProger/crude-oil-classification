"""
Trains and persists the model served by app.py.

Uses the honest feature set only -- density and API are excluded, because the
label is derived from API and API is a closed-form function of density (see
experiments.py, study A). Everything the model needs at inference time lives
inside the pipeline, so app.py does no preprocessing of its own.

Outputs: model.pkl, results/fig_confusion_matrix.png, results/fig_ablation.png
"""

import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from features import CLASSES, FEATURES, log_viscosity

RANDOM_STATE = 42


def build_pipeline():
    return Pipeline([
        ("log", FunctionTransformer(log_viscosity, feature_names_out="one-to-one")),
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ])


def main():
    df = pd.read_csv("dataset.csv")
    X, y = df[FEATURES], df["class"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipe = build_pipeline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"5-fold CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    print("\nHeld-out test set:")
    print(classification_report(y_te, pred))

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay.from_predictions(
        y_te, pred, labels=CLASSES, display_labels=CLASSES, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Random Forest, density excluded")
    fig.tight_layout()
    fig.savefig("results/fig_confusion_matrix.png", dpi=150)

    ablation = pd.read_csv("results/feature_ablation.csv").sort_values("Alone_CV_accuracy")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(ablation["Feature"], ablation["Alone_CV_accuracy"], color="#4C78A8")
    ax.axvline(0.5372, color="#E45756", linestyle="--", label="majority-class baseline")
    ax.set_xlabel("5-fold CV accuracy using that feature alone")
    ax.set_xlim(0.4, 0.8)
    ax.legend()
    ax.set_title("Which measurements carry signal on their own")
    fig.tight_layout()
    fig.savefig("results/fig_ablation.png", dpi=150)

    # Refit on all data for the served artefact
    final = build_pipeline().fit(X, y)
    with open("model.pkl", "wb") as f:
        pickle.dump({"pipeline": final, "features": FEATURES, "classes": list(final.classes_)}, f)
    print(f"\nSaved model.pkl (features: {FEATURES})")


if __name__ == "__main__":
    main()
