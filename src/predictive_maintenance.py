"""Train and evaluate predictive-maintenance models on synthetic sensor data.

Two complementary models, both standard scikit-learn practice:
  1. RandomForest classifier  -> predicts whether a failure occurs within
     a short horizon, from rolling sensor features.
  2. IsolationForest          -> flags anomalous operating points without
     needing labels (unsupervised).

Run `generate_synthetic_data.py` first to create synthetic_sensor_data.csv.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

FEATURES = ["vibration", "temperature", "pressure", "current"]


def add_rolling_features(df: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """Rolling mean/std per machine capture drift better than raw samples."""
    df = df.sort_values(["machine_id", "hour"]).copy()
    for col in FEATURES:
        rolled = df.groupby("machine_id")[col].rolling(window, min_periods=1)
        df[f"{col}_mean"] = rolled.mean().reset_index(level=0, drop=True)
        df[f"{col}_std"] = rolled.std().reset_index(level=0, drop=True).fillna(0)
    # Machines have different baselines, so normalise each channel per
    # machine: the model learns "unusual for THIS machine", which is how
    # plant monitoring is actually done.
    for col in list(FEATURES) + [f"{c}_{stat}" for c in FEATURES for stat in ("mean", "std")]:
        g = df.groupby("machine_id")[col]
        df[col] = (df[col] - g.transform("mean")) / g.transform("std").replace(0, 1)
    # Failure within the next 48 hours = the prediction target.
    df["failure_soon"] = (
        df.groupby("machine_id")["failure"]
        .transform(lambda s: s[::-1].rolling(48, min_periods=1).max()[::-1])
    )
    return df


def main() -> None:
    df = add_rolling_features(pd.read_csv("synthetic_sensor_data.csv"))
    feature_cols = [f"{c}_{stat}" for c in FEATURES for stat in ("mean", "std")]
    X, y = df[feature_cols], df["failure_soon"]

    # Split by machine, not by row: rolling features of neighbouring hours
    # overlap, so a random row split would leak and inflate the scores.
    # Testing on machines the model never saw is the honest benchmark.
    tr_idx, te_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                            random_state=42).split(X, y, df["machine_id"]))
    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    print("RandomForest (failure within 48h)")
    print(classification_report(y_test, clf.predict(X_test)))
    print(f"ROC AUC: {roc_auc_score(y_test, proba):.3f}")

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values()
    importances.plot.barh(figsize=(8, 5), title="Feature importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=120)

    iso = IsolationForest(contamination=0.02, random_state=42)
    df["anomaly"] = iso.fit_predict(X) == -1
    print(f"IsolationForest flagged {df['anomaly'].mean():.1%} of hours as anomalous.")

    sample = df[df["machine_id"] == 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sample["hour"], sample["vibration"], label="vibration (synthetic)", lw=0.8)
    ax.scatter(sample.loc[sample["anomaly"], "hour"],
               sample.loc[sample["anomaly"], "vibration"],
               color="red", s=8, label="anomaly")
    ax.legend()
    ax.set_xlabel("hour")
    fig.tight_layout()
    fig.savefig("anomaly_example.png", dpi=120)
    print("Saved feature_importance.png and anomaly_example.png")


if __name__ == "__main__":
    main()
