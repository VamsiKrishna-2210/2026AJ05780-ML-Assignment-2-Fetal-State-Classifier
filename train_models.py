"""Fit the five classifiers, save them, and print the test-set metrics.

Run this on the BITS virtual lab so the terminal output can be screenshotted:

    python train_models.py
"""

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd

from common import (
    CLASS_ORDER,
    DATA_PATH,
    FEATURES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILENAMES,
    TARGET,
    TEST_PATH,
    aligned_probabilities,
    compute_metrics,
    make_models,
)


def main():
    frame = pd.read_csv(DATA_PATH)
    train_rows, test_rows = train_test_split(
        frame,
        test_size=0.2,
        random_state=42,
        stratify=frame[TARGET],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_rows.to_csv(TEST_PATH, index=False)

    records = []
    report_blocks = []
    for name, estimator in make_models().items():
        estimator.fit(train_rows[FEATURES], train_rows[TARGET])
        joblib.dump(estimator, MODEL_DIR / MODEL_FILENAMES[name])

        predicted = estimator.predict(test_rows[FEATURES])
        proba = aligned_probabilities(estimator, test_rows[FEATURES], CLASS_ORDER)
        scores = compute_metrics(test_rows[TARGET], predicted, proba, CLASS_ORDER)
        scores["ML Model Name"] = name
        records.append(scores)

        report_blocks.append(
            name
            + "\n"
            + classification_report(
                test_rows[TARGET],
                predicted,
                labels=CLASS_ORDER,
                digits=4,
                zero_division=0,
            )
        )

    metrics = pd.DataFrame(records)[
        ["ML Model Name", "Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    ]
    metrics.to_csv(METRICS_PATH, index=False)
    (METRICS_PATH.parent / "classification_reports.txt").write_text(
        "\n".join(report_blocks),
        encoding="utf-8",
    )

    print("Cardiotocography fetal-state classification")
    print(f"Source file : {DATA_PATH.name}")
    print(f"Train rows  : {len(train_rows)}")
    print(f"Test rows   : {len(test_rows)}  (written to {TEST_PATH.name})")
    print(f"Features    : {len(FEATURES)}")
    print()
    print("Metrics on the held-out test rows. Models were not refit on this split.")
    print("Precision, Recall and F1 are macro averages. AUC is one-vs-rest macro.")
    print()
    header = (
        f"{'ML Model Name':<24} {'Accuracy':>9} {'AUC':>9} "
        f"{'Precision':>9} {'Recall':>9} {'F1':>9} {'MCC':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in metrics.itertuples(index=False):
        print(
            f"{row[0]:<24} {row.Accuracy:9.4f} {row.AUC:9.4f} "
            f"{row.Precision:9.4f} {row.Recall:9.4f} {row.F1:9.4f} {row.MCC:9.4f}"
        )
    print()
    print(f"Saved models in {MODEL_DIR}")
    print(f"Saved metric table to {METRICS_PATH}")


if __name__ == "__main__":
    main()
