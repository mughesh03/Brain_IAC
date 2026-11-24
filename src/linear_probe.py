"""Train a linear three-class probe on frozen features and score val and test."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


def assert_feature_splits_disjoint(data):
    """Refuse a feature file whose splits still share identical vectors."""
    groups = {}
    for name in ("train", "val", "test"):
        rows = np.ascontiguousarray(data[f"{name}_features"])
        keys = [row.tobytes() for row in rows]
        if len(keys) != len(set(keys)):
            raise SystemExit(f"duplicate feature rows inside the {name} split")
        groups[name] = set(keys)
    shared = (groups["train"] & groups["val"]) | (groups["train"] & groups["test"]) | (groups["val"] & groups["test"])
    if shared:
        raise SystemExit(
            f"{len(shared)} identical feature rows appear in more than one split. "
            "Re-extract features after the channel-0 fix."
        )


def score_split(name, classifier, scaler, features, labels):
    x = scaler.transform(features)
    pred = classifier.predict(x)
    proba = classifier.predict_proba(x)
    metrics = {
        "split": name,
        "accuracy": accuracy_score(labels, pred),
        "precision": precision_score(labels, pred, average="macro", zero_division=0),
        "recall": recall_score(labels, pred, average="macro", zero_division=0),
        "f1": f1_score(labels, pred, average="macro", zero_division=0),
        "auroc": roc_auc_score(labels, proba, multi_class="ovr", average="macro"),
        "confusion_matrix": confusion_matrix(labels, pred, labels=[0, 1, 2]),
    }
    return metrics


def format_report(metrics):
    matrix = np.array2string(metrics["confusion_matrix"], separator=", ")
    return (
        f"{metrics['split']}\n"
        f"accuracy: {metrics['accuracy']:.4f}\n"
        f"precision: {metrics['precision']:.4f}\n"
        f"recall: {metrics['recall']:.4f}\n"
        f"f1: {metrics['f1']:.4f}\n"
        f"auroc: {metrics['auroc']:.4f}\n"
        f"confusion_matrix:\n{matrix}\n"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Linear probe on frozen BrainIAC features")
    parser.add_argument("--features", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    features_path = Path(args.features)
    if not features_path.is_file():
        raise SystemExit(f"Missing features: {features_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(features_path)
    assert_feature_splits_disjoint(data)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(data["train_features"])
    classifier = LogisticRegression(max_iter=2000, random_state=args.seed)
    classifier.fit(x_train, data["train_labels"])

    reports = []
    for split_name in ("val", "test"):
        metrics = score_split(
            split_name,
            classifier,
            scaler,
            data[f"{split_name}_features"],
            data[f"{split_name}_labels"],
        )
        text = format_report(metrics)
        print(text)
        reports.append(text)
        np.savetxt(
            output_dir / f"confusion_matrix_{split_name}.csv",
            metrics["confusion_matrix"],
            delimiter=",",
            fmt="%d",
        )
    (output_dir / "metrics.txt").write_text("".join(reports), encoding="utf-8")
    print(f"saved {output_dir / 'metrics.txt'}")


if __name__ == "__main__":
    main()
