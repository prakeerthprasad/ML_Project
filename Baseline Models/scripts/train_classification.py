import argparse
import os
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)

import matplotlib.pyplot as plt


# -----------------------------
# Helpers for loading & labels
# -----------------------------

VALID_PITCH_TYPES = [
    "fastball",
    "slider",
    "curveball",
    "changeup",
    "sinker",
    "knucklecurve",
]


def normalize_pitch_type(raw):
    """
    Normalize raw pitch type strings into one of the 6 canonical types.
    Returns None for anything we want to drop (unknown/other).
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()

    # basic mappings / synonyms
    if s in ["fastball", "four-seam fastball", "fourseam", "four-seam"]:
        return "fastball"
    if s in ["slider"]:
        return "slider"
    if s in ["curveball", "curve"]:
        return "curveball"
    if s in ["changeup", "change-up", "change up"]:
        return "changeup"
    if s in ["sinker", "two-seam fastball", "two-seam", "twoseam"]:
        return "sinker"
    if s in ["knucklecurve", "knuckle curve", "knuckle-curve"]:
        return "knucklecurve"

    # everything else -> drop
    return None


def load_frame_table(frame_csv: str) -> pd.DataFrame:
    df = pd.read_csv(frame_csv)
    if "pitch_type" not in df.columns:
        raise ValueError("frame_dataset.csv must contain a 'pitch_type' column")

    # normalize pitch types
    df["pitch_type_norm"] = df["pitch_type"].apply(normalize_pitch_type)

    # keep only the 6 canonical types
    df = df[df["pitch_type_norm"].isin(VALID_PITCH_TYPES)].copy()
    df.reset_index(drop=True, inplace=True)

    if "clip_id" not in df.columns:
        raise ValueError("frame_dataset.csv must contain a 'clip_id' column")

    print("Frame table loaded:")
    print("  rows:", len(df))
    print("  pitch_type distribution:")
    print(df["pitch_type_norm"].value_counts())
    return df


# -----------------------------
# Build clip-level features
# -----------------------------

def build_clip_level_features(
    df: pd.DataFrame,
    img_size=(64, 64),
    max_frames_per_clip: int = 16,
):
    """
    Aggregate frames into a single feature vector per clip by averaging
    flattened RGB pixels across up to `max_frames_per_clip` frames.

    Returns:
        X: (num_clips, D) float32
        y: (num_clips,)  string labels (canonical pitch types)
    """
    groups = df.groupby("clip_id")
    clip_ids = list(groups.groups.keys())

    X_list = []
    y_list = []

    print("Building clip-level representations...")
    for clip_id in tqdm(clip_ids, desc="Loading clip reps"):
        g = groups.get_group(clip_id)

        # most common normalized pitch type in this clip
        pitch_counts = Counter(g["pitch_type_norm"])
        label = pitch_counts.most_common(1)[0][0]

        # optionally limit number of frames per clip to speed things up
        g = g.iloc[:max_frames_per_clip]

        feat_sum = None
        count = 0

        for frame_path in g["frame_path"]:
            if not isinstance(frame_path, str) or not os.path.exists(frame_path):
                continue
            try:
                img = Image.open(frame_path).convert("RGB")
                img = img.resize(img_size)
                arr = np.array(img, dtype=np.float32) / 255.0
                vec = arr.flatten()
            except Exception:
                continue

            if feat_sum is None:
                feat_sum = vec
            else:
                feat_sum += vec
            count += 1

        if feat_sum is None or count == 0:
            # no valid frames
            continue

        feat_mean = feat_sum / float(count)
        X_list.append(feat_mean)
        y_list.append(label)

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list)

    print(f"Built clip-level feature matrix: {X.shape} labels: {y.shape}")
    return X, y


# -----------------------------
# Training + plotting
# -----------------------------

def train_and_evaluate(model, X_train, y_train, X_test, y_test, label_encoder, name: str):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)

    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    test_f1 = f1_score(y_test, y_test_pred, average="macro")

    print(f"\n=== {name} ===")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test  accuracy: {test_acc:.4f}")
    print(f"Train macro F1: {train_f1:.4f}")
    print(f"Test  macro F1: {test_f1:.4f}")

    print("\nClassification report (test set):")
    print(
        classification_report(
            y_test,
            y_test_pred,
            target_names=label_encoder.classes_,
            digits=4,
        )
    )

    return {
        "model": name,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_macro_f1": train_f1,
        "test_macro_f1": test_f1,
    }


def plot_baseline_metrics(results, out_dir: str):
    models = [r["model"] for r in results]
    train_acc = [r["train_accuracy"] for r in results]
    test_acc = [r["test_accuracy"] for r in results]
    train_f1 = [r["train_macro_f1"] for r in results]
    test_f1 = [r["test_macro_f1"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(models, train_acc, marker="o", label="Train Accuracy")
    plt.plot(models, test_acc, marker="o", label="Test Accuracy")
    plt.plot(models, train_f1, marker="s", label="Train Macro F1")
    plt.plot(models, test_f1, marker="s", label="Test Macro F1")

    plt.ylim(0.0, 1.05)
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.title("Baseline Model Performance (Clip-level, 6 Pitch Types)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "baseline_performance.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved baseline performance plot to {out_path}")


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--frame_csv",
        type=str,
        required=True,
        help="Path to frame_dataset.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        nargs=2,
        default=[64, 64],
        help="Resize frames to this (H W) before flattening",
    )
    parser.add_argument(
        "--max_frames_per_clip",
        type=int,
        default=16,
        help="Maximum number of frames per clip to average",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test fraction for train/test split",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load frame table & normalize labels to 6 pitch types
    df = load_frame_table(args.frame_csv)

    # 2) Build clip-level feature matrix
    X, y_str = build_clip_level_features(
        df,
        img_size=tuple(args.img_size),
        max_frames_per_clip=args.max_frames_per_clip,
    )

    # 3) Encode labels as integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    print("Label classes:", list(label_encoder.classes_))

    # 4) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42,
        stratify=y,
    )

    # 5) Train models
    results = []

    # Decision Tree (limit depth to avoid trivially 100% train acc)
    dt = DecisionTreeClassifier(
        max_depth=12,
        min_samples_leaf=10,
        random_state=42,
    )
    results.append(
        train_and_evaluate(dt, X_train, y_train, X_test, y_test, label_encoder, "DecisionTree")
    )

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42,
    )
    results.append(
        train_and_evaluate(rf, X_train, y_train, X_test, y_test, label_encoder, "RandomForest")
    )

    # 6) Save results CSV
    results_df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "classification_results_clip_level_6pitch.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved baseline metrics to {csv_path}")

    # 7) Plot combined train/test metrics
    plot_baseline_metrics(results, args.out_dir)


if __name__ == "__main__":
    main()
