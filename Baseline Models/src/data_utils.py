import os
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def load_frame_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load frame_dataset.csv and do basic cleaning:
      - drop rows with missing frame_path or pitch_type
      - keep only rows whose frame file exists.
    """
    df = pd.read_csv(csv_path)

    if "frame_path" not in df.columns:
        raise ValueError("frame_dataset.csv must contain a 'frame_path' column.")
    if "pitch_type" not in df.columns:
        raise ValueError("frame_dataset.csv must contain a 'pitch_type' column.")
    if "clip_id" not in df.columns:
        raise ValueError("frame_dataset.csv must contain a 'clip_id' column for clip-level aggregation.")

    # Drop rows with NaNs in key columns
    df = df.dropna(subset=["frame_path", "pitch_type", "clip_id"])

    # Keep only existing frame files
    df = df[df["frame_path"].apply(lambda p: isinstance(p, str) and os.path.exists(p))]

    return df.reset_index(drop=True)


def build_clip_level_features(
    df: pd.DataFrame,
    img_size: Tuple[int, int] = (64, 64),
    max_clips: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build a clip-level feature matrix X and label vector y from frame-level data.

    For each clip_id:
      - load all its frames from disk
      - resize to img_size
      - convert to float32 [0,1]
      - flatten
      - average across frames to get a single feature vector per clip

    Returns:
      X: (n_clips, D) float32
      y: (n_clips,) with pitch_type labels (strings)
      clip_ids: list of clip_id values in the same order as X/y
    """
    if "clip_id" not in df.columns:
        raise ValueError("DataFrame must contain 'clip_id' column.")
    if "frame_path" not in df.columns:
        raise ValueError("DataFrame must contain 'frame_path' column.")
    if "pitch_type" not in df.columns:
        raise ValueError("DataFrame must contain 'pitch_type' column.")

    # Group by clip_id
    grouped = df.groupby("clip_id")
    clip_ids = list(grouped.groups.keys())
    clip_ids = sorted(clip_ids)

    # Optional subsampling for speed / memory
    if max_clips is not None and len(clip_ids) > max_clips:
        rng = np.random.RandomState(42)
        clip_ids = list(rng.choice(clip_ids, size=max_clips, replace=False))
        print(f"Subsampled to {len(clip_ids)} clips for baseline.")

    X_list = []
    y_list = []
    kept_clip_ids: List[str] = []

    print(f"Loading clip reps: {len(clip_ids)} clips...")
    for cid in tqdm(clip_ids, desc="Loading clip reps"):
        sub = grouped.get_group(cid)

        # All frames for this clip
        frame_paths = sub["frame_path"].tolist()
        pitch_types = sub["pitch_type"].tolist()

        # Choose the pitch_type (they should all match per clip)
        # If not, we take the most frequent.
        pt_series = pd.Series(pitch_types)
        pitch_type = pt_series.mode().iloc[0]

        frame_feats = []
        for fp in frame_paths:
            if not (isinstance(fp, str) and os.path.exists(fp)):
                continue
            try:
                img = Image.open(fp).convert("RGB")
                img = img.resize(img_size)
                arr = np.array(img, dtype=np.float32) / 255.0
                feat = arr.flatten()
                frame_feats.append(feat)
            except Exception:
                continue

        if len(frame_feats) == 0:
            # No valid frames; skip this clip
            continue

        # Average over frames to get a single clip-level representation
        clip_feat = np.stack(frame_feats, axis=0).mean(axis=0)

        X_list.append(clip_feat)
        y_list.append(pitch_type)
        kept_clip_ids.append(str(cid))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=object)

    print(f"Built clip-level feature matrix: {X.shape} labels: {y.shape}")
    return X, y, kept_clip_ids
