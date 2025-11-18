#!/usr/bin/env python3
import pandas as pd, argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default="data/labels.csv",
                    help="Path to labels.csv created from segmented.json")
    ap.add_argument("--out_csv", default="data/labels.clean.csv",
                    help="Filtered valid labels output path")
    ap.add_argument("--min_duration", type=float, default=0.5,
                    help="Minimum valid clip duration in seconds")
    ap.add_argument("--max_duration", type=float, default=12.0,
                    help="Maximum valid clip duration in seconds")
    ap.add_argument("--min_speed", type=float, default=40.0,
                    help="Minimum plausible pitch speed (mph)")
    ap.add_argument("--max_speed", type=float, default=110.0,
                    help="Maximum plausible pitch speed (mph)")
    args = ap.parse_args()

    if not os.path.exists(args.labels_csv):
        raise FileNotFoundError(f"Cannot find {args.labels_csv}")

    df = pd.read_csv(args.labels_csv)
    n0 = len(df)
    print(f"Loaded {n0} rows from {args.labels_csv}")

    # drop obvious missing
    df = df.dropna(subset=["video_id", "start_time", "end_time"])
    # duration sanity
    df["duration"] = df["end_time"] - df["start_time"]
    df = df[(df["duration"] >= args.min_duration) & (df["duration"] <= args.max_duration)]

    # pitch speed sanity
    if "speed_mph" in df.columns:
        df = df[(df["speed_mph"].isna()) |
                ((df["speed_mph"] >= args.min_speed) & (df["speed_mph"] <= args.max_speed))]

    # drop duplicates
    df = df.drop_duplicates(subset=["clip_id"])

    n1 = len(df)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Cleaned labels saved to {args.out_csv}  ({n1}/{n0} valid rows)")

    # show quick stats
    if "pitch_type" in df.columns:
        print("\nPitch type counts:")
        print(df["pitch_type"].value_counts())

    if "speed_mph" in df.columns:
        print("\nSpeed stats (mph):")
        print(df["speed_mph"].describe())

if __name__ == "__main__":
    main()

    # python3 tools/00_verify_labels.py --labels_csv data/labels.csv

