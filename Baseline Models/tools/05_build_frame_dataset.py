import os
import pandas as pd
import argparse

def load_labels(labels_csv):
    df = pd.read_csv(labels_csv)
    mapping = {}
    for _, row in df.iterrows():
        clip_id = row["clip_id"]
        pitch_type = row["pitch_type"]
        speed = row.get("speed_mph", None)
        mapping[clip_id] = (pitch_type, speed)
    return mapping

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames_dir", required=True,
                    help="Root folder with per-clip frame subfolders")
    ap.add_argument("--labels_csv", required=True,
                    help="CSV with clip_id,pitch_type,speed_mph,...")
    ap.add_argument("--out_csv", required=True,
                    help="Output CSV listing frame_path + labels")
    args = ap.parse_args()

    label_dict = load_labels(args.labels_csv)

    rows = []
    for clip_folder in os.listdir(args.frames_dir):
        clip_path = os.path.join(args.frames_dir, clip_folder)
        if not os.path.isdir(clip_path):
            continue

        clip_id = clip_folder  # folder name == clip_id, e.g. RHlEdXq2DuI_E40GRXPSLG7N
        if clip_id not in label_dict:
            # no label for this clip, skip
            continue

        pitch_type, speed = label_dict[clip_id]

        for f in os.listdir(clip_path):
            if not f.lower().endswith(".jpg"):
                continue
            frame_path = os.path.join(clip_path, f)
            rows.append({
                "frame_path": frame_path,
                "clip_id": clip_id,
                "pitch_type": pitch_type,
                "speed_mph": speed
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print("Dataset ready →", args.out_csv)
    print("Total frames →", len(df))
    print("Pitch type counts:")
    print(df["pitch_type"].value_counts())

if __name__ == "__main__":
    main()
