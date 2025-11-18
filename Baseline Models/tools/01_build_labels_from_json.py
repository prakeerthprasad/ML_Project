#!/usr/bin/env python3
import os, csv, json, argparse
from urllib.parse import urlparse, parse_qs

def youtube_id_from_url(u: str) -> str:
    try:
        q = parse_qs(urlparse(u).query)
        return q.get("v", [""])[0]
    except Exception:
        return ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmented_json", default="data/mlb-youtube-segmented.json",
                    help="Path to segmented JSON (dict keyed by uid)")
    ap.add_argument("--out_csv", default="data/labels.csv",
                    help="Output CSV path")
    ap.add_argument("--keep_classes", nargs="+",
                    default=["fastball","curveball","slider"],
                    help="Pitch classes to keep (case-insensitive)")
    ap.add_argument("--min_speed", type=float, default=0.0,
                    help="Drop rows with speed <= this")
    args = ap.parse_args()

    with open(args.segmented_json, "r") as f:
        obj = json.load(f)  # expects dict: {uid: {...}, ...}

    keep = set(c.lower() for c in args.keep_classes)

    rows = []
    for uid, rec in obj.items():
        # required fields
        url   = rec.get("url", "")
        v_id  = youtube_id_from_url(url)
        ptype = (rec.get("type") or "").strip().lower()
        speed = rec.get("speed", None)
        st    = rec.get("start", None)
        et    = rec.get("end", None)

        if not v_id or st is None or et is None:
            continue
        if ptype and keep and ptype not in keep:
            continue
        if speed is not None:
            try:
                speed = float(speed)
            except Exception:
                speed = None
        if speed is not None and speed <= args.min_speed:
            continue

        # clip_id = "<youtubeid>_<uid>"
        clip_id = f"{v_id}_{uid}"
        rows.append([
            clip_id,                 # clip_id
            ptype if ptype else "unknown",  # pitch_type
            "" if speed is None else speed, # speed_mph
            v_id,                    # video_id (YouTube)
            uid,                     # play_id (use the dict key)
            float(st), float(et)     # start_time, end_time
        ])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id","pitch_type","speed_mph","video_id","play_id","start_time","end_time"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {args.out_csv}")

if __name__ == "__main__":
    main()

    # python3 tools/01_build_labels_from_json.py --segmented_json data/mlb-youtube-segmented.json --out_csv data/labels.csv
