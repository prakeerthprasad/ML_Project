import pandas as pd, os, argparse, subprocess, math

def ffmpeg_cut(src, dst, start, end):
    dur = max(0.1, end - start)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", src,
        "-t", f"{dur:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",  # ensure both video + audio if present
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23",
        "-c:a", "aac", "-movflags", "+faststart",
        "-loglevel", "error",
        dst
    ]
    subprocess.run(cmd, check=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--videos_dir", required=True)  # data/raw_videos
    ap.add_argument("--out_dir", required=True)     # data/clips_mp4
    ap.add_argument("--max_clips", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.labels_csv)
    if args.max_clips > 0:
        df = df.sample(n=args.max_clips, random_state=42).reset_index(drop=True)

    n_ok = 0
    for _, r in df.iterrows():
        vid, pid = r["video_id"], r["play_id"]
        st, et = float(r["start_time"]), float(r["end_time"])
        src = os.path.join(args.videos_dir, f"{vid}.mp4")
        dst = os.path.join(args.out_dir, f"{vid}_{str(pid)}.mp4")
        if not os.path.exists(src):
            print(f"missing video: {src}")
            continue
        if os.path.exists(dst):
            n_ok += 1
            continue
        try:
            ffmpeg_cut(src, dst, st, et)
            n_ok += 1
        except Exception as e:
            print("ffmpeg error:", e)
    print(f"cut {n_ok} / {len(df)} clips -> {args.out_dir}")

if __name__ == "__main__":
    main()
    # python3 tools/03_cut_clips.py --labels_csv data/labels.csv --videos_dir data/raw_videos --out_dir data/clips_mp4 --max_clips 150