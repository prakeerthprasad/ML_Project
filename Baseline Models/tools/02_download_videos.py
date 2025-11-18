import pandas as pd, os, subprocess, argparse, shlex

# HLS formats we saw:
# 91  -> 256x144
# 93  -> 640x360
# 300 -> 1280x720 60fps
FORMAT = "300/93/91/best"

BROWSER = "safari"  # we confirmed this works

def run_cmd(cmd):
    print(" ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd).returncode == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_videos", type=int, default=0)  # 0 = all
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.labels_csv)
    vids = sorted(df["video_id"].dropna().unique().tolist())
    if args.max_videos > 0:
        vids = vids[:args.max_videos]

    ok, fail = 0, 0
    for v in vids:
        out = os.path.join(args.out_dir, f"{v}.mp4")
        if os.path.exists(out):
            print(f"skip {out}")
            ok += 1
            continue

        url = f"https://www.youtube.com/watch?v={v}"
        cmd = [
            "yt-dlp",
            "--no-playlist",
            "--cookies-from-browser", BROWSER,
            "-f", FORMAT,
            "--merge-output-format", "mp4",
            "-o", out,
            "-N", "4",
            url,
        ]

        if run_cmd(cmd):
            ok += 1
        else:
            print(f"[ERROR] failed to download {v}")
            fail += 1

    print(f"\nDONE: success={ok}, failed={fail}, total={len(vids)}")

if __name__ == "__main__":
    main()
