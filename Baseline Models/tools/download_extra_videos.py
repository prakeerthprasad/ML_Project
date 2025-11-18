import os
import subprocess

# Folder where raw YouTube videos are stored
RAW_DIR = os.path.join("data", "raw_videos")

# Make sure the folder exists
os.makedirs(RAW_DIR, exist_ok=True)

# Extra video IDs to add for diversity
EXTRA_VIDEO_IDS = [
    "6pBQVjveD6o",
    "F2QVELDu_Y4",
    "EX98PC6yuRk",
    "TI7AgD-BC6M",
    "Sq2Igg3XjbM",
    "_6fD6Lx7tQ0",
    "Nvv1gdF_ed8",
    "xcNCZK_g4_A",
]

def download_video(video_id: str):
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_path = os.path.join(RAW_DIR, f"{video_id}.mp4")

    # Skip if already downloaded
    if os.path.exists(out_path):
        print(f"[SKIP] {video_id} already exists at {out_path}")
        return

    # yt-dlp command using Safari cookies (you’re logged into YouTube in Safari)
    cmd = [
        "yt-dlp",
        "--cookies-from-browser", "safari",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best",
        "--merge-output-format", "mp4",
        "-o", out_path,
        url,
    ]

    print("\n=== Downloading", video_id, "===")
    print(" ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print(f"[OK] Downloaded {video_id} → {out_path}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to download {video_id}: {e}")


def main():
    print("Raw video directory:", RAW_DIR)
    for vid in EXTRA_VIDEO_IDS:
        download_video(vid)


if __name__ == "__main__":
    main()
