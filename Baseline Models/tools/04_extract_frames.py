import os
import cv2
import argparse

def extract_frames(clip_path, out_dir, fps=2):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(clip_path)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    step = int(frame_rate / fps) if frame_rate > 0 else 10

    frame_index = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % step == 0:
            out_path = os.path.join(out_dir, f"frame_{frame_index}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_index += 1

    cap.release()
    return saved

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips_dir", required=True, help="Folder with .mp4 clips")
    ap.add_argument("--out_dir", required=True, help="Output frame folder")
    ap.add_argument("--fps", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    clips = [f for f in os.listdir(args.clips_dir) if f.endswith(".mp4")]
    print(f"Found {len(clips)} clips.")

    total_frames = 0
    for clip in clips:
        clip_path = os.path.join(args.clips_dir, clip)
        clip_out = os.path.join(args.out_dir, clip.replace(".mp4", ""))
        saved = extract_frames(clip_path, clip_out, fps=args.fps)
        print(f"{clip}: {saved} frames")
        total_frames += saved

    print(f"\nDONE — Total frames extracted: {total_frames}")

if __name__ == "__main__":
    main()
