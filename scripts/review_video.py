"""Create labeled contact sheets from the original top/head view."""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="+", type=int, default=list(range(30)))
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--segment", nargs=2, type=float, default=[0.08, 0.97])
    parser.add_argument("--crop", nargs=4, type=float)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    size = (args.width, round(args.width * 9 / 16))
    rows = []
    for episode in args.episodes:
        video = next((args.raw / str(episode)).glob("videos/*/top.mp4"))
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        requested = np.linspace(*args.segment, args.samples) * (frames - 1)
        row = []
        for frame in requested.astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
            ok, image = cap.read()
            if not ok:
                raise RuntimeError(f"Unusable frame {frame} in episode {episode}")
            if args.crop:
                h, w = image.shape[:2]
                x1, y1, x2, y2 = args.crop
                image = image[round(y1*h):round(y2*h), round(x1*w):round(x2*w)]
            frame_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
            footer = np.zeros((22, args.width, 3), dtype=np.uint8)
            cv2.putText(footer, f"EP {episode:02d}  {frame/fps:05.1f}s", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1, cv2.LINE_AA)
            row.append(np.vstack((frame_image, footer)))
        cap.release()
        rows.append(np.hstack(row))
    for index in range(0, len(rows), 5):
        out = np.vstack(rows[index:index + 5])
        cv2.imwrite(str(args.output / f"top_{args.episodes[index]:02d}.jpg"), out,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])


if __name__ == "__main__":
    main()
