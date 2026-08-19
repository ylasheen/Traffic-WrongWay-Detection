"""
run.py
======
Command-line entry point.

Examples
--------
# Run on the included demo clip with the default "driveway" zone config:
python run.py --source wrongway.mp4 --output traffic_analysis_output.mp4

# Run on your own footage using the two-lane highway template:
python run.py --source my_highway.mp4 --output highway_out.mp4 --config highway

# Quick test on just the first 150 frames:
python run.py --source wrongway.mp4 --output test.mp4 --max-frames 150
"""

import argparse
import subprocess

from wrongway_detector import WrongWayDetector, run_on_video
from configs import DRIVEWAY_ZONES, HIGHWAY_ZONES

CONFIGS = {
    "driveway": DRIVEWAY_ZONES,
    "highway": HIGHWAY_ZONES,
}


def main():
    p = argparse.ArgumentParser(description="YOLOv8 wrong-way vehicle detection")
    p.add_argument("--source", required=True, help="path to input video")
    p.add_argument("--output", default="traffic_analysis_output.mp4", help="path to final H.264 output")
    p.add_argument("--config", default="driveway", choices=list(CONFIGS.keys()),
                    help="zone preset to use (see configs.py)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO checkpoint (yolov8n/s/m/l/x.pt)")
    p.add_argument("--width", type=int, default=1280, help="processing width")
    p.add_argument("--height", type=int, default=720, help="processing height")
    p.add_argument("--max-frames", type=int, default=None, help="limit frames (for quick tests)")
    p.add_argument("--no-reencode", action="store_true", help="skip the ffmpeg H.264 re-encode step")
    args = p.parse_args()

    zones = CONFIGS[args.config]
    detector = WrongWayDetector(
        zones=zones,
        model_path=args.model,
        process_size=(args.width, args.height),
    )

    def progress(i, total):
        if i % 50 == 0 or i == total:
            print(f"{i}/{total} frames")

    raw_path = args.output if args.no_reencode else "_raw_" + args.output
    summary = run_on_video(detector, args.source, raw_path,
                            progress_cb=progress, max_frames=args.max_frames)
    print("\nFinal summary:")
    for zone, counts in summary.items():
        print(f"  {zone}: correct={counts['correct']}  wrong_way={counts['wrong_way']}")

    if not args.no_reencode:
        print("\nRe-encoding to H.264 for universal playback...")
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "21",
            "-movflags", "+faststart", args.output,
        ], check=True)
        print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
