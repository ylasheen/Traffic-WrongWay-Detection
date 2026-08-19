# 🚦 AI Traffic Monitoring — Wrong-Way Vehicle Detection

YOLOv8 + ByteTrack based system that detects vehicles, tracks them across frames,
works out which direction each one is actually heading, and flags any vehicle
moving **against** the allowed direction of its lane — with a live per-lane stats
panel (correct count / wrong-way count / Clear-or-Alert status), in the same
spirit as professional traffic-analytics dashboards.

Built by **Youssef Lasheen** (Lasheen Academy) as a computer-vision portfolio /
LinkedIn demo project.

---

## What's in this folder

| File | What it is |
|---|---|
| `Traffic_WrongWay_Detection.ipynb` | **Start here.** Self-contained notebook — runs in Google Colab or local Jupyter, no other files required. Already executed once end-to-end (outputs included) so you can see the result without running anything. |
| `wrongway_detector.py` | The reusable detection engine (`Zone`, `WrongWayDetector`, `run_on_video`). |
| `configs.py` | Two ready-made zone presets: `DRIVEWAY_ZONES` (used for the demo) and `HIGHWAY_ZONES` (template for a two-lane divided-highway look). |
| `run.py` | One-line command-line runner. |
| `requirements.txt` | Python dependencies. |
| `wrongway.mp4` | Sample input clip (real CCTV footage of a driveway wrong-way entry). |
| `traffic_analysis_output.mp4` | The annotated result — **this is the file to watch / post.** |

---

## Quick start (local machine)

```bash
pip install -r requirements.txt

# run on the included demo clip
python run.py --source wrongway.mp4 --output traffic_analysis_output.mp4

# quick test on just the first 150 frames (fast, good for iterating)
python run.py --source wrongway.mp4 --output test.mp4 --max-frames 150

# run on your own video
python run.py --source my_video.mp4 --output my_output.mp4
```

`ffmpeg` needs to be installed and on your PATH (it's used to re-encode the
output to standard H.264 so it plays everywhere / uploads cleanly to LinkedIn).
- Windows: `winget install ffmpeg` or download from ffmpeg.org
- Mac: `brew install ffmpeg`
- Linux: `sudo apt install ffmpeg`

## Quick start (Google Colab — no local setup, and free GPU)

1. Upload `Traffic_WrongWay_Detection.ipynb` to [colab.research.google.com](https://colab.research.google.com).
2. `Runtime → Change runtime type → T4 GPU` (optional but much faster).
3. `Runtime → Run all`. The notebook downloads the demo clip itself, or uncomment
   the `files.upload()` line in section 1 to use your own video instead.
4. Download `traffic_analysis_output.mp4` from the Colab file browser when it finishes.

---

## How it works

1. **Detect** — every frame is run through a pretrained YOLOv8 model
   (`yolov8n.pt` by default) restricted to vehicle classes: car, motorcycle, bus, truck.
2. **Track** — ByteTrack (built into `ultralytics`) assigns a stable ID to each
   vehicle across frames, so the same car keeps the same ID as it moves.
3. **Classify direction** — each tracked vehicle keeps a short history of its
   center point. Once it has moved far enough to have a clear heading, that
   heading is compared (via cosine similarity) to the *allowed* direction of the
   zone/lane it's in.
4. **Flag & count** — vehicles heading the wrong way are drawn in red with a
   "Wrong Way" tag (and "(Stopped)" if they've since stopped moving); correct
   ones are drawn in blue. Each zone keeps a running tally and a live
   "Clear" / "N Wrong-Way!" status.

## Adapting it to your own footage

Zones are defined in `configs.py` as a list of `Zone(label, allowed_direction,
polygon, panel_anchor, box_color)`. `polygon=None` means "the whole frame is one
zone" (used for the driveway demo). For a proper **two-lane divided highway**
matching the classic bird's-eye "Left Lane Stats / Right Lane Stats" look, use
the included `HIGHWAY_ZONES` template — it splits the frame down the middle,
with the left half expecting traffic to move **down** and the right half
expecting traffic to move **up**. Point `--config highway` at your own drone/
CCTV highway clip and adjust:

- the polygon coordinates, to match where the actual road lanes sit in your frame
- `allowed_direction` per lane, to match the real traffic flow
- `min_displacement_px` / `cos_threshold` in `WrongWayDetector(...)`, to tune
  sensitivity (see comments in `wrongway_detector.py`)

## Recording a demo video for LinkedIn

1. Run the pipeline (Colab + GPU is fastest for longer clips) and grab
   `traffic_analysis_output.mp4`.
2. Post it directly, or screen-record yourself narrating over it playing
   (OBS Studio / QuickTime / Xbox Game Bar) for a more personal walkthrough post.
3. Good caption angle: what the system *does*, what you *learned* going from
   plain object detection to behavior-aware analytics, and a question inviting
   engagement — e.g. *"What additional signal would you want a traffic-
   monitoring AI to surface?"*
4. Hashtags: `#ComputerVision #YOLO #TrafficAnalytics #SmartCities #DeepLearning #Python`

---

## Credits

- Detection model: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- Tracking: ByteTrack, bundled with `ultralytics`.
- Demo clip (`wrongway.mp4`): from the public repo
  [Ikshan-Tango/Vehicle_Wrong_Way_Detection_YoloV8](https://github.com/Ikshan-Tango/Vehicle_Wrong_Way_Detection_YoloV8),
  used here for demonstration/education. Swap in your own footage for anything
  you plan to publish as original work.
"# Traffic-WrongWay-Detection" 
