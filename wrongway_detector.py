"""
wrongway_detector.py
=====================
AI Traffic Monitoring: multi-lane / multi-zone vehicle detection, tracking,
direction classification, and wrong-way alerting — built with YOLOv8 (Ultralytics)
+ ByteTrack.

Inspired by the "Left Lane Stats / Right Lane Stats" traffic-analytics style
(vehicle boxes colored by status, per-zone counters, live status banner).

Author: built for Youssef Lasheen (Lasheen Academy) as a portfolio / LinkedIn demo.

Core idea
---------
1. Detect vehicles every frame with YOLOv8 (car, motorcycle, bus, truck).
2. Track each vehicle across frames with ByteTrack -> stable ID per vehicle.
3. Keep a short history of each vehicle's center point.
4. Once a vehicle has moved enough to have a clear heading, compare that
   heading to the "allowed direction" of the zone (lane) it is in.
5. If the heading points the opposite way -> flag as WRONG WAY (draw in red/
   magenta, add to that zone's wrong-way tally). Otherwise draw in blue/green
   as a correctly-flowing vehicle.
6. Draw a live stats panel per zone (correct count, wrong-way count, status).

Everything here is deliberately generic: you plug in your own `Zone` list to
match ANY camera angle (a single one-way road, a two-lane divided highway,
a roundabout approach, etc). See `configs.py` for two ready-made presets.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[float, float]

# COCO class ids for vehicles (from the pretrained YOLOv8 model)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# --------------------------------------------------------------------------- #
# Direction helpers
# --------------------------------------------------------------------------- #

_DIRECTION_VECTORS = {
    "up": (0.0, -1.0),
    "down": (0.0, 1.0),
    "left": (-1.0, 0.0),
    "right": (1.0, 0.0),
}


def _resolve_direction(direction) -> np.ndarray:
    """Accepts 'up'/'down'/'left'/'right' or an (dx, dy) tuple -> unit vector."""
    if isinstance(direction, str):
        vec = _DIRECTION_VECTORS[direction.lower()]
    else:
        vec = direction
    v = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else v


# --------------------------------------------------------------------------- #
# Zone definition
# --------------------------------------------------------------------------- #


@dataclass
class Zone:
    """A monitored lane / region of the frame.

    polygon: list of (x, y) points in pixel coordinates of the PROCESSING
        resolution (see WrongWayDetector.process_size). If None, the zone
        covers the entire frame (use this for single-road / driveway setups).
    allowed_direction: 'up' | 'down' | 'left' | 'right' | (dx, dy) tuple.
        The direction a vehicle in this zone SHOULD be moving.
    label: short display name, e.g. "Left Lane", "Right Lane", "Driveway".
    panel_anchor: (x, y) top-left pixel where this zone's stats panel is drawn.
    box_color: BGR color used for the zone outline / correct-direction boxes.
    """

    label: str
    allowed_direction: object
    polygon: Optional[List[Point]] = None
    panel_anchor: Point = (20, 20)
    box_color: Tuple[int, int, int] = (255, 176, 0)  # blue-ish (BGR)

    def __post_init__(self):
        self._dir_vec = _resolve_direction(self.allowed_direction)
        self._poly_np = (
            np.array(self.polygon, dtype=np.int32) if self.polygon else None
        )

    def contains(self, pt: Point) -> bool:
        if self._poly_np is None:
            return True
        return cv2.pointPolygonTest(self._poly_np, pt, False) >= 0

    def heading_is_correct(self, motion_vec: np.ndarray, cos_threshold: float) -> bool:
        """cos_threshold: e.g. 0.0 means anything within 90 degrees counts as
        'correct'; a vehicle moving more than 90 degrees off the allowed
        heading is flagged wrong-way. Use a small negative number
        (e.g. -0.15) to be a bit more lenient with noisy tracks."""
        norm = np.linalg.norm(motion_vec)
        if norm < 1e-6:
            return True  # no motion -> can't judge, default to "not wrong"
        cos_sim = float(np.dot(motion_vec / norm, self._dir_vec))
        return cos_sim >= cos_threshold


# --------------------------------------------------------------------------- #
# Per-track state
# --------------------------------------------------------------------------- #


@dataclass
class _TrackState:
    history: Deque[Point] = field(default_factory=lambda: deque(maxlen=18))
    cls_name: str = "car"
    zone_label: Optional[str] = None
    status: str = "unclassified"  # unclassified | correct | wrong
    stopped: bool = False
    last_seen_frame: int = 0
    counted: bool = False  # has this ID already been tallied into a zone's stats?


# --------------------------------------------------------------------------- #
# Main detector
# --------------------------------------------------------------------------- #


class WrongWayDetector:
    def __init__(
        self,
        zones: Sequence[Zone],
        model_path: str = "yolov8n.pt",
        process_size: Optional[Tuple[int, int]] = (1280, 720),
        conf: float = 0.35,
        tracker: str = "bytetrack.yaml",
        classes: Sequence[int] = tuple(VEHICLE_CLASSES.keys()),
        min_displacement_px: float = 14.0,
        cos_threshold: float = -0.10,
        stop_speed_px: float = 1.5,
    ):
        """
        zones: list of Zone objects (lanes / monitored regions).
        model_path: any YOLOv8/YOLO11 detection checkpoint (.pt).
        process_size: (w, h) the video is resized to before inference &
            drawing. Smaller = faster on CPU. Set to None to keep native size.
        min_displacement_px: minimum total movement (in the processing
            resolution) before we trust a vehicle's heading enough to judge it.
        cos_threshold: see Zone.heading_is_correct.
        stop_speed_px: below this recent per-frame speed, a vehicle is
            considered "Stopped" (drawn as e.g. "Wrong Way (Stopped)").
        """
        self.zones = list(zones)
        self.model = YOLO(model_path)
        self.process_size = process_size
        self.conf = conf
        self.tracker = tracker
        self.classes = list(classes)
        self.min_displacement_px = min_displacement_px
        self.cos_threshold = cos_threshold
        self.stop_speed_px = stop_speed_px

        self.tracks: Dict[int, _TrackState] = defaultdict(_TrackState)
        self.zone_correct_ids: Dict[str, set] = {z.label: set() for z in self.zones}
        self.zone_wrong_ids: Dict[str, set] = {z.label: set() for z in self.zones}
        self.frame_idx = 0

    # ---------------------------------------------------------------- #
    def _find_zone(self, pt: Point) -> Optional[Zone]:
        for z in self.zones:
            if z.contains(pt):
                return z
        return None

    # ---------------------------------------------------------------- #
    def _update_track(self, track_id: int, box, cls_name: str):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        st = self.tracks[track_id]
        st.history.append((cx, cy))
        st.cls_name = cls_name
        st.last_seen_frame = self.frame_idx

        zone = self._find_zone((cx, cy))
        if zone is None:
            return st, None

        st.zone_label = zone.label

        if len(st.history) >= 2:
            p_old = np.array(st.history[0])
            p_new = np.array(st.history[-1])
            motion_vec = p_new - p_old
            total_disp = float(np.linalg.norm(motion_vec))

            # recent per-frame speed (last 2 points) -> "stopped" detection
            p_prev = np.array(st.history[-2])
            recent_speed = float(np.linalg.norm(p_new - p_prev))
            st.stopped = recent_speed < self.stop_speed_px

            if total_disp >= self.min_displacement_px:
                is_correct = zone.heading_is_correct(motion_vec, self.cos_threshold)
                st.status = "correct" if is_correct else "wrong"

                if not st.counted:
                    st.counted = True
                    if is_correct:
                        self.zone_correct_ids[zone.label].add(track_id)
                    else:
                        self.zone_wrong_ids[zone.label].add(track_id)
                elif not is_correct and track_id not in self.zone_wrong_ids[zone.label]:
                    # a vehicle that was previously fine but has now turned
                    # the wrong way (e.g. a U-turn) gets flagged too
                    self.zone_wrong_ids[zone.label].add(track_id)
                    self.zone_correct_ids[zone.label].discard(track_id)
                    st.status = "wrong"

        return st, zone

    # ---------------------------------------------------------------- #
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run detection+tracking on one BGR frame, return an annotated copy."""
        self.frame_idx += 1

        if self.process_size is not None:
            frame = cv2.resize(frame, self.process_size)

        results = self.model.track(
            frame,
            persist=True,
            conf=self.conf,
            classes=self.classes,
            tracker=self.tracker,
            verbose=False,
        )[0]

        annotated = frame.copy()
        self._draw_zone_outlines(annotated)

        active_wrong_by_zone = defaultdict(int)

        if results.boxes is not None and results.boxes.id is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            ids = results.boxes.id.cpu().numpy().astype(int)
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for box, tid, c in zip(boxes, ids, clss):
                cls_name = VEHICLE_CLASSES.get(c, "vehicle")
                st, zone = self._update_track(tid, box, cls_name)
                self._draw_vehicle(annotated, box, tid, st, zone)
                if zone is not None and st.status == "wrong":
                    active_wrong_by_zone[zone.label] += 1

        self._draw_stats_panels(annotated, active_wrong_by_zone)
        return annotated

    # ---------------------------------------------------------------- #
    def _draw_zone_outlines(self, frame: np.ndarray):
        for z in self.zones:
            if z.polygon:
                pts = np.array(z.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], True, z.box_color, 2, cv2.LINE_AA)

    def _draw_vehicle(self, frame, box, tid, st: _TrackState, zone: Optional[Zone]):
        x1, y1, x2, y2 = map(int, box)
        if st.status == "wrong":
            color = (60, 40, 230)  # red/magenta (BGR) - wrong way
        elif st.status == "correct":
            color = (255, 176, 0)  # blue - correct
        else:
            color = (0, 200, 255)  # amber - not yet classified

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label = f"{st.cls_name} ID:{tid}"
        cv2.putText(
            frame, label, (x1, max(20, y1 - 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )
        if st.status == "wrong":
            tag = "Wrong Way (Stopped)" if st.stopped else "Wrong Way"
            cv2.putText(
                frame, tag, (x1, max(40, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 40, 230), 2, cv2.LINE_AA,
            )

    def _draw_stats_panels(self, frame, active_wrong_by_zone: Dict[str, int]):
        for z in self.zones:
            correct_n = len(self.zone_correct_ids[z.label])
            wrong_n = len(self.zone_wrong_ids[z.label])
            live_wrong = active_wrong_by_zone.get(z.label, 0)

            x, y = int(z.panel_anchor[0]), int(z.panel_anchor[1])
            w, h = 260, 100
            overlay = frame.copy()
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (20, 20, 20), -1)
            frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

            cv2.putText(frame, z.label, (x + 12, y + 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Correct: {correct_n}", (x + 12, y + 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 176, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f"Wrong Way: {wrong_n}", (x + 12, y + 72),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 40, 230), 2, cv2.LINE_AA)

            if live_wrong > 0:
                status_txt, status_col = f"Status: {live_wrong} Wrong-Way!", (60, 40, 230)
            else:
                status_txt, status_col = "Status: Clear", (0, 200, 0)
            cv2.putText(frame, status_txt, (x + 12, y + 94),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_col, 2, cv2.LINE_AA)

    # ---------------------------------------------------------------- #
    def summary(self) -> Dict[str, Dict[str, int]]:
        return {
            z.label: {
                "correct": len(self.zone_correct_ids[z.label]),
                "wrong_way": len(self.zone_wrong_ids[z.label]),
            }
            for z in self.zones
        }


# --------------------------------------------------------------------------- #
# Convenience: run on a whole video file
# --------------------------------------------------------------------------- #


def run_on_video(
    detector: WrongWayDetector,
    source_path: str,
    output_path: str,
    fps_override: Optional[float] = None,
    progress_cb: Optional[Callable[[int, int], None]] = None,
    max_frames: Optional[int] = None,
) -> Dict[str, Dict[str, int]]:
    """Process a video file end-to-end and write an annotated copy.

    Returns the final per-zone summary dict.
    """
    cap = cv2.VideoCapture(source_path)
    if not cap.isOpened():
        raise IOError(f"Could not open source video: {source_path}")

    src_fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames:
        total = min(total, max_frames)

    w, h = detector.process_size or (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, (w, h))

    t0 = time.time()
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotated = detector.process_frame(frame)
        writer.write(annotated)
        i += 1
        if progress_cb:
            progress_cb(i, total)
        if max_frames and i >= max_frames:
            break

    cap.release()
    writer.release()
    elapsed = time.time() - t0
    print(f"Processed {i} frames in {elapsed:.1f}s ({i/elapsed:.2f} FPS) -> {output_path}")
    return detector.summary()
