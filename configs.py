"""
configs.py
==========
Ready-made `Zone` presets for wrongway_detector.WrongWayDetector.

PROCESS_W / PROCESS_H must match the `process_size` you pass to
WrongWayDetector, since zone polygons are defined in that pixel space.
"""

from wrongway_detector import Zone

PROCESS_W, PROCESS_H = 1280, 720


# --------------------------------------------------------------------------- #
# Preset 1 — "driveway_oneway"
# Used for the included demo clip (a home CCTV camera watching a one-way
# entrance / driveway). One monitored zone covering the whole frame: normal
# traffic flows away from the camera (toward the top of frame); a vehicle
# heading back down toward the camera/private area is flagged Wrong Way.
# --------------------------------------------------------------------------- #

DRIVEWAY_ZONES = [
    Zone(
        label="Driveway Entrance",
        allowed_direction="up",   # legal flow = away from camera (up-frame)
        polygon=None,             # None = whole frame is monitored
        panel_anchor=(20, 20),
        box_color=(255, 176, 0),
    ),
]


# --------------------------------------------------------------------------- #
# Preset 2 — "highway_two_lane"
# A TEMPLATE matching the classic bird's-eye "Left Lane Stats / Right Lane
# Stats" divided-highway look (two lanes of opposite-direction traffic,
# split by a line down the middle of the frame). Point this at your own
# drone/CCTV highway footage and adjust the polygon coordinates + the split
# x-position to match your camera framing.
#
#   Left  lane (left half of frame)  -> correct direction = DOWN (toward viewer)
#   Right lane (right half of frame) -> correct direction = UP   (away from viewer)
# --------------------------------------------------------------------------- #

_split_x = PROCESS_W // 2

HIGHWAY_ZONES = [
    Zone(
        label="Left Lane",
        allowed_direction="down",
        polygon=[(0, 0), (_split_x, 0), (_split_x, PROCESS_H), (0, PROCESS_H)],
        panel_anchor=(20, 20),
        box_color=(255, 176, 0),
    ),
    Zone(
        label="Right Lane",
        allowed_direction="up",
        polygon=[(_split_x, 0), (PROCESS_W, 0), (PROCESS_W, PROCESS_H), (_split_x, PROCESS_H)],
        panel_anchor=(PROCESS_W - 280, 20),
        box_color=(0, 200, 255),
    ),
]
