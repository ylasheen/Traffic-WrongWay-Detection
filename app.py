"""
app.py
======
Streamlit demo for the AI Traffic Monitoring / Wrong-Way Vehicle Detection
project. Lets a visitor upload a short video clip, run it through the
YOLOv8 + ByteTrack pipeline, and view / download the annotated result.

Run locally:
    streamlit run app.py
"""

import os
import tempfile
import time

import streamlit as st

from wrongway_detector import WrongWayDetector, run_on_video
from configs import DRIVEWAY_ZONES, HIGHWAY_ZONES

CONFIGS = {
    "Single road / driveway": DRIVEWAY_ZONES,
    "Two-lane divided highway": HIGHWAY_ZONES,
}

st.set_page_config(
    page_title="AI Traffic Monitoring — Wrong-Way Detection",
    page_icon=None,
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Minimal professional styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
        .block-container {padding-top: 2.5rem; max-width: 780px;}
        h1 {font-weight: 600; letter-spacing: -0.5px;}
        .subtitle {color: #6b7280; font-size: 0.95rem; margin-top: -0.6rem; margin-bottom: 1.8rem;}
        .info-box {
            background-color: #f5f6f8;
            border: 1px solid #e2e4e8;
            border-radius: 8px;
            padding: 16px 20px;
            font-size: 0.92rem;
            line-height: 1.55;
            color: #374151;
            margin-bottom: 1.4rem;
        }
        .footer-note {color: #9ca3af; font-size: 0.8rem; margin-top: 2.5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Traffic Monitoring — Wrong-Way Vehicle Detection")
st.markdown(
    '<div class="subtitle">Vehicle detection, tracking, and direction analysis '
    "built by Youssef Lasheen — Lasheen Academy</div>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-box">
    This demo runs a computer vision pipeline that detects vehicles in traffic
    footage, tracks each one across frames, determines the direction it is
    heading, and flags any vehicle moving against the allowed direction of
    its lane. Upload a short clip below to see it applied to your own footage.
    <br><br>
    Processing runs on CPU in this environment, so a short clip (10–20 seconds)
    is recommended. Expect the analysis to take a few minutes depending on
    video length and resolution.
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Controls
# --------------------------------------------------------------------------- #
col1, col2 = st.columns(2)
with col1:
    config_label = st.selectbox("Road layout", list(CONFIGS.keys()))
with col2:
    max_seconds = st.slider("Limit clip length (seconds)", 5, 30, 15)

uploaded = st.file_uploader("Upload a traffic video", type=["mp4", "mov", "avi", "mkv"])

if uploaded is not None:
    if st.button("Run analysis", type="primary"):
        with tempfile.TemporaryDirectory() as tmpdir:
            in_path = os.path.join(tmpdir, "input.mp4")
            with open(in_path, "wb") as f:
                f.write(uploaded.read())

            out_path = os.path.join(tmpdir, "output.mp4")

            status = st.empty()
            progress_bar = st.progress(0)

            status.write("Loading model...")
            detector = WrongWayDetector(
                zones=CONFIGS[config_label],
                process_size=(960, 540),
            )

            import cv2
            cap = cv2.VideoCapture(in_path)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            cap.release()
            max_frames = int(max_seconds * src_fps)

            def progress_cb(i, total):
                total = max(total, 1)
                progress_bar.progress(min(i / total, 1.0))
                status.write(f"Processing frame {i} / {total}")

            t0 = time.time()
            summary = run_on_video(
                detector, in_path, out_path,
                progress_cb=progress_cb, max_frames=max_frames,
            )
            elapsed = time.time() - t0

            status.write(f"Done in {elapsed:.1f} seconds.")
            progress_bar.progress(1.0)

            st.subheader("Result")
            st.video(out_path)

            with open(out_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    data=f.read(),
                    file_name="traffic_analysis_output.mp4",
                    mime="video/mp4",
                )

            st.subheader("Summary")
            for zone, counts in summary.items():
                st.write(f"{zone} — correct: {counts['correct']}, wrong-way: {counts['wrong_way']}")

st.markdown(
    '<div class="footer-note">Detection: Ultralytics YOLOv8. Tracking: ByteTrack. '
    "Built as a computer vision portfolio project.</div>",
    unsafe_allow_html=True,
)
