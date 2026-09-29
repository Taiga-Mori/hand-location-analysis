import os
import signal
import subprocess
import sys
from pathlib import Path

# Streamlit runs this file as a script, so make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from handloc import HandLocationAnalyzer
from handloc.config import ConfigManager
from handloc.constants import DEFAULT_ORIENTATION, DEFAULT_POSE_MODEL, HANDS, ORIENTATION_MODES, POSE_MODELS, VIDEO_EXTENSIONS
from handloc.progress import format_elapsed

APP_TITLE = "Hand Location Analysis"
APP_CAPTION = "Nine-sector hand location annotation for frontal conversation videos (Choi et al., 2014, Fig. 3)"
TRACKER_KEYS = ["track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh"]
SECTOR_DIAGRAM = """
| | person's right | center | person's left |
|---|:---:|:---:|:---:|
| above shoulders | 1 | 2 | 3 |
| shoulders–hips | 4 | **5** (torso) | 6 |
| below hips | 7 | 8 | 9 |
"""


def terminate_current_process() -> None:
    try:
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception:
        os._exit(0)


def browse_input_file() -> tuple[Path | None, str | None]:
    if sys.platform == "win32":
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            patterns = " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
            selected = filedialog.askopenfilename(
                title="Select input video",
                filetypes=[("Video files", patterns), ("All files", "*.*")],
            )
            root.destroy()
        except Exception as exc:
            return None, str(exc)
        return (Path(selected), None) if selected else (None, "No file was selected.")

    if sys.platform != "darwin":
        return None, "Native file browsing is only available on macOS and Windows."
    try:
        result = subprocess.run(
            [
                "osascript",
                "-e",
                'POSIX path of (choose file with prompt "Select input video" of type {"public.movie", "public.video"})',
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        error_text = (exc.stderr or exc.stdout or str(exc)).strip()
        return None, error_text or "The native file dialog could not be opened."
    except Exception as exc:
        return None, str(exc)
    selected = result.stdout.strip()
    return (Path(selected), None) if selected else (None, "No file was selected.")


def reset_to_start() -> None:
    st.session_state.state = "start"


def render_header() -> None:
    col_title, col_quit = st.columns([5, 1], vertical_alignment="center")
    with col_title:
        st.title(APP_TITLE)
        st.caption(APP_CAPTION)
    with col_quit:
        if st.button("Quit", width="stretch"):
            terminate_current_process()


def init_state() -> None:
    defaults = {
        "state": "start",
        "results": None,
        "error_message": None,
        "selected_input_path": "",
        "browse_error_message": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if "analyzer" not in st.session_state:
        with st.spinner("Loading..."):
            st.session_state.analyzer = HandLocationAnalyzer()


def render_start(analyzer: HandLocationAnalyzer) -> None:
    ss = st.session_state
    st.subheader("Basic Settings")
    col_input, col_browse = st.columns([5, 1])
    with col_input:
        input_path_str = st.text_input("Input video", value=ss.selected_input_path).strip()
    with col_browse:
        st.write("")
        st.write("")
        if st.button("Browse", width="stretch"):
            selected_path, browse_error = browse_input_file()
            if selected_path is not None:
                ss.selected_input_path = str(selected_path)
                ss.browse_error_message = None
                st.rerun()
            if browse_error and browse_error != "No file was selected.":
                ss.browse_error_message = browse_error
    if ss.browse_error_message:
        st.warning(f"Native Browse did not complete. You can still paste a path.\n\nDetails: {ss.browse_error_message}")

    input_path = Path(input_path_str).expanduser() if input_path_str else None
    output_parent = None
    output_dir = None
    if input_path is not None:
        default_output = ConfigManager.default_output_dir(input_path)
        output_parent = default_output.parent
        output_name = st.text_input("Output folder name", value=default_output.name).strip() or default_output.name
        output_dir = output_parent / output_name

    device_options = analyzer.device_options
    saved_device = ss.get("device", device_options[0])
    device = st.selectbox(
        "Device",
        options=device_options,
        index=device_options.index(saved_device) if saved_device in device_options else 0,
    )

    valid_input = False
    if input_path is None:
        st.info("Choose an input video to begin.")
    elif not input_path.exists():
        st.warning(f"File not found: `{input_path}`")
    elif input_path.suffix.lower() not in VIDEO_EXTENSIONS:
        st.warning(f"Supported inputs: {', '.join(sorted(VIDEO_EXTENSIONS))}")
    else:
        valid_input = True
        st.caption(f"Output directory: `{output_dir}`")

    with st.expander("Sector definition", expanded=False):
        st.markdown(
            "The torso rectangle spans both **shoulder** keypoints horizontally and runs from the shoulder line "
            "down to the **hip** line. Its extended edges split the space into nine sectors, numbered from the "
            "person's own viewpoint. Each hand's location is the sector containing its **wrist** keypoint."
        )
        st.markdown(SECTOR_DIAGRAM)
        st.caption("When hips are hidden (e.g. behind a table), the hip line is estimated from the shoulder width (dashed line in the video).")

    with st.expander("Detailed Settings", expanded=False):
        detection_tab, location_tab, tracker_tab, output_tab = st.tabs(["Detection", "Location", "Tracking", "Output"])
        with detection_tab:
            pose_model = st.selectbox(
                "Pose model",
                POSE_MODELS,
                index=POSE_MODELS.index(ss.get("pose_model", DEFAULT_POSE_MODEL)),
                help="Ultralytics YOLO26 pose models. Weights are downloaded to ~/.handloc on first use.",
            )
            person_det_thresh = st.slider("Person threshold", 0.0, 1.0, float(ss.get("person_det_thresh", 0.5)), 0.05)
            keypoint_conf_thresh = st.slider(
                "Keypoint threshold",
                0.0,
                1.0,
                float(ss.get("keypoint_conf_thresh", 0.5)),
                0.05,
                help="Keypoints below this confidence are treated as missing.",
            )
            use_source_fps = st.checkbox("Track every frame (source FPS)", value=bool(ss.get("use_source_fps", True)))
            person_target_fps = None
            if not use_source_fps:
                person_target_fps = st.number_input(
                    "Tracking FPS", min_value=1.0, max_value=240.0, value=float(ss.get("person_target_fps") or 15.0), step=1.0
                )
        with location_tab:
            orientation = st.selectbox(
                "Orientation",
                ORIENTATION_MODES,
                index=ORIENTATION_MODES.index(ss.get("orientation", DEFAULT_ORIENTATION)),
                help="frontal: the person's right side is the image left. auto: decide it from the shoulder order each frame.",
            )
            smoothing_window = st.number_input(
                "Keypoint smoothing window (frames)", min_value=1, max_value=61, value=int(ss.get("smoothing_window", 5)), step=1
            )
            max_gap_seconds = st.number_input(
                "Max interpolated gap (s)", min_value=0.0, max_value=10.0, value=float(ss.get("max_gap_seconds", 0.5)), step=0.1
            )
            min_segment_seconds = st.number_input(
                "Min segment duration (s)",
                min_value=0.0,
                max_value=10.0,
                value=float(ss.get("min_segment_seconds", 0.2)),
                step=0.05,
                help="Shorter location segments are merged into neighbouring segments to suppress flicker at sector borders.",
            )
            min_track_seconds = st.number_input(
                "Min track duration (s)", min_value=0.0, max_value=60.0, value=float(ss.get("min_track_seconds", 1.0)), step=0.5
            )
            hip_ratio = st.number_input(
                "Fallback torso ratio",
                min_value=0.3,
                max_value=3.0,
                value=float(ss.get("hip_ratio", 1.3)),
                step=0.05,
                help="(hip height - shoulder height) / shoulder width, used only when a person's hips are never visible.",
            )
        with tracker_tab:
            tracker_defaults = analyzer.config_manager.load_tracker_defaults()
            tracker_updates = {}
            for key in TRACKER_KEYS:
                default = tracker_defaults.get(key)
                if isinstance(default, int) and not isinstance(default, bool):
                    tracker_updates[key] = int(st.number_input(key, min_value=1, max_value=600, value=int(ss.get(key, default)), step=1))
                else:
                    tracker_updates[key] = float(st.slider(key, 0.0, 1.0, float(ss.get(key, default)), 0.05))
        with output_tab:
            make_video = st.checkbox("Render annotated video", value=bool(ss.get("make_video", True)))
            reuse_cached_poses = st.checkbox(
                "Reuse existing poses.csv when available",
                value=bool(ss.get("reuse_cached_poses", True)),
                help="Pose tracking is skipped when the input, model, thresholds, FPS, and tracker settings match.",
            )

    if st.button("Run", type="primary", disabled=not valid_input):
        ss.input_path = str(input_path)
        ss.output_dir = str(output_dir)
        ss.device = device
        ss.pose_model = pose_model
        ss.person_det_thresh = person_det_thresh
        ss.keypoint_conf_thresh = keypoint_conf_thresh
        ss.use_source_fps = use_source_fps
        ss.person_target_fps = person_target_fps
        ss.orientation = orientation
        ss.smoothing_window = int(smoothing_window)
        ss.max_gap_seconds = float(max_gap_seconds)
        ss.min_segment_seconds = float(min_segment_seconds)
        ss.min_track_seconds = float(min_track_seconds)
        ss.hip_ratio = float(hip_ratio)
        ss.tracker_updates = tracker_updates
        for key, value in tracker_updates.items():
            ss[key] = value
        ss.make_video = make_video
        ss.reuse_cached_poses = reuse_cached_poses
        ss.state = "processing"
        st.rerun()


def render_processing(analyzer: HandLocationAnalyzer) -> None:
    ss = st.session_state
    progress_bar = st.progress(0, text="Preparing...")
    try:
        analyzer.preprocess(
            input_path=ss.input_path,
            output_dir=ss.output_dir,
            device=ss.device,
            pose_model=ss.pose_model,
            person_det_thresh=ss.person_det_thresh,
            keypoint_conf_thresh=ss.keypoint_conf_thresh,
            person_target_fps=ss.person_target_fps,
            tracker_updates=ss.tracker_updates,
            smoothing_window=ss.smoothing_window,
            max_gap_seconds=ss.max_gap_seconds,
            min_track_seconds=ss.min_track_seconds,
            min_segment_seconds=ss.min_segment_seconds,
            hip_ratio=ss.hip_ratio,
            orientation=ss.orientation,
            make_video=ss.make_video,
            reuse_cached_poses=ss.reuse_cached_poses,
        )
        ss.results = analyzer.run_all(progress_bar=progress_bar)
        ss.error_message = None
        ss.state = "end"
    except Exception as error:
        ss.error_message = f"{type(error).__name__}: {error}"
        ss.state = "error"
    st.rerun()


def location_summary_table(summary: dict) -> pd.DataFrame:
    rows = []
    for person in summary.get("persons", []):
        for hand in HANDS:
            row = {"track_id": person["track_id"], "hand": hand}
            by_location = person.get(f"{hand}_seconds_by_location", {})
            for location in range(1, 10):
                row[str(location)] = by_location.get(str(location), 0.0)
            rows.append(row)
    return pd.DataFrame(rows)


def render_end() -> None:
    ss = st.session_state
    results = ss.results or {}
    elapsed = results.get("elapsed_seconds")
    st.success("Completed" if elapsed is None else f"Completed in {format_elapsed(float(elapsed))}")
    st.write(f"Hand location CSV: `{results.get('segments_path')}`")
    st.write(f"Per-frame CSV: `{results.get('frames_path')}`")
    st.write(f"Pose CSV: `{results.get('poses_path')}`")
    video_path = results.get("video_path")
    if video_path is not None:
        st.write(f"Annotated video: `{video_path}`")

    summary = results.get("summary") or {}
    st.subheader("Seconds per sector")
    table = location_summary_table(summary)
    if table.empty:
        st.warning("No persons with usable keypoints were found.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")

    segments = results.get("segments")
    if segments is not None and not segments.empty:
        with st.expander(f"Segments ({len(segments)})", expanded=False):
            st.dataframe(segments, hide_index=True, width="stretch")
            st.download_button(
                "Download hand_locations.csv",
                segments.to_csv(index=False).encode("utf-8"),
                file_name="hand_locations.csv",
                mime="text/csv",
            )

    if video_path is not None and Path(video_path).exists():
        with st.expander("Annotated video", expanded=False):
            st.video(str(video_path))

    if st.button("Back to settings"):
        reset_to_start()
        st.rerun()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    render_header()
    init_state()
    analyzer: HandLocationAnalyzer = st.session_state.analyzer

    state = st.session_state.state
    if state == "start":
        render_start(analyzer)
    elif state == "processing":
        render_processing(analyzer)
    elif state == "error":
        st.error(st.session_state.error_message or "Unknown error")
        if st.button("Back"):
            reset_to_start()
            st.rerun()
    elif state == "end":
        render_end()


main()
