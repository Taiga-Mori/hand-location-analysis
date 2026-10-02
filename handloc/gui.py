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
from handloc.constants import DEFAULT_MODE, DEFAULT_ORIENTATION, DEFAULT_POSE_MODEL, HANDS, MODES, ORIENTATION_MODES, POSE_MODELS, VIDEO_EXTENSIONS
from handloc.progress import format_elapsed

APP_TITLE = "Hand Location Analysis"
APP_CAPTION = "Nine-sector hand location annotation for conversation videos (Choi et al., 2014, Fig. 3)"
TRACKER_KEYS = ["track_high_thresh", "track_low_thresh", "new_track_thresh", "track_buffer", "match_thresh"]
SECTOR_DIAGRAMS = {
    "frontal": """
| | person's right | center | person's left |
|---|:---:|:---:|:---:|
| above shoulders | 1 | 2 | 3 |
| shoulders–hips | 4 | **5** (torso) | 6 |
| below hips | 7 | 8 | 9 |
""",
    "side": """
| | in front of the knees | between body and knees | behind the body |
|---|:---:|:---:|:---:|
| above shoulders | 1 | 2 | 3 |
| shoulders–thighs | 4 | **5** | 6 |
| below thighs | 7 | 8 | 9 |
""",
}
SECTOR_TEXT = {
    "frontal": "The central rectangle spans both **shoulder** keypoints horizontally and runs from the shoulder line "
    "down to the **hip** line. Sectors are numbered from the person's own viewpoint.",
    "side": "Assumes the person sits upright. The vertical lines are the **body** (mean of shoulder and hip x) and the "
    "**knees**; the horizontal lines are the **shoulders** and the **thighs** (mean of hip and knee y). Sectors 1/4/7 "
    "are always in front of the person.",
}


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

    mode = st.radio(
        "Mode",
        MODES,
        index=MODES.index(ss.get("mode", DEFAULT_MODE)),
        format_func=lambda value: {"frontal": "Frontal (camera facing the person)", "side": "Side (camera at the person's side)"}[value],
        horizontal=True,
    )

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
        st.markdown(SECTOR_TEXT[mode] + " Each hand's location is the sector containing its **wrist** keypoint.")
        st.markdown(SECTOR_DIAGRAMS[mode])
        st.caption(
            "The person's keypoints are first interpolated over the whole video. With a keypoint threshold above 0, "
            "a hand is then left out in frames where its wrist or any keypoint used for the grid is below the threshold."
        )

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
                float(ss.get("keypoint_conf_thresh", 0.0)),
                0.05,
                help="0 uses every frame. Otherwise a hand is left out in frames where its wrist or any shoulder/hip "
                "keypoint is below this confidence (after interpolation). Changing it does not rerun YOLO.",
            )
            use_source_fps = st.checkbox("Track every frame (source FPS)", value=bool(ss.get("use_source_fps", True)))
            person_target_fps = None
            if not use_source_fps:
                person_target_fps = st.number_input(
                    "Tracking FPS", min_value=1.0, max_value=240.0, value=float(ss.get("person_target_fps") or 15.0), step=1.0
                )
        with location_tab:
            orientation = st.selectbox(
                "Orientation (frontal mode)",
                ORIENTATION_MODES,
                index=ORIENTATION_MODES.index(ss.get("orientation", DEFAULT_ORIENTATION)),
                help="frontal: the person's right side is the image left. auto: decide it from the shoulder order each frame.",
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
                help="YOLO is skipped when summary.json records the same input, model, person threshold, FPS, and tracker settings.",
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
        ss.mode = mode
        ss.orientation = orientation
        ss.tracker_updates = tracker_updates
        for key, value in tracker_updates.items():
            ss[key] = value
        ss.make_video = make_video
        ss.reuse_cached_poses = reuse_cached_poses
        ss.state = "tracking"
        st.rerun()


def render_tracking(analyzer: HandLocationAnalyzer) -> None:
    """Run YOLO (or reuse poses.csv), then list the tracked persons for selection."""

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
            mode=ss.mode,
            orientation=ss.orientation,
            make_video=ss.make_video,
            reuse_cached_poses=ss.reuse_cached_poses,
        )
        poses = analyzer.det_poses(progress_bar=progress_bar)
        progress_bar.progress(1.0, text="Collecting tracked persons...")
        ss.poses = poses
        ss.tracks = analyzer.list_tracks(poses)
        ss.thumbnails = analyzer.track_thumbnails(poses, ss.tracks)
        ss.error_message = None
        ss.state = "select"
    except Exception as error:
        ss.error_message = f"{type(error).__name__}: {error}"
        ss.state = "error"
    st.rerun()


def render_select() -> None:
    """Show every unique tracked person and let the user pick the one target person."""

    ss = st.session_state
    tracks = ss.tracks
    st.subheader("Select the target person")
    if tracks.empty:
        st.warning("No persons were tracked in this video.")
        if st.button("Back to settings"):
            reset_to_start()
            st.rerun()
        return
    st.caption(
        "Each image is the frame where YOLO was most confident about that person. One person is annotated per run; "
        "to annotate another person, run again (YOLO is reused when its settings are unchanged)."
    )
    columns_per_row = 4
    for start in range(0, len(tracks), columns_per_row):
        row = tracks.iloc[start : start + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, track in zip(columns, row.itertuples()):
            with column:
                thumbnail = ss.thumbnails.get(int(track.track_id))
                if thumbnail is not None:
                    st.image(thumbnail, width="stretch")
                st.markdown(f"**ID {track.track_id}**")
                st.caption(f"{track.first_time:.1f}-{track.last_time:.1f} s, {track.detected_frames} frames")
    track_ids = [int(t) for t in tracks["track_id"]]
    default = int(tracks.loc[tracks["detected_frames"].idxmax(), "track_id"])
    selected = st.radio(
        "Target person",
        track_ids,
        index=track_ids.index(default),
        format_func=lambda track_id: f"ID {track_id}",
        horizontal=True,
        key="target_track_id",
    )
    col_run, col_back = st.columns([1, 1])
    with col_run:
        if st.button("Annotate this person", type="primary", width="stretch"):
            ss.selected_track_id = int(selected)
            ss.state = "annotating"
            st.rerun()
    with col_back:
        if st.button("Back to settings", width="stretch"):
            reset_to_start()
            st.rerun()


def render_annotating(analyzer: HandLocationAnalyzer) -> None:
    ss = st.session_state
    progress_bar = st.progress(0, text="Preparing...")
    try:
        ss.results = analyzer.annotate(ss.poses, ss.selected_track_id, progress_bar=progress_bar)
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
    st.caption(f"Annotated person: ID {results.get('track_id')}")
    st.write(f"Locations: `{results.get('locations_path')}`")
    st.write(f"Normalized wrist coordinates: `{results.get('frames_path')}`")
    st.write(f"Raw YOLO poses: `{results.get('poses_path')}`")
    st.write(f"Summary: `{results.get('summary_path')}`")
    video_path = results.get("video_path")
    if video_path is not None:
        st.write(f"Annotated video: `{video_path}`")

    summary = results.get("summary") or {}
    st.subheader("Seconds per sector")
    table = location_summary_table(summary)
    if table.empty:
        st.warning("No persons were tracked.")
    else:
        st.dataframe(table, hide_index=True, width="stretch")

    segments = results.get("segments")
    if segments is not None and not segments.empty:
        with st.expander(f"Segments ({len(segments)})", expanded=False):
            st.dataframe(segments, hide_index=True, width="stretch")
            st.download_button(
                "Download locations.csv",
                segments.to_csv(index=False).encode("utf-8"),
                file_name="locations.csv",
                mime="text/csv",
            )

    heatmap_paths = results.get("heatmap_paths") or {}
    if heatmap_paths:
        st.subheader("Wrist heatmaps")
        for name, path in heatmap_paths.items():
            st.caption(f"{name}: `{path}`")
            st.image(str(path), width="stretch")

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
    elif state == "tracking":
        render_tracking(analyzer)
    elif state == "select":
        render_select()
    elif state == "annotating":
        render_annotating(analyzer)
    elif state == "error":
        st.error(st.session_state.error_message or "Unknown error")
        if st.button("Back"):
            reset_to_start()
            st.rerun()
    elif state == "end":
        render_end()


main()
