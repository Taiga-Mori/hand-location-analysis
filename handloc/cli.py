import argparse
import sys
import time
from pathlib import Path

from .constants import DEFAULT_MODE, DEFAULT_ORIENTATION, DEFAULT_POSE_MODEL, MODES, ORIENTATION_MODES, POSE_MODELS
from .progress import reset_progress_timers

TRACKER_OPTIONS = {
    "track_high_thresh": float,
    "track_low_thresh": float,
    "new_track_thresh": float,
    "track_buffer": int,
    "match_thresh": float,
}


def print_tracks(tracks, selected: int | None) -> None:
    print("\n" + " Tracked persons ".center(72, "="), flush=True)
    print("  track_id  selected  detected_frames  time (s)        center (x, y)", flush=True)
    for track in tracks.itertuples():
        mark = "yes" if track.track_id == selected else "-"
        print(
            f"  {track.track_id:>8}  {mark:>8}  {track.detected_frames:>15}  {track.first_time:6.2f}-{track.last_time:<6.2f}  "
            f"({track.center_x:.0f}, {track.center_y:.0f})",
            flush=True,
        )
    print("=" * 72, flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handloc", description="Hand location annotation for frontal conversation videos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Annotate hand locations in one or more videos.")
    run_parser.add_argument("input_paths", nargs="+", help="Input video file(s)")
    run_parser.add_argument("--output-name", default=None, help="Output folder name, created next to the input video (default: the video name). Only valid with one input.")
    run_parser.add_argument("--device", default=None, help="cuda:0, mps, or cpu (default: best available)")
    run_parser.add_argument("--pose-model", choices=POSE_MODELS, default=DEFAULT_POSE_MODEL)
    run_parser.add_argument("--person-thresh", type=float, default=0.5, help="Person detection confidence threshold")
    run_parser.add_argument(
        "--keypoint-thresh",
        type=float,
        default=0.0,
        help="After interpolation, exclude a hand in frames where its wrist or any shoulder/hip keypoint is below "
        "this confidence (default 0: use every frame)",
    )
    run_parser.add_argument(
        "--track-id",
        type=int,
        default=None,
        help="The one target person. Tracked persons are listed after YOLO runs; if there are several and no id is "
        "given, only poses.csv and summary.json are written, so rerun with --track-id (YOLO is then reused). "
        "Only valid with one input.",
    )
    run_parser.add_argument("--person-fps", type=float, default=None, help="Run pose tracking at this FPS (default: source FPS)")
    run_parser.add_argument(
        "--mode",
        choices=MODES,
        default=DEFAULT_MODE,
        help="frontal: camera facing the person (shoulder/hip grid). side: camera at the person's side, sitting "
        "upright (body/knee grid, sectors 1/4/7 in front of the person)",
    )
    run_parser.add_argument(
        "--orientation",
        choices=ORIENTATION_MODES,
        default=DEFAULT_ORIENTATION,
        help="frontal mode only. frontal: the person's right side is the image left. auto: decide it from the shoulder order",
    )
    run_parser.add_argument("--no-video", action="store_true", help="Skip rendering the annotated video")
    run_parser.add_argument("--no-cache", action="store_true", help="Always rerun pose tracking even if summary.json records the same YOLO settings")
    for name, value_type in TRACKER_OPTIONS.items():
        run_parser.add_argument(f"--{name.replace('_', '-')}", type=value_type, default=None, help=f"BoT-SORT {name}")

    gui_parser = subparsers.add_parser("gui", help="Launch the Streamlit GUI.")
    gui_parser.add_argument("--port", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "gui":
        from .launcher import launch_gui

        return launch_gui(args.port)

    if args.command == "run":
        if args.output_name and len(args.input_paths) > 1:
            parser.error("--output-name can only be used with a single input file.")
        if args.track_id is not None and len(args.input_paths) > 1:
            parser.error("--track-id can only be used with a single input file.")
        from .pipeline import HandLocationAnalyzer

        tracker_updates = {
            name: getattr(args, name) for name in TRACKER_OPTIONS if getattr(args, name) is not None
        }
        analyzer = HandLocationAnalyzer()
        exit_code = 0
        for input_path in args.input_paths:
            output_dir = Path(input_path).expanduser().parent / args.output_name if args.output_name else None
            analyzer.preprocess(
                input_path=input_path,
                output_dir=output_dir,
                device=args.device,
                pose_model=args.pose_model,
                person_det_thresh=args.person_thresh,
                keypoint_conf_thresh=args.keypoint_thresh,
                person_target_fps=args.person_fps,
                tracker_updates=tracker_updates,
                mode=args.mode,
                orientation=args.orientation,
                make_video=not args.no_video,
                reuse_cached_poses=not args.no_cache,
            )
            reset_progress_timers()
            start = time.monotonic()
            poses = analyzer.det_poses()
            try:
                track_id = analyzer.resolve_track_id(poses, args.track_id)
            except ValueError as error:
                print_tracks(analyzer.list_tracks(poses), None)
                print(f"handloc: {input_path}: {error} Rerun with --track-id.", file=sys.stderr)
                exit_code = 2
                continue
            print_tracks(analyzer.list_tracks(poses), track_id)
            results = analyzer.annotate(poses, track_id, tracking_seconds=time.monotonic() - start)
            print(results["locations_path"])
            if results["video_path"] is not None:
                print(results["video_path"])
        return exit_code

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
