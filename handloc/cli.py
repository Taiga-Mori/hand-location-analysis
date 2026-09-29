import argparse

from .constants import DEFAULT_ORIENTATION, DEFAULT_POSE_MODEL, ORIENTATION_MODES, POSE_MODELS

TRACKER_OPTIONS = {
    "track_high_thresh": float,
    "track_low_thresh": float,
    "new_track_thresh": float,
    "track_buffer": int,
    "match_thresh": float,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="handloc", description="Hand location annotation for frontal conversation videos.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Annotate hand locations in one or more videos.")
    run_parser.add_argument("input_paths", nargs="+", help="Input video file(s)")
    run_parser.add_argument("--output-dir", default=None, help="Output directory (default: <input dir>/<input stem>). Only valid with one input.")
    run_parser.add_argument("--device", default=None, help="cuda:0, mps, or cpu (default: best available)")
    run_parser.add_argument("--pose-model", choices=POSE_MODELS, default=DEFAULT_POSE_MODEL)
    run_parser.add_argument("--person-thresh", type=float, default=0.5, help="Person detection confidence threshold")
    run_parser.add_argument("--keypoint-thresh", type=float, default=0.5, help="Keypoint confidence threshold")
    run_parser.add_argument("--person-fps", type=float, default=None, help="Run pose tracking at this FPS (default: source FPS)")
    run_parser.add_argument("--smoothing-window", type=int, default=5, help="Keypoint moving-average window in frames")
    run_parser.add_argument("--max-gap", type=float, default=0.5, help="Interpolate keypoint gaps up to this many seconds")
    run_parser.add_argument("--min-track", type=float, default=1.0, help="Drop tracks detected for less than this many seconds")
    run_parser.add_argument("--min-segment", type=float, default=0.2, help="Merge location segments shorter than this many seconds")
    run_parser.add_argument("--hip-ratio", type=float, default=1.3, help="Fallback (hip - shoulder height) / shoulder width when hips are never visible")
    run_parser.add_argument("--orientation", choices=ORIENTATION_MODES, default=DEFAULT_ORIENTATION)
    run_parser.add_argument("--no-video", action="store_true", help="Skip rendering the annotated video")
    run_parser.add_argument("--no-cache", action="store_true", help="Always rerun pose tracking even if poses.csv matches")
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
        if args.output_dir and len(args.input_paths) > 1:
            parser.error("--output-dir can only be used with a single input file.")
        from .pipeline import HandLocationAnalyzer

        tracker_updates = {
            name: getattr(args, name) for name in TRACKER_OPTIONS if getattr(args, name) is not None
        }
        analyzer = HandLocationAnalyzer()
        for input_path in args.input_paths:
            analyzer.preprocess(
                input_path=input_path,
                output_dir=args.output_dir,
                device=args.device,
                pose_model=args.pose_model,
                person_det_thresh=args.person_thresh,
                keypoint_conf_thresh=args.keypoint_thresh,
                person_target_fps=args.person_fps,
                tracker_updates=tracker_updates,
                smoothing_window=args.smoothing_window,
                max_gap_seconds=args.max_gap,
                min_track_seconds=args.min_track,
                min_segment_seconds=args.min_segment,
                hip_ratio=args.hip_ratio,
                orientation=args.orientation,
                make_video=not args.no_video,
                reuse_cached_poses=not args.no_cache,
            )
            results = analyzer.run_all()
            print(results["segments_path"])
            if results["video_path"] is not None:
                print(results["video_path"])
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
