import argparse

from handloc.launcher import launch_gui


def main() -> int:
    parser = argparse.ArgumentParser(prog="handloc")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="GUI port (default: first available port starting at 8501)",
    )
    args = parser.parse_args()
    return launch_gui(args.port)


if __name__ == "__main__":
    raise SystemExit(main())
