import time


_last_console_label: str | None = None
_last_console_percent: int | None = None
_run_start_time: float | None = None
_phase_start_time: float | None = None


def _phase_separator(label: str) -> str:
    title = f" {label} "
    width = max(72, len(title) + 8)
    side = max(2, (width - len(title)) // 2)
    line = "=" * side + title + "=" * side
    return line[:width]


class ConsoleProgress:
    """Minimal console progress reporter compatible with Streamlit's progress API."""

    def __init__(self) -> None:
        self._last_text = None
        self._last_percent = None

    def progress(self, value: float, text: str = "") -> None:
        percent = int(round(max(0.0, min(1.0, value)) * 100))
        if text == self._last_text and percent == self._last_percent:
            return
        if text != self._last_text:
            print("\n" + _phase_separator(text), flush=True)
        self._last_text = text
        self._last_percent = percent
        print(f"  - [{percent:3d}%] {text}", flush=True)


def format_elapsed(seconds: float) -> str:
    """Format elapsed seconds as compact wall-clock time."""

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def reset_progress_timers() -> None:
    """Start a new timed pipeline run."""

    global _last_console_label, _last_console_percent, _run_start_time, _phase_start_time

    _last_console_label = None
    _last_console_percent = None
    _run_start_time = time.monotonic()
    _phase_start_time = None


def update_progress(progress_bar, step: int, total: int, label: str) -> None:
    """Update UI progress and mirror phase progress to the terminal."""

    global _last_console_label, _last_console_percent, _run_start_time, _phase_start_time

    now = time.monotonic()
    if _run_start_time is None:
        _run_start_time = now
    safe_total = max(int(total), 1)
    safe_step = max(0, min(int(step), safe_total))
    ratio = min(safe_step / safe_total, 1.0)
    percent = int(round(ratio * 100))

    is_new_phase = label != _last_console_label
    if not is_new_phase and percent == _last_console_percent:
        return

    if is_new_phase:
        _phase_start_time = now
        print("\n" + _phase_separator(label), flush=True)

    elapsed = now - (_phase_start_time or now)
    elapsed_text = format_elapsed(elapsed)
    text = f"{label} {percent} % | elapsed {elapsed_text}"

    _last_console_label = label
    _last_console_percent = percent

    if progress_bar is not None and not isinstance(progress_bar, ConsoleProgress):
        progress_bar.progress(ratio, text=text)

    print(f"  - [{percent:3d}%] {label} ({safe_step}/{safe_total}) | elapsed {elapsed_text}", flush=True)


def finish_progress(progress_bar, label: str = "Pipeline completed") -> float:
    """Report total elapsed time for the current timed run."""

    global _run_start_time, _phase_start_time

    now = time.monotonic()
    if _run_start_time is None:
        _run_start_time = now
    total_elapsed = now - _run_start_time
    elapsed_text = format_elapsed(total_elapsed)
    text = f"{label} | total elapsed {elapsed_text}"
    print("\n" + _phase_separator(label), flush=True)
    print(f"  - {text}", flush=True)
    if progress_bar is not None and not isinstance(progress_bar, ConsoleProgress):
        progress_bar.progress(1.0, text=text)
    _phase_start_time = None
    return total_elapsed
