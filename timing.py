"""Formatting for the "how long did this answer take" caption."""


def format_duration(seconds: float) -> str:
    """0.84 -> "0.8s", 12.3 -> "12s", 75 -> "1m 15s"."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    whole = round(seconds)
    if whole < 60:
        return f"{whole}s"
    minutes, secs = divmod(whole, 60)
    return f"{minutes}m {secs:02d}s"
