#!/usr/bin/env python3
"""Detect a capture render that repeats one slice of itself.

Why this exists: the VEGAS "Render As" defect produced a file that was the first
second of the timeline played N times, because every host chunk started its local
capture cursor at 0.  Nothing in the pipeline complained: the WAV is a perfectly
valid file of the right length, and the plugin logged "cheerful" block stats.  A
listening test catches it, an automated acceptance run does not.

The check is deliberately blunt.  It splits the take into overlapping one second
windows, decimates each to a few kHz of mono, normalises it, and reports every
pair of windows whose normalised cross-correlation exceeds 0.99 while they are
more than one window apart.  Genuine music does not correlate that closely at
unrelated positions; a looping bug correlates at ~1.0 by construction.

Usage:
    python scripts/audit/render-repeat-check.py <file.wav> [more.wav ...]

Exit code 0 = no repetition found, 1 = repetition found, 2 = usage/read error.
"""

from __future__ import annotations

import struct
import sys
import wave

WINDOW_SEC = 1.0
HOP_SEC = 0.5
MIN_GAP_WINDOWS = 2
CORRELATION_LIMIT = 0.99
TARGET_RATE = 4000


def read_mono_stream(path: str) -> tuple[list[float], int]:
    """Read a RIFF/WAVE file as mono floats, decimated to TARGET_RATE."""
    with wave.open(path, "rb") as source:
        channels = source.getnchannels()
        width = source.getsampwidth()
        rate = source.getframerate()
        frames = source.getnframes()
        raw = source.readframes(frames)
    if channels <= 0 or rate <= 0:
        raise ValueError("degenerate stream header")
    sample_bytes = width * channels
    if sample_bytes <= 0:
        raise ValueError("degenerate sample width")
    count = len(raw) // sample_bytes
    scale = {1: 128.0, 2: 32768.0, 3: 8388608.0, 4: 2147483648.0}.get(width)
    if scale is None:
        raise ValueError("unsupported sample width %d" % width)
    step = max(1, rate // TARGET_RATE)
    out: list[float] = []
    acc = 0.0
    taken = 0
    for index in range(0, count, step):
        base = index * sample_bytes
        if width == 3:
            value = int.from_bytes(raw[base : base + 3], "little", signed=True)
        elif width == 1:
            value = raw[base] - 128
        else:
            value = struct.unpack_from("<h" if width == 2 else "<i", raw, base)[0]
        # Down-mix by summing the channels of this frame.
        for channel in range(1, channels):
            offset = base + channel * width
            if width == 3:
                value += int.from_bytes(raw[offset : offset + 3], "little", signed=True)
            elif width == 1:
                value += raw[offset] - 128
            else:
                value += struct.unpack_from("<h" if width == 2 else "<i", raw, offset)[0]
        acc += value / (scale * channels)
        taken += 1
        if taken == step:
            out.append(acc / step)
            acc = 0.0
            taken = 0
    effective = rate / step
    return out, max(1, int(round(effective)))


def normalised(values: list[float]) -> list[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    centred = [v - mean for v in values]
    energy = sum(v * v for v in centred) ** 0.5
    if energy < 1e-9:
        return [0.0] * n
    return [v / energy for v in centred]


def correlation(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def audit(path: str) -> list[tuple[int, int, float]]:
    samples, rate = read_mono_stream(path)
    window = max(16, int(round(WINDOW_SEC * rate)))
    hop = max(1, int(round(HOP_SEC * rate)))
    if len(samples) < window * 3:
        return []
    windows: list[list[float]] = []
    for start in range(0, len(samples) - window + 1, hop):
        windows.append(normalised(samples[start : start + window]))
    hits: list[tuple[int, int, float]] = []
    for i in range(len(windows)):
        for j in range(i + MIN_GAP_WINDOWS, len(windows)):
            score = correlation(windows[i], windows[j])
            if abs(score) >= CORRELATION_LIMIT:
                hits.append((i, j, score))
    return hits


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0])
        print("usage: render-repeat-check.py <file.wav> [more.wav ...]")
        return 2
    failed = False
    for path in argv[1:]:
        try:
            hits = audit(path)
        except (OSError, ValueError, wave.Error) as error:
            print("error: %s: %s" % (path, error))
            return 2
        if not hits:
            print("ok: %s: no repeated window" % path)
            continue
        failed = True
        print("repeat: %s: %d correlated window pair(s)" % (path, len(hits)))
        for i, j, score in hits[:10]:
            print(
                "  window %d ~ window %d: corr=%.4f (%.2fs ~ %.2fs)"
                % (i, j, score, i * HOP_SEC, j * HOP_SEC)
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
