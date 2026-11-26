#!/usr/bin/env python3
"""Detect a capture render that repeats one slice of itself.

Why this exists: the VEGAS "Render As" defect produced a file that was the first
second of the timeline played N times, because every host chunk started its local
capture cursor at 0.  Nothing in the pipeline complained: the WAV is a perfectly
valid file of the right length, and the plugin logged "cheerful" block stats.  A
listening test catches it, an automated acceptance run does not.

# How it decides

Two stages, because a single one is provably wrong in both directions:

1. **Estimate a period** from the 10 ms RMS envelope: autocorrelate the envelope
   (FFT, so the cost stays O(n log n) for long renders) and take the strongest
   peaks inside [0.1 s, duration / 2] as candidates.
2. **Verify each candidate on the waveform**: normalised cross-correlation of the
   *full-rate* samples at "lag" frames, taken at eight probes spread over the
   whole file.  A candidate counts only when the **median** probe reaches
   CORRELATION_LIMIT and the file is at least MIN_REPEATS periods long.

# Why not "compare 1 s windows 0.5 s apart" (the previous criterion)

That criterion was measured against the real defect and it *passed* the bad file:
the repeat period is 49 152 frames (1.024 000 s), so a 0.5 s hop grid never lands
on a 2.048 s interval, and a 1.000 s window straddles a seam.  Its best observed
|correlation| on the defective render was 0.770 -- it would have reported "ok".
The lesson is not "pick a better threshold": the criterion has to be derived from
the shape of the failure, and a window grid is exactly the shape a non-integer
period escapes.  Estimating the period first, then verifying it on the waveform,
has no such grid.

# Why the envelope is not enough on its own

Envelope autocorrelation is insensitive to the fine structure: on the normal
31.8 s vocal take it still returns a peak at 5.12 s.  Only the per-sample
verification separates the two -- every candidate on that take stays at or below
0.12, while the defective render's fundamental scores 0.99.  A gate must be able
to fail on a known-bad sample *and* pass on a known-good one; --self-test pins
both directions.

# The other question: --align

The VEGAS defect did not repeat anything -- the render was the right audio at the
wrong place, shifted by +1.85 s in one run and +18.6 s in another.  A repeat check
says nothing about that, so the tool answers the second question too: where does
the render's audio sit inside its source?

`--align` searches a 10 ms grid exactly, refines on a 1 ms grid, then on full-rate
samples, and prints the offset together with the correlation that backs it.  The
offset is `render[i] == source[i + lag]`, so a positive number means the render
started that far into the take.  Anything above 1 ms is a defect; the tolerance is
printed with the verdict so a reader never has to guess it.

Two more readings answer the questions the offset alone cannot:

* `--expect-offset SEC` replaces "must be zero" with "must be this".  A render of the
  range [28 s, 61 s] is *aligned* at lag 0 when the mapping is right, and still
  *aligned* at some other lag when the mapping shifts the whole product.  Only the
  expectation separates those two.
* `passthrough` is the length of the leading region that is a sample-for-sample copy
  of the source.  A processed render answers 0; a render whose head was passed
  through (the host's original audio written into the product) answers that head.

Usage:
    python scripts/audit/render-repeat-check.py <file.wav> [more.wav ...]
    python scripts/audit/render-repeat-check.py --align <render.wav> <source.wav>
    python scripts/audit/render-repeat-check.py --align <render.wav> <source.wav> --expect-offset 0
    python scripts/audit/render-repeat-check.py --self-test

Exit code 0 = no repetition found / the render is aligned, 1 = repetition found /
the render is misaligned, 2 = usage or read error.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile

# Windows 的 stdout 跟随 ANSI 代码页（中文系统是 GBK）。本文件含中文注释与中文输出，
# 一条中文在 GBK 控制台上会直接 UnicodeEncodeError —— 判据还没跑完，进程就死了，
# 而日志里留下的是一个编码错误，不是结论。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WAVE_FORMAT_PCM = 0x0001
WAVE_FORMAT_IEEE_FLOAT = 0x0003
WAVE_FORMAT_EXTENSIBLE = 0xFFFE

# Shortest period considered a repeat.  Anything faster is a tremolo, not a loop.
MIN_PERIOD_SEC = 0.1
# Envelope frame.  10 ms resolves a 1 s period to ~1 %, which the waveform pass refines.
ENVELOPE_HOP_SEC = 0.010
# Probes per candidate: spread over the whole file, so a loop in one corner cannot
# hide behind silence elsewhere.
PROBE_COUNT = 8
# Frames per verification probe (0.34 s at 48 kHz).
PROBE_FRAMES = 16384
# Probes used while searching for the exact frame lag.  Their median, not their
# mean: the defective render's first period is unique, so a mean that includes it
# would drag the true lag below a wrong one.
SEARCH_PROBE_COUNT = 4
# Window for the full-rate refinement (0.17 s at 48 kHz): long enough that a quiet
# passage cannot decide the alignment, short enough to stay sample accurate.
SEARCH_PROBE_FRAMES = 8192
# Decimation target for the lag search.  The search only has to find the peak
# neighbourhood; the verdict is computed on full-rate samples.
SEARCH_RATE = 6000
# How far around an envelope candidate the search looks.
SEARCH_SPAN_SEC = 0.008
# Window for the decimated pass (0.34 s at 6 kHz).
DECIMATED_PROBE_FRAMES = 2048
# Median probe correlation that counts as "the same audio again".
CORRELATION_LIMIT = 0.90
# A file that is one and a half periods long is not a loop.
MIN_REPEATS = 2
# Window for the passthrough probe.  0.1 s is long enough that a quiet passage
# cannot decide it and short enough that a seam is not averaged away.
PASSTHROUGH_WINDOW_SEC = 0.1
# A window at or above this correlation is a *verbatim copy* of the source, not a
# processed render.  The field symptom it names is "the first N seconds of the
# product are the host's original audio".
PASSTHROUGH_LIMIT = 0.995


class WavReadError(ValueError):
    """The file is not a wave file this tool can decode."""


class Repeat:
    """One confirmed repeat: a period, how many times it fits, and the evidence."""

    def __init__(self, period: int, repeats: int, median: float, probes: list[float]) -> None:
        self.period = period
        self.repeats = repeats
        self.median = median
        self.probes = probes

    def describe(self, rate: int) -> str:
        return "period=%d frames (%.6f s), repeats=%d, median ncc=%.3f" % (
            self.period,
            self.period / float(rate),
            self.repeats,
            self.median,
        )


# -- RIFF/WAVE decoding ----------------------------------------------------


def _parse_fmt(chunk: bytes) -> tuple[int, int, int, int, int]:
    """Return (format tag, channels, rate, block align, bits) from a fmt body."""
    if len(chunk) < 16:
        raise WavReadError("fmt chunk is %d bytes, need at least 16" % len(chunk))
    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack_from(
        "<HHIIHH", chunk, 0
    )
    if tag == WAVE_FORMAT_EXTENSIBLE:
        # WAVE_FORMAT_EXTENSIBLE keeps the real tag in the first two bytes of the
        # SubFormat GUID; the rest of the header is the plain 16 bytes.
        if len(chunk) < 40:
            raise WavReadError("extensible fmt chunk is %d bytes, need 40" % len(chunk))
        (tag,) = struct.unpack_from("<H", chunk, 24)
    if channels <= 0:
        raise WavReadError("degenerate channel count")
    if rate <= 0:
        raise WavReadError("degenerate sample rate")
    if bits <= 0:
        raise WavReadError("degenerate sample width")
    return tag, channels, rate, block_align, bits


def _decode(data: bytes, tag: int, channels: int, bits: int) -> list[float]:
    """Decode interleaved frames to mono floats in [-1, 1]."""
    frame_bytes = (bits + 7) // 8 * channels
    frames = len(data) // frame_bytes
    if frames <= 0:
        raise WavReadError("no complete audio frame in the data chunk")
    data = data[: frames * frame_bytes]
    total = frames * channels
    if tag == WAVE_FORMAT_IEEE_FLOAT and bits in (32, 64):
        code = "f" if bits == 32 else "d"
        values = struct.unpack_from("<%d%s" % (total, code), data, 0)
        shift, scale = 0.0, 1.0
    elif tag == WAVE_FORMAT_PCM and bits == 8:
        values = struct.unpack_from("<%dB" % total, data, 0)
        shift, scale = 128.0, 1.0 / 128.0
    elif tag == WAVE_FORMAT_PCM and bits == 16:
        values = struct.unpack_from("<%dh" % total, data, 0)
        shift, scale = 0.0, 1.0 / 32768.0
    elif tag == WAVE_FORMAT_PCM and bits == 24:
        # No 3-byte struct code exists; widen by hand.  Only this path pays per sample.
        values = tuple(
            int.from_bytes(data[index : index + 3], "little", signed=True)
            for index in range(0, len(data), 3)
        )
        shift, scale = 0.0, 1.0 / 8388608.0
    elif tag == WAVE_FORMAT_PCM and bits == 32:
        values = struct.unpack_from("<%di" % total, data, 0)
        shift, scale = 0.0, 1.0 / 2147483648.0
    else:
        raise WavReadError("unsupported sample format: tag=0x%04x bits=%d" % (tag, bits))
    if channels == 1:
        return [(value - shift) * scale for value in values]
    if channels == 2:
        left = values[0::2]
        right = values[1::2]
        gain = scale / 2.0
        return [(a + b - 2.0 * shift) * gain for a, b in zip(left, right)]
    gain = scale / channels
    return [
        sum(values[frame * channels + channel] - shift for channel in range(channels))
        * gain
        for frame in range(frames)
    ]


def read_wave(path: str) -> tuple[list[float], int]:
    """Read a RIFF/WAVE file as mono floats.

    Deliberately not the wave module: it only understands PCM, and the render that
    motivated this tool is 32-bit IEEE float (fmt 0x0003).
    """
    with open(path, "rb") as handle:
        blob = handle.read()
    if len(blob) < 12 or blob[0:4] != b"RIFF" or blob[8:12] != b"WAVE":
        raise WavReadError("not a RIFF/WAVE file")
    offset = 12
    fmt = None
    data = None
    while offset + 8 <= len(blob):
        chunk_id = blob[offset : offset + 4]
        (size,) = struct.unpack_from("<I", blob, offset + 4)
        body = offset + 8
        end = min(body + size, len(blob))
        if chunk_id == b"fmt ":
            fmt = blob[body:end]
        elif chunk_id == b"data":
            data = blob[body:end]
        # Chunks are word aligned.  A streamed writer may declare 0xFFFFFFFF, which
        # simply ends the scan -- the data chunk has already been captured by then.
        offset = body + size + (size & 1)
    if fmt is None:
        raise WavReadError("no fmt chunk")
    if data is None:
        raise WavReadError("no data chunk")
    tag, channels, rate, _block_align, bits = _parse_fmt(fmt)
    return _decode(data, tag, channels, bits), rate


# -- Signal analysis -------------------------------------------------------


def _fft(values: list[complex], inverse: bool) -> None:
    """In-place radix-2 Cooley-Tukey.  Iterative so the call depth stays constant."""
    count = len(values)
    j = 0
    for i in range(1, count):
        bit = count >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            values[i], values[j] = values[j], values[i]
    sign = 1.0 if inverse else -1.0
    length = 2
    while length <= count:
        angle = sign * 2.0 * math.pi / length
        step = complex(math.cos(angle), math.sin(angle))
        half = length >> 1
        for start in range(0, count, length):
            twiddle = 1.0 + 0.0j
            for index in range(start, start + half):
                even = values[index]
                odd = values[index + half] * twiddle
                values[index] = even + odd
                values[index + half] = even - odd
                twiddle *= step
        length <<= 1
    if inverse:
        for index in range(count):
            values[index] /= count


def envelope(samples: list[float], rate: int) -> list[float]:
    """10 ms RMS envelope.  Drops the carrier, keeps the rhythm of the loop."""
    hop = max(1, int(round(ENVELOPE_HOP_SEC * rate)))
    frames = len(samples) // hop
    out = []
    for frame in range(frames):
        base = frame * hop
        total = 0.0
        for index in range(base, base + hop):
            value = samples[index]
            total += value * value
        out.append(math.sqrt(total / hop))
    return out


def _autocorrelation(values: list[float]) -> list[float]:
    size = 1
    while size < 2 * len(values):
        size <<= 1
    data = [complex(value, 0.0) for value in values]
    data.extend([0.0j] * (size - len(values)))
    _fft(data, False)
    for index in range(size):
        magnitude = abs(data[index])
        data[index] = complex(magnitude * magnitude, 0.0)
    _fft(data, True)
    return [value.real for value in data]


def _normalised(values: list[float]) -> list[float]:
    """Unbiased normalisation: each lag is divided by its own overlap before scaling."""
    count = len(values)
    if count == 0:
        return []
    variance = values[0] / count
    if variance <= 0.0:
        return [0.0] * count
    return [0.0] + [(values[lag] / (count - lag)) / variance for lag in range(1, count)]


def _comb_score(acf: list[float], lag: int, high: int) -> float:
    """Mean of the ACF at 1x, 2x, 3x ... of a candidate lag.

    A loop that repeats at L also repeats at 2L, 3L ...; its *fundamental* is the
    smallest lag whose whole comb lines up.  Scoring the comb instead of the single
    peak is what keeps a tall harmonic from being reported as the period -- the
    defective render's envelope ACF peaks hardest at 5x, not at 1x.
    """
    total = 0.0
    weight = 0.0
    scale = 1.0
    multiple = lag
    while multiple <= high and scale > 0.05:
        total += acf[multiple] * scale
        weight += scale
        scale *= 0.7
        multiple += lag
    return total / weight if weight > 0.0 else 0.0


def candidate_periods(samples: list[float], rate: int) -> list[float]:
    """Envelope-derived period candidates in seconds, shortest first."""
    shape = envelope(samples, rate)
    if len(shape) < 8:
        return []
    mean = sum(shape) / len(shape)
    centred = [value - mean for value in shape]
    acf = _normalised(_autocorrelation(centred))
    duration = len(samples) / float(rate)
    hop_seconds = ENVELOPE_HOP_SEC
    low = max(2, int(round(MIN_PERIOD_SEC / hop_seconds)))
    high = min(len(acf) - 2, int(round(duration / 2.0 / hop_seconds)))
    if high <= low:
        return []
    peaks = []
    combed = []
    for lag in range(low, high + 1):
        if acf[lag] < acf[lag - 1] or acf[lag] < acf[lag + 1]:
            continue
        # Parabolic interpolation: the hop grid is 10 ms, the waveform pass needs ~2 ms.
        left, centre, right = acf[lag - 1], acf[lag], acf[lag + 1]
        denominator = left - 2.0 * centre + right
        shift = 0.5 * (left - right) / denominator if denominator != 0.0 else 0.0
        peaks.append((centre, (lag + shift) * hop_seconds))
        combed.append((_comb_score(acf, lag, high), (lag + shift) * hop_seconds))
    peaks.sort(reverse=True)
    combed.sort(reverse=True)
    chosen = {seconds for _score, seconds in peaks[:6]}
    chosen.update(seconds for _score, seconds in combed[:4])
    return sorted(chosen)


def _prefix_sums(samples: list[float]) -> tuple[list[float], list[float]]:
    total = [0.0] * (len(samples) + 1)
    square = [0.0] * (len(samples) + 1)
    running = 0.0
    running_square = 0.0
    for index, value in enumerate(samples):
        running += value
        running_square += value * value
        total[index + 1] = running
        square[index + 1] = running_square
    return total, square


def _ncc(
    samples: list[float],
    prefix: list[float],
    prefix_square: list[float],
    left: int,
    right: int,
    length: int,
) -> float:
    """Normalised cross-correlation of two windows, means removed."""
    sum_a = prefix[left + length] - prefix[left]
    sum_b = prefix[right + length] - prefix[right]
    square_a = prefix_square[left + length] - prefix_square[left]
    square_b = prefix_square[right + length] - prefix_square[right]
    mean_a = sum_a / length
    mean_b = sum_b / length
    energy_a = square_a - sum_a * sum_a / length
    energy_b = square_b - sum_b * sum_b / length
    if energy_a <= 1e-12 or energy_b <= 1e-12:
        return 0.0
    dot = 0.0
    for index in range(length):
        dot += samples[left + index] * samples[right + index]
    covariance = dot - length * mean_a * mean_b
    return covariance / math.sqrt(energy_a * energy_b)


def _ncc_pair(
    a: list[float],
    prefix_a: list[float],
    square_a: list[float],
    b: list[float],
    prefix_b: list[float],
    square_b: list[float],
    left: int,
    right: int,
    length: int,
) -> float:
    """Normalised cross-correlation between a window of `a` and one of `b`.

    Why a second helper instead of `_ncc`: that one takes a single signal and two
    positions (the repeat check).  Using it for alignment would compare the render
    with *itself* at two positions -- every lag would look like a perfect match,
    and the tool would confidently report an offset of zero.
    """
    sum_a = prefix_a[left + length] - prefix_a[left]
    sum_b = prefix_b[right + length] - prefix_b[right]
    square_aa = square_a[left + length] - square_a[left]
    square_bb = square_b[right + length] - square_b[right]
    mean_a = sum_a / length
    mean_b = sum_b / length
    energy_a = square_aa - sum_a * sum_a / length
    energy_b = square_bb - sum_b * sum_b / length
    if energy_a <= 1e-12 or energy_b <= 1e-12:
        return 0.0
    dot = 0.0
    for index in range(length):
        dot += a[left + index] * b[right + index]
    covariance = dot - length * mean_a * mean_b
    return covariance / math.sqrt(energy_a * energy_b)

def _median(values: list[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _even_starts(max_start: int, count: int) -> list[int]:
    """Probes spread evenly over the file: the criterion the report states."""
    if count <= 1 or max_start <= 0:
        return [0]
    return [int(round(index * max_start / (count - 1))) for index in range(count)]


def _loud_starts(
    prefix_square: list[float], max_start: int, length: int, count: int
) -> list[int]:
    """Probes in the loudest windows.

    Correlation on a near-silent window is hiss: inside the defective render's quiet
    passages two copies of the same audio still score only 0.5, and a single quiet
    probe is enough to hide the true lag from the search.  The verdict still uses
    evenly spread probes -- this only picks where to *look*.
    """
    if max_start <= 0:
        return [0]
    slots = 64
    scored = []
    for index in range(slots):
        start = int(round(index * max_start / (slots - 1)))
        scored.append((prefix_square[start + length] - prefix_square[start], start))
    scored.sort(reverse=True)
    return [start for _energy, start in scored[:count]]


def _search_lag(low: int, high: int, score_at) -> tuple[int, float]:
    best_lag = low
    best_score = -2.0
    for lag in range(max(1, low), high + 1):
        score = score_at(lag)
        if score > best_score:
            best_score = score
            best_lag = lag
    return best_lag, best_score


def align_period(
    samples: list[float],
    prefix: list[float],
    prefix_square: list[float],
    seconds: float,
    rate: int,
    decimated: list[float],
    decimated_prefix: list[float],
    decimated_square: list[float],
    factor: int,
) -> tuple[int, float]:
    """Turn a ~10 ms period estimate into an exact frame lag.

    Coarse on the decimated copy over the whole span, then one full-rate refinement
    of +- two decimation steps, so the reported lag is sample accurate.
    """
    span = max(1, int(round(SEARCH_SPAN_SEC * rate / factor)))
    decimated_window = min(DECIMATED_PROBE_FRAMES, len(decimated) // 8)
    centre = max(1, int(round(seconds * rate / factor)))
    decimated_starts = _loud_starts(
        decimated_square, len(decimated) - centre - decimated_window, decimated_window,
        SEARCH_PROBE_COUNT,
    )

    def scan_decimal(lag: int) -> float:
        scores = []
        for start in decimated_starts:
            if start + lag + decimated_window > len(decimated):
                return -2.0
            scores.append(
                _ncc(
                    decimated,
                    decimated_prefix,
                    decimated_square,
                    start,
                    start + lag,
                    decimated_window,
                )
            )
        return _median(scores)

    coarse_lag, _coarse = _search_lag(centre - span, centre + span, scan_decimal)
    centre_full = coarse_lag * factor
    window = min(SEARCH_PROBE_FRAMES, len(samples) - centre_full - 1)
    starts = _loud_starts(
        prefix_square, len(samples) - centre_full - window, window, SEARCH_PROBE_COUNT
    )

    def scan_full(lag: int) -> float:
        scores = []
        for start in starts:
            if start + lag + window > len(samples):
                return -2.0
            scores.append(_ncc(samples, prefix, prefix_square, start, start + lag, window))
        return _median(scores)

    return _search_lag(centre_full - 2 * factor, centre_full + 2 * factor, scan_full)


def verify_period(
    samples: list[float], prefix: list[float], prefix_square: list[float], lag: int
) -> tuple[float, list[float]]:
    """Median correlation over evenly spread full-length probes at that lag."""
    room = len(samples) - lag
    length = min(PROBE_FRAMES, room)
    if length < 1024:
        return 0.0, []
    starts = _even_starts(len(samples) - lag - length, PROBE_COUNT)
    probes = [
        _ncc(samples, prefix, prefix_square, start, start + lag, length)
        for start in starts
    ]
    return _median(probes), probes


def decimate(samples: list[float], factor: int) -> list[float]:
    """Boxcar decimation.  The averaging is what keeps the peak at the true lag."""
    if factor <= 1:
        return samples
    count = len(samples) // factor
    out = [0.0] * count
    for index in range(count):
        base = index * factor
        total = 0.0
        for offset in range(factor):
            total += samples[base + offset]
        out[index] = total / factor
    return out


def audit(path: str) -> tuple[Repeat | None, float, float, int]:
    """Return (repeat, best candidate seconds, its median correlation, rate)."""
    samples, rate = read_wave(path)
    if len(samples) < int(MIN_PERIOD_SEC * 2 * rate):
        return None, 0.0, 0.0, rate
    prefix, prefix_square = _prefix_sums(samples)
    factor = max(1, int(round(rate / SEARCH_RATE)))
    decimated = decimate(samples, factor)
    decimated_prefix, decimated_square = _prefix_sums(decimated)
    best_seconds = 0.0
    best_median = -2.0
    for seconds in candidate_periods(samples, rate):
        duration = len(samples) / float(rate)
        if seconds <= MIN_PERIOD_SEC or seconds >= duration:
            continue
        lag, _score = align_period(
            samples,
            prefix,
            prefix_square,
            seconds,
            rate,
            decimated,
            decimated_prefix,
            decimated_square,
            factor,
        )
        # No score gate here: the verification is cheap and its number is what the
        # report quotes for a rejected candidate, which is the only evidence a
        # reader has that "ok" was decided rather than assumed.
        median, probes = verify_period(samples, prefix, prefix_square, lag)
        if median > best_median:
            best_median = median
            best_seconds = lag / float(rate)
        if median >= CORRELATION_LIMIT:
            repeats = len(samples) // lag
            if repeats >= MIN_REPEATS:
                return Repeat(lag, repeats, median, probes), best_seconds, best_median, rate
    return None, best_seconds, max(best_median, 0.0), rate


# -- Render <-> source alignment -------------------------------------------

# Two grids, because one cannot serve both ends: a 10 ms grid keeps the search
# cheap (a 30 s head is 3 000 samples), and a 1 ms grid is the acceptance
# tolerance.  The coarse peak is never the answer -- it only says where to look.
ALIGN_COARSE_RATE = 100
ALIGN_FINE_RATE = 1000
# Coarse peaks carried into the fine pass.  More than one, because periodic
# material produces several near-equal lags and the fine pass is what separates
# them.
ALIGN_CANDIDATES = 6
# Window for the fine pass: half a second at 1 kHz.  Long enough that a quiet
# passage cannot decide the lag, short enough to stay cheap.
ALIGN_PROBE_FRAMES = 512
# Only the head of both files is aligned: the defect is a *start* offset, and
# scanning a ten minute render costs seconds without changing the answer.
ALIGN_MAX_SEC = 30.0
# Default search span.  The two field offsets were +1.85 s and +18.6 s; 30 s
# covers both with room, and --max-offset widens it.
DEFAULT_MAX_OFFSET_SEC = 30.0
# Below this an offset is rounding, not a defect.
ALIGN_TOLERANCE_SEC = 0.001


class AlignResult:
    """Where the render's audio sits inside the source.

    Convention, and the only one this tool uses:

        render[i] == source[i + lag]

    So `lag > 0` means the render's first sample is source sample `lag`: the
    render started that far *into* the take (head trimmed, audio played late).
    `lag < 0` means the render carries that much extra content before the take.
    """

    def __init__(self, lag: int, rate: int, ncc: float, probes: list[float]) -> None:
        self.lag = lag
        self.rate = rate
        self.ncc = ncc
        self.probes = probes

    @property
    def offset_sec(self) -> float:
        return self.lag / float(self.rate)

    def describe(self) -> str:
        return "offset=%+.6f s (%+d frames), ncc=%.3f" % (
            self.offset_sec,
            self.lag,
            self.ncc,
        )


def _resample_linear(samples: list[float], source_rate: int, target_rate: int) -> list[float]:
    """Linear resampling, for alignment only.

    Why linear: nothing here is ever listened to, it is only correlated.  A linear
    interpolator shifts the waveform by well under one sample while the tolerance
    is 1 ms, so a windowed-sinc resampler would add code and no accuracy where it
    matters.  It also means the reported offset is in the *render's* frames.
    """
    if source_rate == target_rate:
        return samples
    ratio = target_rate / float(source_rate)
    count = max(1, int(round(len(samples) * ratio)))
    last = len(samples) - 1
    out = []
    for index in range(count):
        position = index / ratio
        left = int(position)
        if left >= last:
            out.append(samples[last])
            continue
        fraction = position - left
        out.append(samples[left] * (1.0 - fraction) + samples[left + 1] * fraction)
    return out


def _correlate(a: list[float], b: list[float]) -> list[complex]:
    """Circular correlation whose element k is sum(a[i] * b[i - k]).

    Both inputs are zero padded past `len(a) + len(b)`, which is what makes the
    circular result equal the linear one for every lag either file can support.
    """
    size = 1
    while size < len(a) + len(b):
        size <<= 1
    fa = [complex(value, 0.0) for value in a] + [0.0j] * (size - len(a))
    fb = [complex(value, 0.0) for value in b] + [0.0j] * (size - len(b))
    _fft(fa, False)
    _fft(fb, False)
    for index in range(size):
        fa[index] = fa[index] * fb[index].conjugate()
    _fft(fa, True)
    return fa


def _exact_lag_scores(a: list[float], b: list[float], max_lag: int) -> dict[int, float]:
    """Normalised cross-correlation for every lag in +-max_lag.

    Why exact rather than "correlate a few windows": an offset gate that samples
    the file can miss the true lag on material whose loud passages are periodic,
    and it cannot tell "0.999 at the right lag" from "0.999 at the wrong one".
    The dot products cost one transform; means and energies come from prefix sums,
    so every lag after that is O(1).
    """
    correlation = _correlate(a, b)
    size = len(correlation)
    prefix_a, square_a = _prefix_sums(a)
    prefix_b, square_b = _prefix_sums(b)
    # Asymmetric on purpose: a render that starts late is *shorter* than its source,
    # so the positive side is bounded by the source, not by the render.  Bounding
    # both by `len(a)` is what makes an 18.6 s offset unsearchable on a 5 s render.
    low = -min(max_lag, len(a) - 1)
    high = min(max_lag, len(b) - 1)
    scores: dict[int, float] = {}
    for lag in range(low, high + 1):
        if lag >= 0:
            length = min(len(a), len(b) - lag)
            start_a, start_b = 0, lag
        else:
            length = min(len(a) + lag, len(b))
            start_a, start_b = -lag, 0
        if length < 32:
            continue
        # Element k of the correlation is sum(a[i] * b[i - k]), i.e. dot(-k).
        dot = correlation[(size - lag) % size].real
        sum_a = prefix_a[start_a + length] - prefix_a[start_a]
        sum_b = prefix_b[start_b + length] - prefix_b[start_b]
        energy_a = square_a[start_a + length] - square_a[start_a] - sum_a * sum_a / length
        energy_b = square_b[start_b + length] - square_b[start_b] - sum_b * sum_b / length
        if energy_a <= 1e-12 or energy_b <= 1e-12:
            continue
        covariance = dot - sum_a * sum_b / length
        scores[lag] = covariance / math.sqrt(energy_a * energy_b)
    return scores


def _scan_lags(
    a: list[float], b: list[float], low: int, high: int, count: int
) -> dict[int, float]:
    """Median-probe NCC for a small lag window (the fine pass).

    A window of +-20 ms does not justify a transform; direct sums over a few
    spread probes are both cheaper and easier to read when the number looks wrong.
    """
    scores: dict[int, float] = {}
    prefix_a, square_a = _prefix_sums(a)
    prefix_b, square_b = _prefix_sums(b)
    for lag in range(low, high + 1):
        # A negative lag is the same measurement with the roles swapped:
        # `a[i] == b[i - L]` is `b[j] == a[j + L]`.  Clamping to zero instead (the
        # first version did) silently drops every "extra head" case.
        if lag >= 0:
            x, px, sx, y, py, sy, shift = a, prefix_a, square_a, b, prefix_b, square_b, lag
        else:
            x, px, sx, y, py, sy, shift = b, prefix_b, square_b, a, prefix_a, square_a, -lag
        room = min(len(x), len(y) - shift)
        if room < 256:
            continue
        length = min(ALIGN_PROBE_FRAMES, room)
        starts = _even_starts(room - length, count)
        values = [
            _ncc_pair(x, px, sx, y, py, sy, start, start + shift, length)
            for start in starts
        ]
        scores[lag] = _median(values)
    return scores


def _verify_lag(render: list[float], source: list[float], lag: int) -> tuple[float, list[float]]:
    """Median NCC over evenly spread full-length probes at `lag`.

    The negative case is the same measurement with the roles swapped:
    `render[i] == source[i - L]` is `source[j] == render[j + L]`.
    """
    if lag >= 0:
        a, b, shift = render, source, lag
        room = min(len(a), len(b) - shift)
    else:
        a, b, shift = source, render, -lag
        room = min(len(a), len(b) - shift)
    if room < 1024:
        return 0.0, []
    length = min(PROBE_FRAMES, room)
    starts = _even_starts(room - length, PROBE_COUNT)
    prefix_a, square_a = _prefix_sums(a)
    prefix_b, square_b = _prefix_sums(b)
    values = [
        _ncc_pair(a, prefix_a, square_a, b, prefix_b, square_b, start, start + shift, length)
        for start in starts
    ]
    return _median(values), values


def passthrough_prefix(
    render: list[float], source: list[float], lag: int, rate: int
) -> float:
    """Length (seconds) of the leading region that is a verbatim copy of the source.

    Convention matches `AlignResult`: `render[i] == source[i + lag]`.

    Why this reading exists: a processed render is *not* a copy of its source, so a
    correct product answers 0.  A render whose head was passed through -- the host's
    original audio written into the product instead of the processed result -- answers
    the length of that head.  That is exactly the shape of the VEGAS defect this round
    fixed, and "listen to it" is not an acceptance criterion a script can run.
    """
    if lag >= 0:
        a, b, shift = render, source, lag
    else:
        a, b, shift = source, render, -lag
    room = min(len(a), len(b) - shift)
    window = max(256, int(round(rate * PASSTHROUGH_WINDOW_SEC)))
    if room < window:
        return 0.0
    prefix_a, square_a = _prefix_sums(a)
    prefix_b, square_b = _prefix_sums(b)
    step = max(1, window // 2)
    covered = 0
    start = 0
    while start + window <= room:
        score = _ncc_pair(
            a, prefix_a, square_a, b, prefix_b, square_b, start, start + shift, window
        )
        if score < PASSTHROUGH_LIMIT:
            break
        covered = start + window
        start += step
    return covered / float(rate)


def align_to_source(
    render_path: str, source_path: str, max_offset_sec: float = DEFAULT_MAX_OFFSET_SEC
) -> tuple[AlignResult, int, int]:
    """Locate the render inside its source.  Returns (result, render rate, source rate)."""
    render, render_rate = read_wave(render_path)
    source, source_rate = read_wave(source_path)
    # One time base, the render's: every number this tool prints is in its frames.
    source = _resample_linear(source, source_rate, render_rate)
    head = int(round(ALIGN_MAX_SEC * render_rate))
    render_head = render[:head]
    source_head = source[:head]
    coarse_factor = max(1, int(round(render_rate / float(ALIGN_COARSE_RATE))))
    fine_factor = max(1, int(round(render_rate / float(ALIGN_FINE_RATE))))
    coarse = _exact_lag_scores(
        decimate(render_head, coarse_factor),
        decimate(source_head, coarse_factor),
        int(round(max_offset_sec * ALIGN_COARSE_RATE)),
    )
    if not coarse:
        raise WavReadError("the two files share no usable overlap to align on")
    # Keep peaks at least one coarse step apart, so one broad peak cannot fill the
    # whole candidate list.
    ordered = sorted(coarse.items(), key=lambda item: item[1], reverse=True)
    picked: list[tuple[int, float]] = []
    for lag, score in ordered:
        if all(abs(lag - other) > 1 for other, _score in picked):
            picked.append((lag, score))
        if len(picked) >= ALIGN_CANDIDATES:
            break
    fine_render = decimate(render_head, fine_factor)
    fine_source = decimate(source_head, fine_factor)
    # One coarse step is `coarse_factor / fine_factor` fine steps -- 10 when the
    # grids are 10 ms and 1 ms.  Dividing the other way rounds every candidate to
    # lag 0, and the verification then reports a correlation near zero, which reads
    # like "different audio" instead of like a bug.
    coarse_to_fine = max(1, coarse_factor // fine_factor)
    best_lag = picked[0][0] * coarse_to_fine
    best_score = -2.0
    for coarse_lag, _score in picked:
        centre = coarse_lag * coarse_to_fine
        span = 2 * coarse_to_fine + 2
        window = _scan_lags(
            fine_render, fine_source, centre - span, centre + span, SEARCH_PROBE_COUNT
        )
        if not window:
            continue
        lag, score = max(window.items(), key=lambda item: item[1])
        if score > best_score:
            best_score = score
            best_lag = lag
    # Full-rate refinement: the 1 ms grid is the *search*, not the answer.
    centre = best_lag * fine_factor
    prefix, prefix_square = _prefix_sums(render)
    source_prefix, source_square = _prefix_sums(source)
    probe_window = min(SEARCH_PROBE_FRAMES, max(1024, len(render) // 8))

    def scan(lag: int) -> float:
        # Negative lags are the same measurement with the roles swapped (as in
        # `_verify_lag`).  Without this branch every "the render carries extra head"
        # case would be reported as a positive offset.
        if lag >= 0:
            x, px, sx = render, prefix, prefix_square
            y, py, sy, shift = source, source_prefix, source_square, lag
        else:
            x, px, sx = source, source_prefix, source_square
            y, py, sy, shift = render, prefix, prefix_square, -lag
        room = min(len(x), len(y) - shift)
        if room < 1024:
            return -2.0
        window = min(probe_window, room)
        starts = _loud_starts(px, room - window, window, SEARCH_PROBE_COUNT)
        return _median(
            [
                _ncc_pair(x, px, sx, y, py, sy, start, start + shift, window)
                for start in starts
            ]
        )

    # Not `_search_lag`: it starts at lag 1 by design (periods are positive), which
    # would skip zero and every negative lag -- the two cases that matter most here.
    best_lag = centre
    best_score = -2.0
    for lag in range(centre - fine_factor, centre + fine_factor + 1):
        score = scan(lag)
        if score > best_score:
            best_score = score
            best_lag = lag
    median, probes = _verify_lag(render, source, best_lag)
    return AlignResult(best_lag, render_rate, median, probes), render_rate, source_rate


# -- Self test -------------------------------------------------------------


def _write_wave(path: str, samples: list[float], rate: int, bits: int = 32) -> None:
    """Minimal RIFF/WAVE writer: mono, 32-bit IEEE float or 16-bit PCM."""
    if bits == 16:
        payload = struct.pack(
            "<%dh" % len(samples), *(int(round(v * 32767.0)) for v in samples)
        )
        tag = WAVE_FORMAT_PCM
    else:
        payload = struct.pack("<%df" % len(samples), *samples)
        tag = WAVE_FORMAT_IEEE_FLOAT
        bits = 32
    block = bits // 8
    header = b"RIFF" + struct.pack("<I", 36 + len(payload)) + b"WAVE"
    header += b"fmt " + struct.pack(
        "<IHHIIHH", 16, tag, 1, rate, rate * block, block, bits
    )
    header += b"data" + struct.pack("<I", len(payload))
    with open(path, "wb") as handle:
        handle.write(header)
        handle.write(payload)


def _noise(count: int, seed: int = 12345) -> list[float]:
    state = seed
    out = []
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out.append(state / 1073741824.0 - 1.0)
    return out


def _loop_block(period: int, rate: int, seed: int = 7) -> list[float]:
    """One period of a drum-ish loop: four decaying bursts at different pitches.

    The pitches differ per burst on purpose.  If they did not, the block would also
    repeat at a quarter of its length, and the tool would (correctly) report the
    shorter period -- which would say nothing about the case being tested.
    """
    noise = _noise(period, seed=seed)
    out = []
    for index in range(period):
        t = index / float(rate)
        burst = int(t / 0.25)
        phase = t - burst * 0.25
        pitch = (220.0, 277.18, 329.63, 415.30)[burst % 4]
        fade = math.exp(-9.0 * phase)
        out.append(
            0.6 * fade * math.sin(2.0 * math.pi * pitch * phase)
            + 0.25 * fade * noise[index]
        )
    return out


def self_test() -> int:
    """A gate must fail on a known-bad sample and pass on a known-good one.

    Both repeating samples use periods that are **not** a whole number of seconds
    (49 152 frames = 1.024 s and 33 600 = 0.700 s), because that is the shape the
    real defect had, and a round period is exactly what a window-grid criterion
    gets right by luck.  The noise sample pins the other direction: a gate that
    only ever fails is as useless as one that only ever passes.
    """
    rate = 48000
    directory = tempfile.mkdtemp(prefix="render-repeat-")
    failures = 0
    cases = (
        ("repeating-1.024s.wav", 49152, 30, 32),
        ("repeating-0.700s.wav", 33600, 20, 16),
    )
    for name, period, repeats, bits in cases:
        block = _loop_block(period, rate)
        samples = block + block * (repeats - 1)
        path = os.path.join(directory, name)
        _write_wave(path, samples, rate, bits)
        found, _seconds, median, _rate = audit(path)
        wanted = "repeat at %d frames" % period
        if found is None:
            print("self-test: FAIL: %s not detected (median=%.3f)" % (wanted, median))
            failures += 1
        elif found.period != period:
            print("self-test: FAIL: %d frames reported, expected %d" % (found.period, period))
            failures += 1
        else:
            print("self-test: ok: %s detected (%s)" % (wanted, found.describe(rate)))
    good = os.path.join(directory, "noise.wav")
    _write_wave(good, _noise(rate * 20, seed=11), rate)
    quiet, _seconds, median, _rate = audit(good)
    if quiet is not None:
        print("self-test: FAIL: noise was reported as a repeat (%s)" % quiet.describe(rate))
        failures += 1
    else:
        print("self-test: ok: noise rejected (best median=%.3f)" % median)
    failures += _align_self_test(directory, rate)
    return 1 if failures else 0


def _align_self_test(directory: str, rate: int) -> int:
    """A render shifted by a known amount must report that amount.

    Why noise: its correlation peak is a single sharp spike, so the case tests the
    search itself rather than the material.  Why three cases: an aligner that only
    ever reports "misaligned" would pass a positive-only check, and one that only
    ever reports zero would pass a zero-only check.

    The positive offsets are the two the field actually produced (1.85 s and
    18.6 s); the negative one covers "the render carries extra head", which is what
    a pre-roll leak looks like.
    """
    failures = 0
    source = _noise(rate * 24, seed=21)
    source_path = os.path.join(directory, "align-source.wav")
    _write_wave(source_path, source, rate)
    cases = (
        ("align-1.85s.wav", 88800, 0, source_path),
        ("align-18.6s.wav", 892800, 0, source_path),
        ("align-zero.wav", 0, 0, source_path),
        ("align-negative.wav", -24000, 0, source_path),
    )
    for name, expected, tolerance, reference in cases:
        if expected >= 0:
            samples = source[expected:]
        else:
            samples = [0.0] * (-expected) + source[: len(source) + expected]
        path = os.path.join(directory, name)
        _write_wave(path, samples, rate)
        failures += _check_alignment(name, path, reference, expected, tolerance)
    # Different sample rates: media at 44.1 kHz, project (render) at 48 kHz.  That
    # is the common case in the field, and it is the only path where the source is
    # resampled -- so it is the only one where a rate mix-up could hide.
    media_path = os.path.join(directory, "align-source-44k.wav")
    _write_wave(media_path, _resample_linear(source, rate, 44100), 44100)
    mixed_path = os.path.join(directory, "align-44k-render.wav")
    _write_wave(mixed_path, source, rate)
    # Two frames of slack: the resampler is linear by design (see `_resample_linear`),
    # which is far below the 1 ms the tool actually judges.
    failures += _check_alignment("align-44k-render.wav", mixed_path, media_path, 0, 2)
    failures += _passthrough_self_test(directory, rate, source)
    failures += _expected_offset_self_test(directory, rate, source)
    return failures


def _passthrough_self_test(directory: str, rate: int, source: list[float]) -> int:
    """A head that is a verbatim copy must be reported; a processed file must not be.

    Both directions matter: a check that always reports a prefix would call every
    correct render broken, and one that never reports it would miss the field
    symptom this round fixed ("the first N seconds of the product are the host's
    original audio").
    """
    failures = 0
    boundary = int(round(rate * 1.5))
    # A "processed" tail: the source scaled and mixed with a quiet tone, so it is
    # the same material but not a sample-for-sample copy.
    tail = [
        source[index] * 0.5 + 0.05 * math.sin(index * 0.01)
        for index in range(boundary, len(source))
    ]
    with_head = source[:boundary] + tail
    head_path = os.path.join(directory, "passthrough-head.wav")
    _write_wave(head_path, with_head, rate)
    samples, _rate = read_wave(head_path)
    measured = passthrough_prefix(samples, source, 0, rate)
    if not 1.3 <= measured <= 1.6:
        print("self-test: FAIL: passthrough head reported %.3f s, expected about 1.5 s" % measured)
        failures += 1
    else:
        print("self-test: ok: passthrough head reported (%.3f s)" % measured)
    processed_path = os.path.join(directory, "passthrough-none.wav")
    _write_wave(processed_path, tail, rate)
    processed, _rate = read_wave(processed_path)
    none = passthrough_prefix(processed, source[boundary:], 0, rate)
    if none > 0.0:
        print("self-test: FAIL: a processed render reported a %.3f s passthrough head" % none)
        failures += 1
    else:
        print("self-test: ok: processed render reported no passthrough head")
    return failures


def _expected_offset_self_test(directory: str, rate: int, source: list[float]) -> int:
    """`--expect-offset` must accept the true offset and reject a wrong one.

    This is the verdict the acceptance matrix actually runs: "rendered from 28 s to
    61 s" produces a file whose lag must be 0, and a wrong mapping produces a lag
    equal to the range start.  Both are *aligned* files; only the expectation
    separates them.
    """
    failures = 0
    source_path = os.path.join(directory, "expect-source.wav")
    _write_wave(source_path, source, rate)
    shifted = source[int(round(rate * 2.0)) :]
    shifted_path = os.path.join(directory, "expect-2s.wav")
    _write_wave(shifted_path, shifted, rate)
    matched = align_main(
        [shifted_path, source_path, "--expect-offset", "2.0", "--tolerance", "0.001"]
    )
    if matched != 0:
        print("self-test: FAIL: --expect-offset rejected the true offset (exit %d)" % matched)
        failures += 1
    else:
        print("self-test: ok: --expect-offset accepted the true offset")
    mismatched = align_main(
        [shifted_path, source_path, "--expect-offset", "0.0", "--tolerance", "0.001"]
    )
    if mismatched == 0:
        print("self-test: FAIL: --expect-offset accepted a wrong offset")
        failures += 1
    else:
        print("self-test: ok: --expect-offset rejected a wrong offset")
    return failures


def _check_alignment(
    name: str, path: str, reference: str, expected: int, tolerance: int
) -> int:
    try:
        result, _render_rate, _source_rate = align_to_source(path, reference)
    except (OSError, WavReadError) as error:
        print("self-test: FAIL: %s could not be aligned: %s" % (name, error))
        return 1
    if abs(result.lag - expected) > tolerance:
        print(
            "self-test: FAIL: %s reported %+d frames, expected %+d (%s)"
            % (name, result.lag, expected, result.describe())
        )
        return 1
    print("self-test: ok: %s aligned (%s)" % (name, result.describe()))
    return 0


# -- Entry point -----------------------------------------------------------


def align_main(argv: list[str]) -> int:
    """`--align <render> <source> [--max-offset SEC] [--tolerance SEC] [--expect-offset SEC]`."""
    max_offset = DEFAULT_MAX_OFFSET_SEC
    tolerance = ALIGN_TOLERANCE_SEC
    # 【为什么要 --expect-offset】`--tolerance` 回答的是"这段音频对齐了吗"，
    # 而验收要回答的是另一件事："它对齐到了**它应该在的地方**吗"。一次从工程 28 s
    # 起渲染的产物，正确时 lag 是 0（产物从素材头开始），而"整段平移 28 s"在
    # 对齐读数上与"素材本来就从 28 s 开始"完全同形。给出期望值，判据才有主语。
    expect_offset: float | None = None
    paths: list[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument in ("--max-offset", "--tolerance", "--expect-offset") and index + 1 < len(
            argv
        ):
            try:
                value = float(argv[index + 1])
            except ValueError:
                print("error: %s wants a number, got %r" % (argument, argv[index + 1]))
                return 2
            if argument == "--max-offset":
                max_offset = value
            elif argument == "--tolerance":
                tolerance = value
            else:
                expect_offset = value
            index += 2
            continue
        paths.append(argument)
        index += 1
    if len(paths) != 2:
        print(
            "usage: render-repeat-check.py --align <render.wav> <source.wav> "
            "[--max-offset SEC] [--tolerance SEC] [--expect-offset SEC]"
        )
        return 2
    try:
        result, render_rate, source_rate = align_to_source(paths[0], paths[1], max_offset)
    except (OSError, WavReadError) as error:
        print("error: %s: %s" % (paths[0], error))
        return 2
    print("align: render=%s (%d Hz)" % (paths[0], render_rate))
    print("align: source=%s (%d Hz)" % (paths[1], source_rate))
    print("align: %s" % result.describe())
    for probe_index, probe in enumerate(result.probes):
        print("  probe %d: ncc=%.4f" % (probe_index, probe))
    # 【直通前缀】产物头部有多少秒是**素材的逐样本副本**。处理过的产物是 0；
    # "前 N 秒是宿主原声"就是它的长度。判据与读法见 `passthrough_prefix`。
    try:
        render_samples, _rate = read_wave(paths[0])
        source_samples, source_rate = read_wave(paths[1])
    except (OSError, WavReadError) as error:
        print("error: %s: %s" % (paths[0], error))
        return 2
    source_on_render_rate = _resample_linear(source_samples, source_rate, render_rate)
    passthrough = passthrough_prefix(
        render_samples, source_on_render_rate, result.lag, render_rate
    )
    print("align: passthrough=%.6f s" % passthrough)
    # 【期望值取代"必须为 0"，而不是叠加在它上面】一次"从工程 28 s 渲染"的产物
    # 正确时 lag 是 0；而"整段平移 28 s"的产物 lag 是 -28 s 之外的另一段音频 ——
    # 两者都可能在"是否对齐"上给出同一个答案。给了期望值时，判据只有一个：
    # 实测偏移是否等于期望偏移。
    if expect_offset is not None:
        delta = abs(result.offset_sec - expect_offset)
        if delta <= tolerance:
            print(
                "align: expected offset %+.6f s confirmed (delta %.6f s)"
                % (expect_offset, delta)
            )
            return 0
        print(
            "align: expected offset %+.6f s but measured %+.6f s (delta %.6f s)"
            % (expect_offset, result.offset_sec, delta)
        )
        return 1
    if abs(result.offset_sec) > tolerance:
        print("align: misaligned (tolerance %g s)" % tolerance)
        return 1
    print("align: ok (within %g s)" % tolerance)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0])
        print(
            "usage: render-repeat-check.py <file.wav> [more.wav ...] | "
            "--align <render.wav> <source.wav> | --self-test"
        )
        return 2
    if argv[1] == "--self-test":
        return self_test()
    if argv[1] == "--align":
        return align_main(argv[2:])
    failed = False
    for path in argv[1:]:
        try:
            repeat, seconds, median, rate = audit(path)
        except (OSError, WavReadError) as error:
            print("error: %s: %s" % (path, error))
            return 2
        if repeat is None:
            print(
                "ok: %s: no repeating period (closest candidate %.3f s at ncc=%.3f)"
                % (path, seconds, median)
            )
            continue
        failed = True
        print("repeat: %s: %s" % (path, repeat.describe(rate)))
        for index, probe in enumerate(repeat.probes):
            print("  probe %d: ncc=%.4f" % (index, probe))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
