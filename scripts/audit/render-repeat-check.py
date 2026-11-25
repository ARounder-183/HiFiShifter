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

Usage:
    python scripts/audit/render-repeat-check.py <file.wav> [more.wav ...]
    python scripts/audit/render-repeat-check.py --self-test

Exit code 0 = no repetition found, 1 = repetition found, 2 = usage/read error.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile

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
    return 1 if failures else 0


# -- Entry point -----------------------------------------------------------


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(__doc__.strip().splitlines()[0])
        print("usage: render-repeat-check.py <file.wav> [more.wav ...] | --self-test")
        return 2
    if argv[1] == "--self-test":
        return self_test()
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
