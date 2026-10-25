"""Read-only accompaniment BPM analysis using the existing local Beat This!.

Separate from the chord worker: no BTC, source edits, downloads or project
writes. Input contains editor-decoded visible source regions only.
"""
from pathlib import Path
import argparse
import json
import math
import os
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'deps')]


def estimate_tempo(beats):
    beats = [float(t) for t in beats]
    if len(beats) < 4 or any(not math.isfinite(t) for t in beats):
        raise ValueError('no_beats')
    intervals = [b - a for a, b in zip(beats, beats[1:])]
    if any(t <= 0 for t in intervals):
        raise ValueError('no_beats')
    period = statistics.median(intervals)
    # Long-baseline periods reduce the model's 20 ms event quantization.
    # Account for isolated missed beats without letting a long silence count
    # as a long chain of evenly spaced beats.
    width = min(8, len(intervals))
    periods = []
    for i in range(len(intervals) - width + 1):
        window = intervals[i:i + width]
        if any(t < .55 * period or t > 3.5 * period for t in window):
            continue
        steps = sum(max(1, round(t / period)) for t in window)
        periods.append(sum(window) / steps)
    if periods:
        period = statistics.median(periods)
    bpm = 60 / period
    if not math.isfinite(bpm) or not 20 <= bpm <= 400:
        raise ValueError('no_beats')
    ordered = sorted(periods)
    variation = ((ordered[int(.9 * (len(ordered) - 1))] - ordered[int(.1 * (len(ordered) - 1))])
                 / period) if len(ordered) >= 3 else 0
    return dict(bpm=bpm, variable_tempo=variation > .08)


def analyse(request):
    weights = ROOT / 'models/beat-this-small0.ckpt'
    if not weights.is_file():
        return dict(ok=False, code='missing_model')
    import numpy as np
    import soundfile as sf
    import soxr
    import torch
    from beat_this.inference import Audio2Beats
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    duration = float(request['duration'])
    if not math.isfinite(duration) or duration < 2:
        return dict(ok=False, code='too_short')
    start = time.monotonic()
    rate = 22050
    audio = np.zeros(int(math.ceil(duration * rate)), dtype=np.float32)
    for segment in request['segments']:
        samples, sr = sf.read(segment['file'], dtype='float32', always_2d=True)
        samples = samples.mean(axis=1)
        if sr != rate:
            samples = soxr.resample(samples, sr, rate)
        first = int(round(float(segment['start']) * rate))
        count = min(len(samples), len(audio) - first)
        if count > 0:
            audio[first:first + count] += samples[:count]
    if not np.isfinite(audio).all():
        return dict(ok=False, code='invalid_audio')
    if np.max(np.abs(audio), initial=0) < 1e-7:
        return dict(ok=False, code='no_beats')
    # This is a fixed existing file, never a checkpoint shortname or URL.
    model = Audio2Beats(checkpoint_path=str(weights), device='cpu', dbn=False)
    with torch.inference_mode():
        beats, downbeats = model(audio, rate)
    try:
        tempo = estimate_tempo(beats)
    except ValueError:
        return dict(ok=False, code='no_beats')
    return dict(ok=True, **tempo, beat_count=len(beats), downbeat_count=len(downbeats),
                duration=duration, model='Beat This! small0', processing_seconds=time.monotonic() - start)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        result = analyse(json.loads(Path(args.request).read_text(encoding='utf-8-sig')))
    except Exception as error:
        result = dict(ok=False, code='worker_error', error=str(error))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
