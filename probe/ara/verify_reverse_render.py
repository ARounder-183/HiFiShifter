#!/usr/bin/env python3
"""verify_reverse_render.py —— Task 3.4 的端到端判定：**插件**渲染的倒放片段。

读 `build_reverse_render_probe.lua` 渲染出的 `captures/reverse-render.wav`（挂了真
HiFiShifter FX，一个倒放 item 的播放），与源夹具的四个候选逐样本比对：

  * reverse_forward_trim  = reverse(源[0.25, 1.25])  —— **正确**（镜像窗口已翻回正向）
  * reverse_mirrored      = reverse(源[0.75, 1.75])  —— 窗口没翻正（二次镜像前的旧行为）
  * forward_trim          = 源[0.25, 1.25]           —— 压根没倒放
  * forward_mirrored      = 源[0.75, 1.75]           —— 没倒放且窗口镜像

命中 `reverse_forward_trim` ⇒ Task 3.4 生效。命中 `reverse_mirrored` ⇒ 窗口仍镜像。
命中任一 forward ⇒ 方向没进渲染时间线。都不中 ⇒ 人工看，不硬判。

用法：python probe/ara/verify_reverse_render.py
产物：captures/reverse-render.json
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "phase3a-asymmetric.wav"
RENDERED = HERE / "captures" / "reverse-render.wav"
OUTPUT = HERE / "captures" / "reverse-render.json"

SOURCE_OFFSET = 0.25
ITEM_LENGTH = 1.0
RATE = 44100
# 插件渲染可能有几样本的延迟/对齐偏差；给足搜索范围。
ALIGN_SEARCH = 512
# 插件输出经过声码器链，即便"未编辑"也不会与源逐样本一致；边界淡变也要掐掉。
EDGE_MARGIN = 512
# 插件链的容差：只要候选之间的差异远大于它，就能可靠区分"哪一段"。
MATCH_TOLERANCE = 5e-2


def read_wav_mono(path: Path) -> tuple[int, int, list[float]]:
    """返回 (sample_rate, bits, samples)；只支持 mono PCM16/24/32 与 float32。"""
    data = path.read_bytes()
    if data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path} is not a RIFF/WAVE file")
    offset = 12
    fmt = channels = rate = bits = None
    samples: list[float] = []
    while offset + 8 <= len(data):
        chunk = data[offset : offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        body = data[offset + 8 : offset + 8 + size]
        if chunk == b"fmt ":
            fmt, channels, rate, _, _, bits = struct.unpack_from("<HHIIHH", body, 0)
        elif chunk == b"data":
            if fmt == 3 and bits == 32:
                samples = list(struct.unpack(f"<{size // 4}f", body))
            elif fmt == 1 and bits == 16:
                samples = [v / 32768.0 for v in struct.unpack(f"<{size // 2}h", body)]
            elif fmt == 1 and bits == 24:
                for i in range(0, size, 3):
                    raw = body[i] | (body[i + 1] << 8) | (body[i + 2] << 16)
                    if raw & 0x800000:
                        raw -= 0x1000000
                    samples.append(raw / 8388608.0)
            else:
                raise ValueError(f"unsupported WAV format {fmt} / {bits} bits")
        offset += 8 + size + (size % 2)
    if channels != 1:
        raise ValueError(f"expected mono, got {channels} channels")
    return rate, bits, samples


def max_abs_diff(a: list[float], b: list[float]) -> float:
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def best_aligned_diff(rendered: list[float], expected: list[float]) -> tuple[float, int]:
    """在 ±ALIGN_SEARCH 样本内找对齐，返回 (最小最大绝对差, 偏移)。两端各掐掉
    EDGE_MARGIN 个样本，避开边界淡变与插件链的瞬态。"""
    best = (float("inf"), 0)
    for shift in range(-ALIGN_SEARCH, ALIGN_SEARCH + 1):
        lo = max(EDGE_MARGIN, shift + EDGE_MARGIN)
        hi = min(len(rendered) - EDGE_MARGIN, len(expected) + shift - EDGE_MARGIN)
        if hi - lo < len(expected) // 2:
            continue
        segment = [rendered[i] for i in range(lo, hi)]
        reference = [expected[i - shift] for i in range(lo, hi)]
        diff = max_abs_diff(segment, reference)
        if diff < best[0]:
            best = (diff, shift)
    return best


def main() -> int:
    rate, bits, fixture = read_wav_mono(FIXTURE)
    if rate != RATE:
        raise SystemExit(f"fixture rate {rate} != {RATE}")
    rrate, rbits, rendered = read_wav_mono(RENDERED)
    if rrate != RATE:
        raise SystemExit(f"render rate {rrate} != {RATE}")

    start = round(SOURCE_OFFSET * RATE)
    span = round(ITEM_LENGTH * RATE)
    forward = fixture[start : start + span]
    mirror_start = round((len(fixture) / RATE - SOURCE_OFFSET - ITEM_LENGTH) * RATE)
    mirrored = fixture[mirror_start : mirror_start + span]
    if len(forward) != span or len(mirrored) != span:
        raise SystemExit("fixture is too short for the probe windows")

    candidates = {
        "reverse_forward_trim": list(reversed(forward)),
        "reverse_mirrored": list(reversed(mirrored)),
        "forward_trim": forward,
        "forward_mirrored": mirrored,
    }
    diffs = {}
    for name, expected in candidates.items():
        diff, shift = best_aligned_diff(rendered, expected)
        diffs[name] = {"maxAbsDiff": diff, "alignShift": shift}

    best_name = min(diffs, key=lambda name: diffs[name]["maxAbsDiff"])
    best_diff = diffs[best_name]["maxAbsDiff"]
    matched = best_name if best_diff <= MATCH_TOLERANCE else None

    verdict = {
        "reverse_forward_trim": "task34_active: the plugin reversed the forward window",
        "reverse_mirrored": "window still mirrored: the seeding correction did not apply",
        "forward_trim": "not reversed: direction never reached the render timeline",
        "forward_mirrored": "not reversed and window mirrored",
    }.get(matched, "no candidate matched within tolerance; needs a human")

    result = {
        "provenance": "Rendered the reversed item through the HiFiShifter VST3 plugin; compared against windows of fixtures/phase3a-asymmetric.wav",
        "fixture": {"rate": rate, "bits": bits, "frames": len(fixture)},
        "rendered": {"rate": rrate, "bits": rbits, "frames": len(rendered)},
        "candidates": diffs,
        "conclusion": {"matched": matched, "verdict": verdict},
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = " ".join(f"{name}={diffs[name]['maxAbsDiff']:.3e}" for name in candidates)
    print(f"Task 3.4 render: {summary} -> matched={matched}")
    return 0 if matched else 1


if __name__ == "__main__":
    sys.exit(main())
