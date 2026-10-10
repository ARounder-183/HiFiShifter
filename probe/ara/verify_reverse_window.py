#!/usr/bin/env python3
"""verify_reverse_window.py —— F-4 的决定性判定：倒放 take 实际播放的源窗口。

读 `build_reverse_window_probe.lua` 渲染出的 `captures/reverse-window.wav`（那一个倒放
item、**不带 FX** 的原始播放），与源夹具的两个候选窗口逐样本比对：

  * A：源内 [0.25, 1.25] 倒放  —— 内容不变、只有方向翻转（"坐标被翻"假设）
  * B：源内 [0.75, 1.75] 倒放  —— 内容也变了（"坐标即内容"假设）

命中 A ⇒ ARA region 报的是**已镜像坐标**；命中 B ⇒ 报的是**正向坐标**。
两个都不是 ⇒ 人工看，不硬判。

用法：python probe/ara/verify_reverse_window.py
产物：captures/reverse-window.json
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "phase3a-asymmetric.wav"
RENDERED = HERE / "captures" / "reverse-window.wav"
OUTPUT = HERE / "captures" / "reverse-window.json"

SOURCE_OFFSET = 0.25
ITEM_LENGTH = 1.0
RATE = 44100
# 允许渲染起点有几样本的偏差（REAPER 的渲染对齐不应有，但留一点余量再比对）。
ALIGN_SEARCH = 64
# REAPER 在 item 两端会加极短的防爆音淡变（即便 D_FADE*LEN 设为 0），首尾各几样本
# 会偏离纯倒放。比较时掐掉两端，只看中间 —— 否则边界样本会把匹配判成"不够近"。
EDGE_MARGIN = 256
MATCH_TOLERANCE = 1e-4  # 24bit 量化 + 极短淡变余量；两候选差异远大于此。


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
    EDGE_MARGIN 个样本，避开 REAPER 的防爆音淡变。"""
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
    window_a = fixture[start : start + span]
    # 候选 B：源内 [0.75, 1.75] —— 镜像窗口 = 源长 - 起点 - 时长 = 2.0 - 0.25 - 1.0。
    mirror_start = round((len(fixture) / RATE - SOURCE_OFFSET - ITEM_LENGTH) * RATE)
    window_b = fixture[mirror_start : mirror_start + span]
    if len(window_a) != span or len(window_b) != span:
        raise SystemExit("fixture is too short for the probe windows")

    # 倒放播放 = 窗口的时间反转。
    expected_a = list(reversed(window_a))
    expected_b = list(reversed(window_b))

    diff_a, shift_a = best_aligned_diff(rendered, expected_a)
    diff_b, shift_b = best_aligned_diff(rendered, expected_b)

    covered = None
    if diff_a <= MATCH_TOLERANCE and diff_a < diff_b:
        covered = "forward-trim"
        coordinate = "mirrored"
    elif diff_b <= MATCH_TOLERANCE and diff_b < diff_a:
        covered = "shifted"
        coordinate = "forward"
    else:
        coordinate = None

    result = {
        "provenance": "Rendered the reversed item with no FX; compared against reversed windows of fixtures/phase3a-asymmetric.wav",
        "fixture": {"rate": rate, "bits": bits, "frames": len(fixture)},
        "rendered": {"rate": rrate, "bits": rbits, "frames": len(rendered)},
        "candidates": {
            "A_forward_trim_0.25_1.25": {"maxAbsDiff": diff_a, "alignShift": shift_a},
            "B_shifted_0.75_1.75": {"maxAbsDiff": diff_b, "alignShift": shift_b},
        },
        "conclusion": {
            "coveredWindow": covered,
            "coordinateSystem": coordinate,
            "task34Implication": (
                "Region time is mirrored: recover the forward window before applying the kernel reverse math."
                if coordinate == "mirrored"
                else "Region time is forward: the kernel reverse math can consume source_end_sec as-is."
                if coordinate == "forward"
                else "Neither candidate matched; F-4 needs a human."
            ),
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"F-4 window: A(diff={diff_a:.3e}) B(diff={diff_b:.3e}) -> "
        f"covered={covered} coordinate={coordinate}"
    )
    return 0 if coordinate else 1


if __name__ == "__main__":
    sys.exit(main())
