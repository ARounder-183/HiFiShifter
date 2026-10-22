import { test } from "vitest";

import {
    GAIN_MAX_DB,
    GAIN_MAX_LINEAR,
    GAIN_MIN_DB,
    clampGain,
    gainDbFromKnobTravel,
    gainKnobAngleDeg,
    gainToDbSafe,
} from "./gainDb.ts";

function near(actual: number, expected: number, label: string, eps = 1e-6): void {
    if (Math.abs(actual - expected) > eps) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

test("utils/gainDb.test.ts scripted checks", () => {
    // ── 值域 ────────────────────────────────────────────────────────────
    near(GAIN_MAX_DB, 24.0824, "上界必须是 REAPER 的 +24.08 dB", 1e-3);
    near(clampGain(1), 1, "单位增益不变");
    near(clampGain(-5), 0, "负增益钳到静音");
    near(clampGain(1e9), GAIN_MAX_LINEAR, "超上界钳到 16");
    near(clampGain(Number.NaN), 1, "NaN 回退到不增不减（不能变成静音）");
    near(gainToDbSafe(0), GAIN_MIN_DB, "0 线性映射到实用下界而不是 −Infinity");

    // ── 旋钮角度：0 dB 恒在正中 ───────────────────────────────────────
    near(gainKnobAngleDeg(0), 0, "0 dB 的指示线竖直向上");
    near(gainKnobAngleDeg(GAIN_MAX_DB), 135, "上界偏转 +135°");
    near(gainKnobAngleDeg(GAIN_MIN_DB), -135, "下界偏转 −135°");

    // ── 拖拽手感：全程行程与值域无关，且与旧实现（96px）一致 ───────────
    // 从 0 dB 往上推 48px 恰好到顶（旧实现 ±12dB @ 0.25dB/px 也是 48px 到顶）。
    near(gainDbFromKnobTravel(0, 48), GAIN_MAX_DB, "0 dB 上推 48px 到顶", 1e-3);
    near(gainDbFromKnobTravel(0, -48), GAIN_MIN_DB, "0 dB 下推 48px 到底", 1e-3);
    // 微小位移不得产生大幅变化（用户明确要求的"手感不变"）。
    const tiny = gainDbFromKnobTravel(0, 1);
    if (!(tiny > 0 && tiny < 1)) {
        throw new Error(`1px 位移只应带来不到 1dB 的变化，实际 ${tiny}`);
    }
    // 往返一致：推上去再推回来。
    const up = gainDbFromKnobTravel(0, 12);
    near(gainDbFromKnobTravel(up, -12), 0, "同一段位移往返必须回到原点", 1e-3);
    // 越界钳制。
    near(gainDbFromKnobTravel(0, 10_000), GAIN_MAX_DB, "推到天上去也只到上界");
    near(gainDbFromKnobTravel(0, -10_000), GAIN_MIN_DB, "推到地下也只到下界");
});
