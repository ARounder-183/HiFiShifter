import { test } from "vitest";

import {
    MAX_STRONG_GRID_LINES,
    MAX_WEAK_GRID_LINES,
    resolveGridLineSamplingPlan,
    resolveGridLineSpacing,
    selectStrongGridBarMultiple,
    selectUniformGridStepBeats,
} from "./gridLineSampling.ts";

test("components/layout/timeline/gridLineSampling.test.ts scripted checks", async () => {
    let checks = 0;

    function assertEqual(actual: number, expected: number, label: string): void {
        checks += 1;
        if (Math.abs(actual - expected) > 1e-9) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    // 网格步长足够宽时保持原始网格精度。
    assertEqual(
        selectUniformGridStepBeats({
            pxPerBeat: 240,
            grid: "1/16",
            beatsPerBar: 4,
            minSpacingPx: 12,
        }),
        0.25,
        "wide grid keeps the raw grid step",
    );

    // 网格过密时提升到音乐上均匀的下一级，而不是任意抽样。
    assertEqual(
        selectUniformGridStepBeats({
            pxPerBeat: 12,
            grid: "1/16",
            beatsPerBar: 4,
            minSpacingPx: 12,
        }),
        1,
        "dense grid falls back to the next uniform beat step",
    );

    // 3/4 拍下仍以小节对齐的步长为候选，避免出现不均匀间距。
    assertEqual(
        selectUniformGridStepBeats({
            pxPerBeat: 12,
            grid: "1/8",
            beatsPerBar: 3,
            minSpacingPx: 12,
        }),
        1,
        "3/4 grid never falls back to a half-bar step",
    );

    // 强网格按小节整数倍抽样。
    assertEqual(selectStrongGridBarMultiple(24, 12), 1, "wide bar step stays unchanged");
    assertEqual(selectStrongGridBarMultiple(4, 25), 8, "dense bars sample every eighth bar");

    // 视口越大，允许的绝对间距越大，从而把总行数限制在固定预算内。
    assertEqual(resolveGridLineSpacing(1000, 100, 8), 10, "viewport budget controls spacing");
    assertEqual(resolveGridLineSpacing(4000, MAX_WEAK_GRID_LINES, 8), 25, "wide viewport budget");

    // 宽视口 + 极细网格：采样后的可见线数量保持有界。
    {
        const plan = resolveGridLineSamplingPlan({
            pxPerBeat: 8,
            grid: "1/64",
            beatsPerBar: 4,
            viewportWidth: 4000,
        });
        assertEqual(
            plan.weakStepPx,
            32,
            `weak increment matches ${MAX_WEAK_GRID_LINES}-line budget`,
        );
        assertEqual(
            plan.strongStepPx,
            128,
            `strong increment matches ${MAX_STRONG_GRID_LINES}-line budget`,
        );
    }

});
