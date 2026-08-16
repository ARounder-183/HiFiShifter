/**
 * `gridLineKey.ts` 的回归测试（自执行断言脚本，运行方式：
 * npx tsx src/components/layout/timeline/gridLineKey.test.ts）。
 *
 * 覆盖两个层面：
 * - 拖动 Tempo Map 的“中间”变化点时，旧实现的抽样校验和会被
 *   不变的长度/首尾几条线欺骗，完整内容键必须识别中部的线移动；
 * - `explicitGridLinesKey` 的基础语义：null / 空数组 / 相同数组 /
 *   任意单线移动 / 长度变化。
 */
import type { TempoMap } from "../../../utils/tempoMap.ts";
import { buildTempoGridLineXsForViewport } from "../../../utils/tempoMap.ts";
import { explicitGridLinesKey } from "./gridLineKey.ts";

let checks = 0;

function assertTrue(cond: boolean, label: string): void {
    checks += 1;
    if (!cond) {
        throw new Error(`assertion failed: ${label}`);
    }
}

function assertEqualString(actual: string, expected: string, label: string): void {
    checks += 1;
    if (actual !== expected) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

// 复刻旧实现的“抽样校验和”（长度 + 前 3 条 + 后 3 条），
// 用于证明它在回归场景中会误判“内容未变化”。
function legacySampledChecksum(xs: number[] | null | undefined): string {
    if (!xs) return "-";
    if (xs.length === 0) return "0";
    const head = xs.slice(0, 3).join(",");
    const tail = xs.slice(-3).join(",");
    return `${xs.length}|${head}|${tail}`;
}

// ────────────────────────────────────────────────────────────────────────────
// 回归场景：小水平缩放下拖动 Tempo Map 的“中间”变化点。
//
// 被拖动点所在段的网格线以该点为锚、整体平移；此时数组长度与首尾
// 几条线（属于视口边缘、锚点未受拖动的其它段）都不变，只有数组中部
// 的线移动。旧的抽样校验和会认为“无需重绘”，导致网格线只在拖动到
// “有线进出视口 / 条数变化”时才跳变刷新一次。
// ────────────────────────────────────────────────────────────────────────────

function makeMap(middleSec: number): TempoMap {
    return {
        points: [
            {
                id: "a",
                positionSec: 0,
                bpm: 120,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
            },
            {
                id: "b",
                positionSec: middleSec,
                bpm: 90,
                timeSignature: null,
                scale: null,
            },
            {
                id: "c",
                positionSec: 120,
                bpm: 120,
                timeSignature: null,
                scale: null,
            },
            {
                id: "d",
                positionSec: 180,
                bpm: 150,
                timeSignature: null,
                scale: null,
            },
        ],
    };
}

function linesFor(map: TempoMap) {
    return buildTempoGridLineXsForViewport({
        tempoMap: map,
        scrollLeft: 0,
        viewportWidth: 800,
        pxPerSec: 0.5, // 小水平缩放：1px ≈ 2 秒
        projectSec: 180,
        stepBeats: 1,
        fallbackBpm: 120,
        fallbackBeatsPerBar: 4,
    });
}

{
    // 拖动中间变化点 0.6s（≈1.2px @ pxPerSec=0.5）：中部的线明显移动，
    // 但没有任何线进出视口 —— 长度、前 3 条、后 3 条全部不变。
    const before = linesFor(makeMap(60.4));
    const after = linesFor(makeMap(61.0));
    assertTrue(before != null && after != null, "viewport line sets exist");

    for (const kind of ["weak", "strong"] as const) {
        const a = before![kind];
        const b = after![kind];

        // 前提条件：长度、前 3 条、后 3 条均不变 —— 即旧抽样校验和认为“相同”。
        assertEqualString(
            legacySampledChecksum(a),
            legacySampledChecksum(b),
            `${kind}: legacy sampled checksum is fooled by middle shift`,
        );

        // 完整内容键必须能识别出“中部的线移动了”。
        assertTrue(
            explicitGridLinesKey(a) !== explicitGridLinesKey(b),
            `${kind}: complete key detects the middle line shift`,
        );
    }
}

// ── 键的基础语义 ────────────────────────────────────────────────────────────

assertEqualString(explicitGridLinesKey(null), "-", "null key");
assertEqualString(explicitGridLinesKey(undefined), "-", "undefined key");
assertEqualString(explicitGridLinesKey([]), "0", "empty key");
assertEqualString(
    explicitGridLinesKey([10, 20.5, 30]),
    explicitGridLinesKey([10, 20.5, 30]),
    "identical arrays share a key",
);
assertTrue(
    explicitGridLinesKey([10, 20.5, 30]) !== explicitGridLinesKey([10, 20.6, 30]),
    "any single line movement changes the key",
);
assertTrue(
    explicitGridLinesKey([1, 2, 3, 4]) !== explicitGridLinesKey([1, 2, 4]),
    "length change changes the key",
);

console.log(`gridLineKey checks passed (${checks})`);
