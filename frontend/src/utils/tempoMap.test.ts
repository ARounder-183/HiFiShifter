/**
 * Tempo Map 工具模块单元测试（自执行断言脚本，运行方式：npx tsx src/utils/tempoMap.test.ts）。
 */
import {
    barBeatAtSec,
    beatToSec,
    buildScaleSegments,
    buildTempoGridLines,
    createTempoPointAt,
    effectiveScaleAtSec,
    fromBackendTempoMap,
    normalizeTempoMap,
    pointIndexAtSec,
    removeTempoPoint,
    secToBeat,
    snapSecToTempoGrid,
    tempoAtSec,
    toBackendTempoMap,
    updateTempoPoint,
} from "./tempoMap";
import type { TempoMap, TempoPoint } from "./tempoMap";

let checks = 0;

function assertEqual(actual: unknown, expected: unknown, label: string): void {
    checks += 1;
    if (actual !== expected) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

function assertNear(actual: number, expected: number, label: string): void {
    checks += 1;
    if (Math.abs(actual - expected) > 1e-6) {
        throw new Error(`${label}: expected ${expected}, received ${actual}`);
    }
}

type RawPoint = Partial<Omit<TempoPoint, "id">> & { positionSec: number };

function mapWith(points: RawPoint[]): TempoMap {
    return {
        points: points.map((p, i) => ({
            id: `p${i}`,
            bpm: 120,
            numerator: 4,
            denominator: 4,
            scale: null,
            ...p,
        })),
    };
}

// ── 基础转换 ──────────────────────────────────────────────────────────────────

{
    const map = mapWith([{ positionSec: 0, bpm: 120 }]);
    for (const sec of [0, 1, 2.5, 30]) {
        assertNear(beatToSec(map, secToBeat(map, sec, 120), 120), sec, "sec↔beat round-trip");
    }
}

{
    // Reaper_Example.rpp：169.6 BPM @ 0s，184 BPM @ 0.353773584906s（恰好 1 拍后）。
    const map = mapWith([
        { positionSec: 0, bpm: 169.6 },
        { positionSec: 0.353773584906, bpm: 184 },
    ]);
    assertNear(secToBeat(map, 0.353773584906, 120), 1, "beat integration at first point");
    const secondBeatSec = 0.353773584906 + 60 / 184;
    assertNear(secToBeat(map, secondBeatSec, 120), 2, "beat integration after tempo change");
    assertNear(beatToSec(map, 2, 120), secondBeatSec, "beat→sec after tempo change");
}

{
    const map = mapWith([
        { positionSec: 0, bpm: 100 },
        { positionSec: 5, bpm: 200 },
    ]);
    assertEqual(tempoAtSec(map, 2, { bpm: 120, beatsPerBar: 4 }).bpm, 100, "tempo at sec 2");
    assertEqual(tempoAtSec(map, 5, { bpm: 120, beatsPerBar: 4 }).bpm, 200, "tempo at sec 5");
    assertEqual(pointIndexAtSec(map, 4.999), 0, "point index before change");
    assertEqual(pointIndexAtSec(map, 5.001), 1, "point index after change");
}

{
    // 0s 起 4/4 @120（每拍 0.5s），2s（=4 拍）处切 3/4。
    const map = mapWith([
        { positionSec: 0, bpm: 120, numerator: 4 },
        { positionSec: 2, bpm: 120, numerator: 3 },
    ]);
    const at2 = barBeatAtSec(map, 2, 120, 4);
    assertEqual(at2.bar, 2, "bar number at meter change");
    assertEqual(at2.beat, 1, "beat resets at meter change");
    const at25 = barBeatAtSec(map, 2.5, 120, 4);
    assertEqual(at25.bar, 2, "bar number in 3/4 segment");
    assertEqual(at25.beat, 2, "beat number in 3/4 segment");
}

{
    const map = mapWith([
        { positionSec: 0, bpm: 120 },
        { positionSec: 1, bpm: 240 }, // 1s = 2 拍；此后每拍 0.25s
    ]);
    const snapped = snapSecToTempoGrid(1.24, map, 1, 120);
    assertNear(snapped, 1.25, "snap across tempo change");
}

// ── 编辑辅助 ──────────────────────────────────────────────────────────────────

{
    const raw: TempoMap = {
        points: [
            { id: "b", positionSec: 5, bpm: 9999, numerator: 99, denominator: 7, scale: null },
            { id: "a", positionSec: 2, bpm: 140, numerator: 3, denominator: 8, scale: null },
        ],
    };
    const normalized = normalizeTempoMap(raw, 120, 4);
    if (!normalized) throw new Error("normalizeTempoMap returned null");
    assertEqual(normalized.points[0].positionSec, 0, "normalize: point 0 inserted");
    assertEqual(normalized.points[0].bpm, 120, "normalize: fallback bpm");
    assertEqual(
        normalized.points.map((p) => p.positionSec).join(","),
        "0,2,5",
        "normalize: sorted",
    );
    assertEqual(normalized.points[1].denominator, 8, "normalize: denominator kept");
    assertEqual(normalized.points[2].bpm, 960, "normalize: bpm clamped");
    assertEqual(normalized.points[2].denominator, 4, "normalize: invalid denominator");
}

{
    const { map, point } = createTempoPointAt(null, 3, { bpm: 150, beatsPerBar: 3 });
    assertEqual(map.points.length, 2, "create: two points");
    assertEqual(map.points[0].positionSec, 0, "create: first at 0");
    assertEqual(map.points[0].numerator, 3, "create: fallback beats");
    assertEqual(point.positionSec, 3, "create: point position");
    assertEqual(point.bpm, 150, "create: inherited bpm");
}

{
    const map = mapWith([
        { positionSec: 0, bpm: 100 },
        { positionSec: 4, bpm: 200 },
    ]);
    const after = removeTempoPoint(map, "p0");
    if (!after) throw new Error("removeTempoPoint returned null");
    assertEqual(after.points.length, 1, "remove: one point left");
    assertEqual(after.points[0].positionSec, 0, "remove: next point pinned to 0");
    assertEqual(after.points[0].bpm, 200, "remove: next point value kept");
    assertEqual(removeTempoPoint(after, "p1"), null, "remove: last point clears map");
}

{
    const map = mapWith([{ positionSec: 0, bpm: 120 }]);
    const updated = updateTempoPoint(map, "p0", { bpm: 5, numerator: 100 });
    assertEqual(updated.points[0].bpm, 10, "update: bpm clamped");
    assertEqual(updated.points[0].numerator, 32, "update: numerator clamped");
}

// ── 音阶 ──────────────────────────────────────────────────────────────────────

{
    const map = mapWith([
        { positionSec: 0, scale: null },
        { positionSec: 4, scale: { key: "G" } },
    ]);
    assertEqual(effectiveScaleAtSec(map, 2, "C"), "C", "scale before change = project");
    assertEqual(effectiveScaleAtSec(map, 4, "C"), "G", "scale at change point");
    assertEqual(effectiveScaleAtSec(map, 9, "C"), "G", "scale inherited forward");
}

{
    const map = mapWith([
        { positionSec: 0, scale: null },
        { positionSec: 2, scale: { key: "G" } },
        { positionSec: 5, scale: null },
    ]);
    const segments = buildScaleSegments(map, "C", 0, 10);
    if (!segments) throw new Error("buildScaleSegments returned null");
    assertEqual(segments.length, 2, "scale segments count");
    assertNear(segments[0].startSec, 0, "scale segment 0 start");
    assertNear(segments[0].endSec, 2, "scale segment 0 end");
    assertEqual(segments[0].scale, "C", "scale segment 0 scale");
    assertNear(segments[1].startSec, 2, "scale segment 1 start");
    assertNear(segments[1].endSec, 10, "scale segment 1 end");
    assertEqual(segments[1].scale, "G", "scale segment 1 scale");
}

// ── 网格线 ────────────────────────────────────────────────────────────────────

{
    const map = mapWith([{ positionSec: 0, bpm: 120, numerator: 4 }]);
    const lines = buildTempoGridLines({
        startSec: 0,
        endSec: 2,
        map,
        stepBeats: 1,
        fallbackBpm: 120,
        fallbackBeatsPerBar: 4,
    });
    const barLines = lines.filter((l) => l.isBar).map((l) => l.sec);
    if (!barLines.includes(0)) throw new Error("grid: missing bar line at 0");
    if (!barLines.some((s) => Math.abs(s - 2) < 1e-9)) {
        throw new Error("grid: missing bar line at 2s");
    }
    const weak = lines.filter((l) => !l.isBar).map((l) => l.sec);
    if (!weak.some((s) => Math.abs(s - 0.5) < 1e-9)) {
        throw new Error("grid: missing weak line at beat 2");
    }
}

{
    // 空工程 120bpm 4/4（每小节 2s），在 7.25s（旧标尺 4.3.500）插入变化点（同 BPM）。
    // 变化点后网格必须以该点为原点重新对齐 —— 不允许出现旧的 7.5/8.0/8.5 位置。
    const map = mapWith([
        { positionSec: 0, bpm: 120, numerator: 4 },
        { positionSec: 7.25, bpm: 120, numerator: 4 },
    ]);
    const lines = buildTempoGridLines({
        startSec: 7,
        endSec: 9.5,
        map,
        stepBeats: 1,
        fallbackBpm: 120,
        fallbackBeatsPerBar: 4,
    });
    const secs = lines.map((l) => l.sec);
    for (const bad of [7.5, 8.0, 8.5]) {
        if (secs.some((s) => Math.abs(s - bad) < 1e-9)) {
            throw new Error(`grid: stale old-grid line at ${bad}`);
        }
    }
    for (const good of [7.25, 7.75, 8.25, 9.25]) {
        if (!secs.some((s) => Math.abs(s - good) < 1e-9)) {
            throw new Error(`grid: missing segment-local line at ${good}`);
        }
    }
    // 段起点与小节线标记。
    const at725Bar = lines.some((l) => Math.abs(l.sec - 7.25) < 1e-9 && l.isBar);
    assertEqual(at725Bar, true, "grid: segment start is bar line");
    const at925 = lines.find((l) => Math.abs(l.sec - 9.25) < 1e-9 && l.isBar);
    assertEqual(at925?.isBar, true, "grid: segment-local bar line");
    const at775 = lines.find((l) => Math.abs(l.sec - 7.75) < 1e-9 && !l.isBar);
    assertEqual(at775?.isBar, false, "grid: segment-local weak line");
}

{
    // 段内吸附：第二段（1s 起 240bpm，每拍 0.25s）按段原点吸附。
    const map = mapWith([
        { positionSec: 0, bpm: 120 },
        { positionSec: 1, bpm: 240 },
    ]);
    assertNear(snapSecToTempoGrid(1.1, map, 1, 120), 1.0, "snap: to segment start bar");
    assertNear(snapSecToTempoGrid(1.14, map, 1, 120), 1.25, "snap: segment-local grid");
    assertNear(snapSecToTempoGrid(0.4, map, 1, 120), 0.5, "snap: first segment grid");
}

// ── 后端载荷序列化契约（裸数组，两端必须一致）───────────────────────────────

{
    const map = mapWith([
        { positionSec: 0, bpm: 130, numerator: 4, denominator: 4, scale: null },
        { positionSec: 2.5, bpm: 90, numerator: 3, denominator: 8, scale: { key: "G" } },
    ]);

    // 前端 → 后端：必须是“裸数组”（后端 set_timeline_tempo_map 参数为 Vec<TempoPointPayload>）。
    const payload = toBackendTempoMap(map);
    if (!Array.isArray(payload)) throw new Error("toBackendTempoMap: expected bare array");
    assertEqual(payload.length, 2, "wire: point count");
    assertEqual(payload[0].positionSec, 0, "wire: camelCase positionSec");
    assertEqual((payload[0] as { points?: unknown }).points, undefined, "wire: no points wrapper");
    assertEqual(payload[1].scale?.key, "G", "wire: scale key");

    // 后端 → 前端：裸数组必须能解析（此前误读 .points 导致永远为 null —— Tempo Map 行不显示）。
    const parsed = fromBackendTempoMap(payload, 120, 4);
    if (!parsed) throw new Error("fromBackendTempoMap: bare array must parse");
    assertEqual(parsed.points.length, 2, "wire: parsed point count");
    assertNear(parsed.points[1].positionSec, 2.5, "wire: parsed positionSec");
    assertEqual(parsed.points[1].bpm, 90, "wire: parsed bpm");
    assertEqual(parsed.points[1].denominator, 8, "wire: parsed denominator");
    assertEqual(parsed.points[1].scale?.key, "G", "wire: parsed scale key");

    // 无 Tempo Map：null 往返。
    assertEqual(toBackendTempoMap(null), null, "wire: null round-trip (send)");
    assertEqual(fromBackendTempoMap(null, 120, 4), null, "wire: null round-trip (parse)");

    // 兼容性：旧 { points } 包装形状仍可解析。
    const wrapped = { points: payload };
    const wrappedParsed = fromBackendTempoMap(wrapped, 120, 4);
    if (!wrappedParsed) throw new Error("fromBackendTempoMap: wrapped shape must parse");
    assertEqual(wrappedParsed.points.length, 2, "wire: wrapped shape parsed");
}

{
    // 模拟完整后端快照（TimelineStatePayload.tempo_map 的 JSON 形状）→ 解析 → 重序列化，应稳定。
    const backendSnapshot = [
        {
            id: "tp0",
            positionSec: 0,
            bpm: 100,
            numerator: 4,
            denominator: 4,
            scale: null,
        },
        {
            id: "tp1",
            positionSec: 4,
            bpm: 100,
            numerator: 6,
            denominator: 8,
            scale: { key: null, name: "custom", notes: [0, 2, 4, 5, 7, 9, 11] },
        },
    ];
    const parsed = fromBackendTempoMap(backendSnapshot, 120, 4);
    if (!parsed) throw new Error("fromBackendTempoMap: snapshot parse failed");
    assertEqual(parsed.points[1].numerator, 6, "wire: numerator from snapshot");
    assertEqual(parsed.points[1].scale?.name, "custom", "wire: custom scale name");
    assertEqual(
        JSON.stringify(parsed.points[1].scale?.notes),
        "[0,2,4,5,7,9,11]",
        "wire: custom scale notes",
    );

    // 编辑后再发送回后端：保持裸数组 + camelCase，且内容与快照一致。
    const reserialized = toBackendTempoMap(parsed);
    assertEqual(
        JSON.stringify(reserialized),
        JSON.stringify(backendSnapshot),
        "wire: full round-trip stable",
    );
}

console.log(`tempoMap.test.ts: all ${checks} checks passed.`);
