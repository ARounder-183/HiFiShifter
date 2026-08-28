import { test } from "vitest";

import {
    barBeatAtSec,
    beatToSec,
    buildScaleSegments,
    buildTempoGridLineXsForViewport,
    buildTempoGridLines,
    computeTempoFloatingLabelState,
    createTempoPointAt,
    effectiveScaleAtSec,
    effectiveTimeSignatureAt,
    fromBackendTempoMap,
    insertTempoPoint,
    normalizeTempoMap,
    parseTempoPointText,
    pointIndexAtSec,
    removeTempoPoint,
    scaleChangesInRange,
    secToBeat,
    snapSecToTempoGrid,
    tempoAtSec,
    tempoFlagLabelWidthPx,
    tempoPointFlagLabel,
    tempoPointHitTest,
    toBackendTempoMap,
    updateTempoPoint,
} from "./tempoMap";
import type { TempoMap, TempoPoint, TempoTimeSignature } from "./tempoMap";

test("utils/tempoMap.test.ts scripted checks", async () => {
    /**
     * Tempo Map 工具模块单元测试（自执行断言脚本，运行方式：npx tsx src/utils/tempoMap.test.ts）。
     */

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

    /** 构造 Tempo Map：未指定拍号时显式 4/4（0 位置初始点必须显式）。 */
    function mapWith(points: RawPoint[]): TempoMap {
        return {
            points: points.map((p, i) => ({
                id: `p${i}`,
                bpm: 120,
                timeSignature: { numerator: 4, denominator: 4 },
                scale: null,
                ...p,
            })),
        };
    }

    function sig(numerator: number, denominator: number): TempoTimeSignature {
        return { numerator, denominator };
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
            { positionSec: 0, bpm: 120, timeSignature: sig(4, 4) },
            { positionSec: 2, bpm: 120, timeSignature: sig(3, 4) },
        ]);
        const at2 = barBeatAtSec(map, 2, 120, 4);
        assertEqual(at2.bar, 2, "bar number at meter change");
        assertEqual(at2.beat, 1, "beat resets at meter change");
        const at25 = barBeatAtSec(map, 2.5, 120, 4);
        assertEqual(at25.bar, 2, "bar number in 3/4 segment");
        assertEqual(at25.beat, 2, "beat number in 3/4 segment");
    }

    {
        // 浮点边界：1.999999999999 拍（≈ 第 3 拍起点）→ 应进位为 1.3，而不是 "1.2.1000"。
        const map = mapWith([{ positionSec: 0, bpm: 120, timeSignature: sig(4, 4) }]);
        const bbt = barBeatAtSec(map, 0.9999999999995, 120, 4);
        assertEqual(bbt.bar, 1, "barBeat: near-boundary bar");
        assertEqual(bbt.beat, 3, "barBeat: near-boundary beat carries");
        assertNear(bbt.sub, 0, "barBeat: near-boundary sub zero");
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
                {
                    id: "b",
                    positionSec: 5,
                    bpm: 9999,
                    timeSignature: { numerator: 99, denominator: 7 },
                    scale: null,
                },
                {
                    id: "a",
                    positionSec: 2,
                    bpm: 140,
                    timeSignature: { numerator: 3, denominator: 8 },
                    scale: null,
                },
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
        assertEqual(
            normalized.points[1].timeSignature?.denominator,
            8,
            "normalize: denominator kept",
        );
        assertEqual(normalized.points[2].bpm, 960, "normalize: bpm clamped");
        assertEqual(
            normalized.points[2].timeSignature?.numerator,
            32,
            "normalize: numerator clamped",
        );
        assertEqual(
            normalized.points[2].timeSignature?.denominator,
            4,
            "normalize: invalid denominator",
        );
    }

    {
        const { map, point } = createTempoPointAt(null, 3, {
            bpm: 150,
            beatsPerBar: 3,
            denominator: 8,
        });
        assertEqual(map.points.length, 2, "create: two points");
        assertEqual(map.points[0].positionSec, 0, "create: first at 0");
        assertEqual(map.points[0].timeSignature?.numerator, 3, "create: fallback beats numerator");
        assertEqual(map.points[0].timeSignature?.denominator, 8, "create: fallback denominator");
        assertEqual(point.positionSec, 3, "create: point position");
        assertEqual(point.bpm, 150, "create: inherited bpm");
        // 新添加的变化点默认“跟随之前的拍号”。
        assertEqual(point.timeSignature, null, "create: new point follows previous time signature");
        assertEqual(point.scale, null, "create: new point follows project scale");
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
        // 删除 0 位置点后，被钉到 0 的“跟随拍号”点必须物化为显式拍号。
        const map = mapWith([
            { positionSec: 0, bpm: 120, timeSignature: sig(3, 4) },
            { positionSec: 4, bpm: 120, timeSignature: null },
        ]);
        const after = removeTempoPoint(map, "p0");
        if (!after) throw new Error("removeTempoPoint (follow) returned null");
        assertEqual(
            after.points[0].timeSignature?.numerator,
            3,
            "remove: pinned point materializes sig",
        );
        assertEqual(after.points[0].timeSignature?.denominator, 4, "remove: pinned denominator");
    }

    {
        const map = mapWith([{ positionSec: 0, bpm: 120 }]);
        const updated = updateTempoPoint(map, "p0", {
            bpm: 5,
            timeSignature: { numerator: 100, denominator: 3 },
        });
        assertEqual(updated.points[0].bpm, 10, "update: bpm clamped");
        assertEqual(updated.points[0].timeSignature?.numerator, 32, "update: numerator clamped");
        assertEqual(updated.points[0].timeSignature?.denominator, 4, "update: denominator clamped");
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

    // ── 拍号“跟随之前的拍号” ─────────────────────────────────────────────────────

    {
        // 0 位置 3/4；5s 处拍号跟随（继续 3/4）；10s 处切 6/8。
        const map = mapWith([
            { positionSec: 0, bpm: 120, timeSignature: sig(3, 4) },
            { positionSec: 5, bpm: 120, timeSignature: null },
            { positionSec: 10, bpm: 120, timeSignature: sig(6, 8) },
        ]);
        const eff = effectiveTimeSignatureAt(map, 1);
        assertEqual(eff.numerator, 3, "effective sig: follow resolves numerator");
        assertEqual(eff.denominator, 4, "effective sig: follow resolves denominator");
        assertEqual(
            tempoAtSec(map, 5, { bpm: 120, beatsPerBar: 4 }).numerator,
            3,
            "tempoAtSec: effective numerator at following point",
        );
        assertEqual(
            tempoAtSec(map, 7.5, { bpm: 120, beatsPerBar: 4 }).denominator,
            4,
            "tempoAtSec: effective denominator in following segment",
        );
        assertEqual(
            tempoAtSec(map, 12, { bpm: 120, beatsPerBar: 4 }).denominator,
            8,
            "tempoAtSec: explicit denominator after change",
        );
    }

    {
        // 跟随拍号的小节对齐：0s 3/4（每小节 1.5s @120），3s 处变化点跟随 → 小节按 3/4 连续。
        const map = mapWith([
            { positionSec: 0, bpm: 120, timeSignature: sig(3, 4) },
            { positionSec: 3, bpm: 120, timeSignature: null },
        ]);
        const at3 = barBeatAtSec(map, 3, 120, 4);
        assertEqual(at3.bar, 3, "follow sig: bar continues across following point");
        assertEqual(at3.beat, 1, "follow sig: segment start is bar start");
        const at45 = barBeatAtSec(map, 4.5, 120, 4);
        assertEqual(at45.bar, 4, "follow sig: bar number in following segment");
    }

    {
        // 网格线：跟随拍号的段继续按 3/4 生成小节线（每 1.5s）。
        const map = mapWith([
            { positionSec: 0, bpm: 120, timeSignature: sig(3, 4) },
            { positionSec: 3, bpm: 120, timeSignature: null },
        ]);
        const lines = buildTempoGridLines({
            startSec: 3,
            endSec: 6,
            map,
            stepBeats: 1,
            fallbackBpm: 120,
            fallbackBeatsPerBar: 4,
        });
        const barLines = lines.filter((l) => l.isBar).map((l) => l.sec);
        if (!barLines.some((s) => Math.abs(s - 3) < 1e-9)) {
            throw new Error("follow sig: missing bar line at segment start");
        }
        if (!barLines.some((s) => Math.abs(s - 4.5) < 1e-9)) {
            throw new Error("follow sig: missing 3/4 bar line at 4.5s");
        }
        if (!barLines.some((s) => Math.abs(s - 6) < 1e-9)) {
            throw new Error("follow sig: missing 3/4 bar line at 6s");
        }
    }

    // ── 网格线 ────────────────────────────────────────────────────────────────────

    {
        const map = mapWith([{ positionSec: 0, bpm: 120, timeSignature: sig(4, 4) }]);
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
            { positionSec: 0, bpm: 120, timeSignature: sig(4, 4) },
            { positionSec: 7.25, bpm: 120, timeSignature: sig(4, 4) },
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

    // ── 悬浮标签状态（viewport 左侧黏性参数标签）─────────────────────────────

    {
        const map = mapWith([
            {
                positionSec: 0,
                bpm: 120,
                timeSignature: sig(4, 4),
                scale: { key: "C" },
            },
            {
                positionSec: 7.25,
                bpm: 150,
                timeSignature: sig(3, 4),
                scale: { key: "G" },
            },
        ]);
        const pxPerSec = 100;

        // 滚动 0：首点旗帜可见 → 无需悬浮标签。
        let st = computeTempoFloatingLabelState({ tempoMap: map, scrollLeft: 0, pxPerSec });
        assertEqual(st.governingOffscreen, false, "float: hidden when flag visible");

        // 首点旗帜完全滚出画面左侧 → 显示悬浮标签（120 段）。
        st = computeTempoFloatingLabelState({ tempoMap: map, scrollLeft: 120, pxPerSec });
        assertEqual(st.governingOffscreen, true, "float: shown when flag offscreen");
        assertEqual(st.blocked, false, "float: not blocked far from next flag");
        assertEqual(st.label, "120 4/4 - C / Am", "float: first segment label");

        // 下一旗帜（7.25s = 725px）进入画面左侧 → 与悬浮标签区域重叠 → 隐藏。
        st = computeTempoFloatingLabelState({ tempoMap: map, scrollLeft: 722, pxPerSec });
        assertEqual(st.blocked, true, "float: blocked while next flag overlaps");

        // 下一旗帜越过左边缘但仍在标签区内 → 依然隐藏。
        st = computeTempoFloatingLabelState({ tempoMap: map, scrollLeft: 727, pxPerSec });
        assertEqual(st.blocked, true, "float: blocked while flag inside chip area");

        // 下一旗帜完全滚出画面左侧 → 显示新段标签（150 段）。
        st = computeTempoFloatingLabelState({ tempoMap: map, scrollLeft: 850, pxPerSec });
        assertEqual(st.blocked, false, "float: unblocked after flag exits");
        assertEqual(st.label, "150 3/4 - G / Em", "float: second segment label");
    }

    // ── 旗帜文本（展示与解析必须完全一致）───────────────────────────────────────

    {
        assertEqual(
            tempoPointFlagLabel({
                id: "x",
                positionSec: 0,
                bpm: 120,
                timeSignature: null,
                scale: null,
            }),
            "120",
            "label: sig+scale follow → bpm only",
        );
        assertEqual(
            tempoPointFlagLabel({
                id: "x",
                positionSec: 0,
                bpm: 120,
                timeSignature: sig(4, 4),
                scale: null,
            }),
            "120 4/4",
            "label: explicit sig, scale follows",
        );
        assertEqual(
            tempoPointFlagLabel({
                id: "x",
                positionSec: 0,
                bpm: 120,
                timeSignature: null,
                scale: { key: "C" },
            }),
            "120 - C / Am",
            "label: sig follows, explicit scale",
        );
        assertEqual(
            tempoPointFlagLabel({
                id: "x",
                positionSec: 0,
                bpm: 120,
                timeSignature: sig(4, 4),
                scale: { key: "C" },
            }),
            "120 4/4 - C / Am",
            "label: both explicit",
        );
    }

    // ── 旗帜文本解析（内联编辑）─────────────────────────────────────────────────

    {
        const presets = [{ id: "c1", name: "自定义音阶", notes: [0, 2, 4, 5, 7, 9, 11] }];

        const full = parseTempoPointText("120 4/4 - C / Am", presets);
        assertEqual(full?.bpm, 120, "parse: bpm");
        assertEqual(full?.timeSignature?.numerator, 4, "parse: numerator");
        assertEqual(full?.timeSignature?.denominator, 4, "parse: denominator");
        assertEqual(full?.scale?.key, "C", "parse: scale by label");

        const sigOnly = parseTempoPointText("150.5 3/8", presets);
        assertEqual(sigOnly?.bpm, 150.5, "parse: decimal bpm");
        assertEqual(sigOnly?.timeSignature?.denominator, 8, "parse: denominator 8");
        assertEqual(sigOnly?.scale, null, "parse: no scale = inherit");

        const bpmOnly = parseTempoPointText("120", presets);
        assertEqual(bpmOnly?.bpm, 120, "parse: bpm only");
        assertEqual(bpmOnly?.timeSignature, null, "parse: no sig = follow previous");
        assertEqual(bpmOnly?.scale, null, "parse: no scale = inherit");

        const scaleOnly = parseTempoPointText("120 - C / Am", presets);
        assertEqual(scaleOnly?.bpm, 120, "parse: scale-only bpm");
        assertEqual(scaleOnly?.timeSignature, null, "parse: scale-only follows sig");
        assertEqual(scaleOnly?.scale?.key, "C", "parse: scale-only scale");

        const bareKey = parseTempoPointText("90 4/4 - G", presets);
        assertEqual(bareKey?.scale?.key, "G", "parse: bare key");

        const custom = parseTempoPointText("90 4/4 - 自定义音阶", presets);
        assertEqual(custom?.scale?.name, "自定义音阶", "parse: custom preset name");
        assertEqual(
            JSON.stringify(custom?.scale?.notes),
            "[0,2,4,5,7,9,11]",
            "parse: custom preset notes",
        );

        const customScaleOnly = parseTempoPointText("90 - 自定义音阶", presets);
        assertEqual(customScaleOnly?.timeSignature, null, "parse: custom scale-only follows sig");
        assertEqual(customScaleOnly?.scale?.name, "自定义音阶", "parse: custom scale-only name");

        assertEqual(parseTempoPointText("abc", presets), null, "parse: garbage fails");
        assertEqual(parseTempoPointText("", presets), null, "parse: empty fails");
        assertEqual(
            parseTempoPointText("120 5/3", presets),
            null,
            "parse: invalid denominator fails",
        );
        assertEqual(
            parseTempoPointText("120 4/4 - X / Y", presets),
            null,
            "parse: unknown scale fails",
        );
        assertEqual(parseTempoPointText("99999 4/4", presets)?.bpm, 960, "parse: bpm clamped");
        assertEqual(
            parseTempoPointText("120 4/4 leftover", presets),
            null,
            "parse: trailing garbage fails",
        );
    }

    // ── 旗帜命中检测（右键菜单）─────────────────────────────────────────────────

    {
        const map = mapWith([
            { positionSec: 0, bpm: 120, timeSignature: sig(4, 4), scale: { key: "C" } },
            { positionSec: 7.25, bpm: 150, timeSignature: sig(3, 4), scale: { key: "G" } },
        ]);
        const pxPerSec = 100;
        const secondLabel = tempoPointFlagLabel(map.points[1]);
        // 旗帜文本向右延伸：点击文本中部（位置 + 文本宽度的一半）也必须命中。
        const textHalfPx = tempoFlagLabelWidthPx(secondLabel) / 2;
        const clickPx = map.points[1].positionSec * pxPerSec + textHalfPx;
        assertEqual(tempoPointHitTest(map, clickPx / pxPerSec, pxPerSec), 1, "hit: flag text body");
        // 点击位置本体（旗帜线）命中。
        assertEqual(
            tempoPointHitTest(map, map.points[0].positionSec, pxPerSec),
            0,
            "hit: flag line",
        );
        // 远离旗帜 → 不命中。
        assertEqual(tempoPointHitTest(map, 3.0, pxPerSec), null, "hit: empty area misses");
    }

    // ── 后端载荷序列化契约（裸数组，两端必须一致）───────────────────────────────

    {
        const map = mapWith([
            { positionSec: 0, bpm: 130, timeSignature: sig(4, 4), scale: null },
            {
                positionSec: 2.5,
                bpm: 90,
                timeSignature: sig(3, 8),
                scale: { key: "G" },
            },
        ]);

        // 前端 → 后端：必须是“裸数组”（后端 set_timeline_tempo_map 参数为 Vec<TempoPointPayload>）。
        const payload = toBackendTempoMap(map);
        if (!Array.isArray(payload)) throw new Error("toBackendTempoMap: expected bare array");
        assertEqual(payload.length, 2, "wire: point count");
        assertEqual(payload[0].positionSec, 0, "wire: camelCase positionSec");
        assertEqual(
            (payload[0] as { points?: unknown }).points,
            undefined,
            "wire: no points wrapper",
        );
        assertEqual(payload[1].scale?.key, "G", "wire: scale key");

        // 后端 → 前端：裸数组必须能解析（此前误读 .points 导致永远为 null —— Tempo Map 行不显示）。
        const parsed = fromBackendTempoMap(payload, 120, 4);
        if (!parsed) throw new Error("fromBackendTempoMap: bare array must parse");
        assertEqual(parsed.points.length, 2, "wire: parsed point count");
        assertNear(parsed.points[1].positionSec, 2.5, "wire: parsed positionSec");
        assertEqual(parsed.points[1].bpm, 90, "wire: parsed bpm");
        assertEqual(parsed.points[1].timeSignature?.denominator, 8, "wire: parsed denominator");
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
            {
                id: "tp2",
                positionSec: 8,
                bpm: 100,
                numerator: null,
                denominator: null,
                scale: null,
            },
        ];
        const parsed = fromBackendTempoMap(backendSnapshot, 120, 4);
        if (!parsed) throw new Error("fromBackendTempoMap: snapshot parse failed");
        assertEqual(parsed.points[1].timeSignature?.numerator, 6, "wire: numerator from snapshot");
        assertEqual(parsed.points[1].scale?.name, "custom", "wire: custom scale name");
        assertEqual(
            JSON.stringify(parsed.points[1].scale?.notes),
            "[0,2,4,5,7,9,11]",
            "wire: custom scale notes",
        );
        // 跟随拍号的点：null → timeSignature null。
        assertEqual(parsed.points[2].timeSignature, null, "wire: null sig = follow previous");

        // 编辑后再发送回后端：保持裸数组 + camelCase，且内容与快照一致。
        const reserialized = toBackendTempoMap(parsed);
        assertEqual(
            JSON.stringify(reserialized),
            JSON.stringify(backendSnapshot),
            "wire: full round-trip stable",
        );
    }

    {
        // 初始点缺失拍号（非法数据）：规范化时用工程基准值物化，不允许“跟随”初始点。
        const raw: TempoMap = {
            points: [
                { id: "a", positionSec: 0, bpm: 120, timeSignature: null, scale: null },
                { id: "b", positionSec: 2, bpm: 120, timeSignature: null, scale: null },
            ],
        };
        const normalized = normalizeTempoMap(raw, 120, 7, { projectDenominator: 8 });
        if (!normalized) throw new Error("normalizeTempoMap (null sig) returned null");
        assertEqual(
            normalized.points[0].timeSignature?.numerator,
            7,
            "normalize: point-0 numerator",
        );
        assertEqual(
            normalized.points[0].timeSignature?.denominator,
            8,
            "normalize: point-0 project denominator",
        );
        assertEqual(
            normalized.points[1].timeSignature,
            null,
            "normalize: later point stays follow",
        );
    }

    // ── 回归：不变量修复 ───────────────────────────────────────────────────────────

    {
        // updateTempoPoint 修改 positionSec 后必须重排（拖拽跨越相邻点会乱序，
        // 破坏 pointIndexAtSec 的二分查找）并钉住 0 位置点。
        const map = mapWith([
            { positionSec: 0, bpm: 120 },
            { positionSec: 2, bpm: 130 },
            { positionSec: 5, bpm: 140 },
        ]);
        // 把 5s 的点拖到 1s（跨过 2s 的点）。
        const updated = updateTempoPoint(map, "p2", { positionSec: 1 });
        assertEqual(updated.points[0].positionSec, 0, "update: point-0 stays at 0");
        assertEqual(updated.points[1].positionSec, 1, "update: dragged point sorted first");
        assertEqual(updated.points[1].id, "p2", "update: dragged point keeps id");
        assertEqual(updated.points[2].positionSec, 2, "update: crossed point keeps position");
        assertEqual(pointIndexAtSec(updated, 1.5), 1, "update: binary search works after re-sort");
    }

    {
        // normalizeTempoMap：乱序输入 + 非相邻重复位置 → 排序后去重。
        const raw: TempoMap = {
            points: [
                { id: "a", positionSec: 5, bpm: 120, timeSignature: sig(4, 4), scale: null },
                { id: "b", positionSec: 0, bpm: 120, timeSignature: sig(4, 4), scale: null },
                { id: "c", positionSec: 5, bpm: 130, timeSignature: sig(4, 4), scale: null },
            ],
        };
        const normalized = normalizeTempoMap(raw, 120, 4);
        if (!normalized) throw new Error("normalizeTempoMap (unsorted dup) returned null");
        assertEqual(normalized.points.length, 2, "normalize: duplicate position collapsed");
        assertEqual(normalized.points[0].positionSec, 0, "normalize: sorted first");
        assertEqual(normalized.points[1].positionSec, 5, "normalize: sorted second");
    }

    {
        // insertTempoPoint：与已有点过近时不插入（契约：不允许重复位置）。
        const map = mapWith([
            { positionSec: 0, bpm: 120 },
            { positionSec: 5, bpm: 120 },
        ]);
        const close = insertTempoPoint(map, {
            id: "dup",
            positionSec: 5,
            bpm: 200,
            timeSignature: null,
            scale: null,
        });
        assertEqual(close.points.length, 2, "insert: too-close point rejected");
    }

    {
        // createTempoPointAt(null, 0)：初始点必须携带显式拍号/音阶（工程基准记录），
        // 而不是 timeSignature: null 的“跟随”点（序列化后后端会按 4/4 物化）。
        const { map, point } = createTempoPointAt(
            null,
            0,
            { bpm: 90, beatsPerBar: 3, denominator: 8 },
            {
                projectScale: "G",
            },
        );
        assertEqual(map.points.length, 1, "create at 0: single point");
        assertEqual(point.timeSignature?.numerator, 3, "create at 0: explicit numerator");
        assertEqual(point.timeSignature?.denominator, 8, "create at 0: explicit denominator");
        assertEqual(point.scale?.key, "G", "create at 0: project scale attached");
    }

    {
        // buildBeatCache / secToBeat：非法 BPM（0）不能产生 NaN/Infinity（防御）。
        const map = mapWith([{ positionSec: 0, bpm: 0 }]);
        const beat = secToBeat(map, 10, 120);
        if (!Number.isFinite(beat)) throw new Error("secToBeat with bpm 0 must be finite");
        const sec = beatToSec(map, beat, 120);
        if (!Number.isFinite(sec)) throw new Error("beatToSec with bpm 0 must be finite");
        assertNear(sec, 10, "sec↔beat round-trip with clamped bpm");
    }

    {
        // scaleChangesInRange：管辖范围起点之前的变化点必须包含（范围起点生效音阶
        // 由它决定），否则 MenuBar 的“选区受 Tempo Map 影响”提示会漏报。
        const map = mapWith([
            { positionSec: 0, scale: null },
            { positionSec: 1, scale: { key: "G" } },
            { positionSec: 4, scale: { key: "D" } },
        ]);
        const changes = scaleChangesInRange(map, 2, 3);
        assertEqual(changes.length, 1, "scaleChangesInRange: governing change included");
        assertEqual(changes[0].positionSec, 1, "scaleChangesInRange: governing position");
        assertEqual(changes[0].scale, "G", "scaleChangesInRange: governing scale");
        const none = scaleChangesInRange(map, 5, 6);
        assertEqual(none.length, 1, "scaleChangesInRange: later range still governed");
        assertEqual(none[0].scale, "D", "scaleChangesInRange: later governing scale");
    }

    {
        // 网格密度上限：细网格 + 长范围必须先估算步长，不能全量生成
        // （长工程 + 1/64 网格曾先分配数百万条线再过滤，导致界面卡死）。
        const map = mapWith([
            { positionSec: 0, bpm: 960, timeSignature: sig(4, 4) },
            { positionSec: 3600, bpm: 960, timeSignature: sig(4, 4) },
        ]);
        const xs = buildTempoGridLineXsForViewport({
            tempoMap: map,
            scrollLeft: 0,
            viewportWidth: 1200,
            pxPerSec: 0.1,
            projectSec: 7200,
            stepBeats: 1 / 64,
            fallbackBpm: 960,
            fallbackBeatsPerBar: 4,
        });
        if (!xs) throw new Error("buildTempoGridLineXsForViewport returned null");
        // 2 小时 @960BPM、1/64 网格：全量生成会产生数百万条线；
        // 修复后弱网格按步长预缩放、强网格按 stride 抽取，总量有界。
        if (xs.weak.length > 500) {
            throw new Error(`grid density: weak lines bounded (got ${xs.weak.length})`);
        }
        if (xs.strong.length > 52) {
            throw new Error(`grid density: strong lines bounded (got ${xs.strong.length})`);
        }
    }

});
