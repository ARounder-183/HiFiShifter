/**
 * 后端回声的**幂等**契约（滚轮调 BPM 时标尺抽搐的根因之一）。
 *
 * 【要钉死的性质】乐观写入已经把值放进了状态；后端回声只是"确认"，**不应**再产生
 * 一次状态变更 —— 订阅方（标尺刻度 / 网格 / 波形）会因此每滚一格重算两次。
 * 状态是否"变了"用**引用**判定：Redux 的订阅者拿到的是新对象引用，引用不变即没有
 * 任何重渲染。
 */
import { test } from "vitest";

import reducer from "./sessionSlice.js";
import { updateTransportBpm } from "./thunks/transportThunks.js";
import { setTempoMapRemote } from "./thunks/tempoMapThunks.js";
import type { TempoMap } from "../../utils/tempoMap.js";

test("features/session/sessionSlice.echo.test.ts scripted checks", async () => {
    function createState(): ReturnType<typeof reducer> {
        return reducer(undefined, { type: "@@INIT" });
    }

    function assertTrue(condition: boolean, label: string): void {
        if (!condition) throw new Error(`${label}: expected true`);
    }

    function assertEqual(actual: unknown, expected: unknown, label: string): void {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${String(expected)}, received ${String(actual)}`);
        }
    }

    const tempoMap: TempoMap = {
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
                positionSec: 8,
                bpm: 90,
                timeSignature: { numerator: 3, denominator: 4 },
                scale: null,
            },
        ],
    };

    // ── BPM 回声 ────────────────────────────────────────────────
    {
        const base = reducer(createState(), { type: "@@INIT" });
        const withBpm = reducer(base, { type: "session/setBpm", payload: 121 });
        assertEqual(withBpm.bpm, 121, "乐观写入生效");

        // 后端确认同一个值：不得产生新的状态引用。
        const echoed = reducer(
            withBpm,
            updateTransportBpm.fulfilled({ ok: true, bpm: 121 } as never, "req-1", 121),
        );
        assertTrue(echoed === withBpm, "同值回声不产生新状态（无第二次提交）");

        // 后端确实钳到另一个值：必须采纳。
        const corrected = reducer(
            withBpm,
            updateTransportBpm.fulfilled({ ok: true, bpm: 960 } as never, "req-2", 121),
        );
        assertTrue(corrected !== withBpm, "不同值回声必须采纳");
        assertEqual(corrected.bpm, 960, "采纳后端钳制后的值");
    }

    // ── BPM：迟到的回声不得把值拽回去 ──────────────────────────
    //
    // 【真实场景】传输命令串行执行，滚轮手势期间积压几十个请求；每个回声都在乐观值
    // 已经前进之后才到达。若照单采纳，state.bpm 会在"最新乐观值"与"旧回声值"之间
    // 反复被拽回，刻度时间（sec = beat×60/bpm）与网格随之来回平移 = 用户报告的
    // "标尺和网格抽搐"。判据与 seekPlayhead 一致：前端值仍等于本次请求值才采纳。
    {
        const base = reducer(createState(), { type: "@@INIT" });
        // 手势推进到 125（本地乐观值）。
        const advanced = reducer(base, { type: "session/setBpm", payload: 125 });
        assertEqual(advanced.bpm, 125, "乐观值已前进");

        // 一个"121"的迟到回声（它对应更早的那次请求）。
        const stale = reducer(
            advanced,
            updateTransportBpm.fulfilled({ ok: true, bpm: 121 } as never, "req-old", 121),
        );
        assertTrue(stale === advanced, "迟到回声不得产生任何状态变更（值被拽回）");
        assertEqual(stale.bpm, 125, "BPM 保持最新乐观值");

        // 同一请求若后端确实修正（clamp 到 960），仍要采纳。
        const clamped = reducer(
            reducer(base, { type: "session/setBpm", payload: 125 }),
            updateTransportBpm.fulfilled({ ok: true, bpm: 960 } as never, "req-new", 125),
        );
        assertEqual(clamped.bpm, 960, "本次请求被后端修正时采纳");
    }

    // ── Tempo Map：迟到的地图回声整份忽略 ──────────────────────
    {
        const base = reducer(createState(), { type: "@@INIT" });
        const mapA: TempoMap = {
            points: [
                { id: "a", positionSec: 0, bpm: 120, timeSignature: null, scale: null },
                { id: "b", positionSec: 4, bpm: 90, timeSignature: null, scale: null },
            ],
        };
        const mapB: TempoMap = {
            points: [
                { id: "a", positionSec: 0, bpm: 120, timeSignature: null, scale: null },
                { id: "b", positionSec: 4, bpm: 132, timeSignature: null, scale: null },
            ],
        };
        // 手势已推进到 mapB（本地乐观值）。
        const advanced = reducer(base, { type: "session/setTempoMap", payload: mapB });
        const stalePayload = {
            ok: true,
            tracks: [],
            clips: [],
            selected_track_id: null,
            selected_clip_id: null,
            bpm: 120,
            playhead_sec: 0,
            tempo_map: mapA.points.map((p) => ({
                id: p.id,
                positionSec: p.positionSec,
                bpm: p.bpm,
                numerator: null,
                denominator: null,
                scale: null,
            })),
        } as never;
        // 迟到回声携带的是 mapA：当前地图（mapB）不等于本次请求值 ⇒ 整份忽略。
        const stale = reducer(
            advanced,
            setTempoMapRemote.fulfilled(stalePayload, "req-old-map", mapA),
        );
        assertTrue(stale === advanced, "迟到的地图回声整份忽略（网格不被拽回旧锚点）");
    }

    // ── Tempo Map 回声 ─────────────────────────────────────────
    {
        const base = reducer(createState(), { type: "@@INIT" });
        const withMap = reducer(base, { type: "session/setTempoMap", payload: tempoMap });
        const previousMap = withMap.tempoMap;
        assertTrue(previousMap !== null, "地图已写入");

        // 后端回声带回语义相同（但对象全新、id 重新生成）的地图。
        const echoedPayload = {
            ok: true,
            tracks: [],
            clips: [],
            selected_track_id: null,
            selected_clip_id: null,
            bpm: 120,
            playhead_sec: 0,
            // 后端线格式：camelCase + 拍号拆成 numerator/denominator（见
            // `toBackendTempoMap` / `fromBackendTempoMap`），id 会重新生成。
            tempo_map: tempoMap.points.map((point, index) => ({
                id: `backend-${index}`,
                positionSec: point.positionSec,
                bpm: point.bpm,
                numerator: point.timeSignature?.numerator ?? null,
                denominator: point.timeSignature?.denominator ?? null,
                scale: null,
            })),
        } as never;
        const echoed = reducer(
            withMap,
            setTempoMapRemote.fulfilled(echoedPayload, "req-3", tempoMap),
        );
        assertTrue(
            echoed.tempoMap === previousMap,
            "语义相同的回声保留旧 tempoMap 引用（刻度 memo 不失效）",
        );
    }
});
