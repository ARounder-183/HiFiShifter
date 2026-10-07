/*
 * reducer 纯度回归测试。
 *
 * 【为什么必须有】卫星窗口（`features/dock/detachBridge.ts`）会把主窗口广播的
 * 动作在**本地 reducer 上重放**，以此保持两个窗口的状态收敛。这条机制的隐含
 * 前提是"同一个 action 对象 → 同一个新状态"。一旦 reducer 里出现随机数或时钟
 * 读取，主窗口与卫星窗口就会为同一个动作算出**不同的结果**：
 *
 *   - `addClip` 曾在 reducer 里 `Math.random()` 生成 clip id → 两个窗口的
 *     `session.clips` 里的 id 永久不同（快照只在卫星窗口启动时对齐一次）；
 *   - `saveDockPreset` 曾在 reducer 里 `Date.now()` 写 `createdAtMs` → 预设的
 *     时间戳两边不一致。
 *
 * 这两类偏差不会抛异常、不会有日志，症状只是"另一个窗口里的东西对不上"，
 * 极难归因。因此这里用"同一 action 重放两次必须得到同一状态"直接钉住。
 */
import { describe, expect, test } from "vitest";

import dockReducer, { saveDockPreset } from "../dock/dockSlice";
import sessionReducer, { addClip } from "./sessionSlice";

const sessionInitial = sessionReducer(undefined, { type: "@@init" });
const dockInitial = dockReducer(undefined, { type: "@@init" });

describe("session reducer 纯度", () => {
    test("addClip 生成的 clip id 由 action 携带，重放结果一致", () => {
        const trackId = sessionInitial.tracks[0]?.id ?? "track-1";
        const action = addClip({ trackId });

        // 同一 action 对象重放两次（模拟卫星窗口重放主窗口广播的动作）。
        const first = sessionReducer(sessionInitial, action);
        const second = sessionReducer(sessionInitial, action);

        expect(first.clips.at(-1)?.id).toBe(second.clips.at(-1)?.id);
        // id 必须真的在 action 里，而不是 reducer 现算的。
        expect(action.payload.clipId).toBe(first.clips.at(-1)?.id);
    });

    test("addClip 的 action 仍然可序列化（跨窗口广播的前提）", () => {
        const action = addClip({ trackId: sessionInitial.tracks[0]?.id ?? "track-1" });
        expect(() => JSON.stringify(action)).not.toThrow();
        expect(JSON.parse(JSON.stringify(action))).toEqual(action);
    });

    test("两次 addClip 派发得到不同的 id（prepare 里生成，仍然唯一）", () => {
        const trackId = sessionInitial.tracks[0]?.id ?? "track-1";
        const first = sessionReducer(sessionInitial, addClip({ trackId }));
        const second = sessionReducer(sessionInitial, addClip({ trackId }));
        expect(first.clips.at(-1)?.id).not.toBe(second.clips.at(-1)?.id);
    });
});

describe("dock reducer 纯度", () => {
    test("saveDockPreset 的时间戳由 action 携带，重放结果一致", () => {
        const action = saveDockPreset("my preset");
        const first = dockReducer(dockInitial, action);
        const second = dockReducer(dockInitial, action);

        expect(first.layout.presets?.["my preset"]?.createdAtMs).toBe(
            second.layout.presets?.["my preset"]?.createdAtMs,
        );
        expect(action.payload.createdAtMs).toBe(first.layout.presets?.["my preset"]?.createdAtMs);
    });
});
