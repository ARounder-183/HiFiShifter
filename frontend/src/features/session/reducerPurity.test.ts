/*
 * reducer 纯度回归测试。
 *
 * 【为什么必须有】卫星窗口（`features/dock/detachBridge.ts`）会把主窗口广播的
 * 动作在**本地 reducer 上重放**，以此保持两个窗口的状态收敛。这条机制的隐含
 * 前提是"同一个 action 对象 → 同一个新状态"。一旦 reducer 里出现随机数或时钟
 * 读取，主窗口与卫星窗口就会为同一个动作算出**不同的结果**：
 *
 *   - session 侧的 `addClip` 曾在 reducer 里 `Math.random()` 生成 clip id →
 *     两个窗口的 `session.clips` 里的 id 永久不同（快照只在卫星窗口启动时对齐
 *     一次）。该 action 后经全仓审查确认为死代码（无生产调用）已删除，其
 *     prepare 纯度用例一并移除；"id 必须随动作携带"的契约如今由导入路径的
 *     `importedClipId`（路径确定性导出）从结构上保证；
 *   - `saveDockPreset` 曾在 reducer 里 `Date.now()` 写 `createdAtMs` → 预设的
 *     时间戳两边不一致。
 *
 * 这两类偏差不会抛异常、不会有日志，症状只是"另一个窗口里的东西对不上"，
 * 极难归因。因此这里用"同一 action 重放两次必须得到同一状态"直接钉住。
 */
import { describe, expect, test } from "vitest";

import dockReducer, { saveDockPreset } from "../dock/dockSlice";

const dockInitial = dockReducer(undefined, { type: "@@init" });

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
