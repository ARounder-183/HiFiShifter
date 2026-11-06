/**
 * `set_param_frames` 的位置参数映射。
 *
 * 【为什么值得单独钉】`present` 是显式编辑位图：漏映射时后端会走
 * `vec![true; len]`，把整段显示数组（含未编辑帧）当成用户改过的每一帧写进权威。
 * 之后分析更新不再生效、几何变化也会把这段"编辑"搬走 —— 用户看到的是
 * "画一笔之后整条线被冻住 / 右键重置又回来"。插件模式与 Tauri 模式共用这张表，
 * 所以这里漏一次，两条后端一起中招。
 */
import { describe, expect, it } from "vitest";

import { buildTauriArgs } from "./invoke";

describe("set_param_frames 参数映射", () => {
    it("把第 6 个位置参数 present 映射进命名参数", () => {
        const mapped = buildTauriArgs("set_param_frames", [
            "track-1",
            "pitch",
            100,
            [60, 61, 62],
            true,
            [true, false, true],
        ]);
        expect(mapped).toMatchObject({
            trackId: "track-1",
            param: "pitch",
            startFrame: 100,
            values: [60, 61, 62],
            checkpoint: true,
            present: [true, false, true],
        });
    });

    it("不传 present 时保持缺席（后端按旧前端兼容）", () => {
        const mapped = buildTauriArgs("set_param_frames", ["t", "pitch", 0, [1], true]);
        expect(mapped).not.toHaveProperty("present");
    });
});
