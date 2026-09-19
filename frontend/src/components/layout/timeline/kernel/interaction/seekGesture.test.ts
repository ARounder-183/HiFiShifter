/**
 * 空白区 / 标尺 seek 手势阶段策略单测（`./seekGesture`）。
 *
 * 【要锁住的回归】播放中对时间轴空白处按下或拖拽会"闪回"播放光标：
 * 旧实现按下即写 `playheadSec`（乐观值 + 后端 seek），而拖拽帧只写乐观值，
 * 与 30Hz 播放轮询（用引擎真实播放位置覆写同一字段）反复争夺 —— 视觉插值
 * 在"硬复位"与"按采样外推"之间逐帧交替。
 *
 * 修复后的不变式：
 * 1. 播放中 `press` / `move` 都不写播放头（`"none"`），只有 `release` 提交；
 * 2. 非播放态保持既有行为（press = commit，move = preview，release = commit）；
 * 3. 「清空选中 + 切轨」只在按下那一拍执行一次。
 */
import { describe, expect, it } from "vitest";

import { planSeekGesture, type SeekGesturePhase } from "./seekGesture";

describe("planSeekGesture — 播放中（闪回防护）", () => {
    it("【回归】播放中按下不写播放头", () => {
        expect(planSeekGesture("press", true).playhead).toBe("none");
    });

    it("【回归】播放中拖拽中间帧不写播放头", () => {
        expect(planSeekGesture("move", true).playhead).toBe("none");
    });

    it("播放中松手提交落点（延续播放）", () => {
        expect(planSeekGesture("release", true).playhead).toBe("commit");
    });

    it("播放中整段手势只有松手一拍会写播放头", () => {
        const phases: SeekGesturePhase[] = ["press", "move", "release"];
        const writes = phases
            .map((phase) => planSeekGesture(phase, true).playhead)
            .filter((w) => w !== "none");
        expect(writes).toEqual(["commit"]);
    });
});

describe("planSeekGesture — 非播放态（既有行为不变）", () => {
    it("按下即提交（点空白立即跳转）", () => {
        expect(planSeekGesture("press", false).playhead).toBe("commit");
    });

    it("拖拽帧只预览（不打后端）", () => {
        expect(planSeekGesture("move", false).playhead).toBe("preview");
    });

    it("松手提交", () => {
        expect(planSeekGesture("release", false).playhead).toBe("commit");
    });
});

describe("planSeekGesture — 空白点击选中语义", () => {
    it("只在按下那一拍执行一次（松手不重复）", () => {
        expect(planSeekGesture("press", false).blankClickSemantics).toBe(true);
        expect(planSeekGesture("move", false).blankClickSemantics).toBe(false);
        expect(planSeekGesture("release", false).blankClickSemantics).toBe(false);
        expect(planSeekGesture("press", true).blankClickSemantics).toBe(true);
        expect(planSeekGesture("release", true).blankClickSemantics).toBe(false);
    });
});

describe("planSeekGesture — 播放态是唯一输入变量", () => {
    it("同阶段下播放/非播放必须给出不同计划（press 与 move 都被挡）", () => {
        for (const phase of ["press", "move"] as SeekGesturePhase[]) {
            expect(planSeekGesture(phase, true).playhead).not.toBe(
                planSeekGesture(phase, false).playhead,
            );
        }
        // release 与播放态无关：两条路径都提交。
        expect(planSeekGesture("release", true).playhead).toBe(
            planSeekGesture("release", false).playhead,
        );
    });
});
