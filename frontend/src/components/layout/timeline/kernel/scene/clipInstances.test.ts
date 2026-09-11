/**
 * clip 实例构建（./clipInstances）行为自检。
 *
 * 【主要内容】
 * 1. 实例数 = clip 数，缓冲容量足够；
 * 2. 分隔缝判定：同轨紧贴才画、有间隔不画、跨轨不互相影响；
 * 3. 缓冲跨帧复用（避免每帧大分配）；
 * 4. 空输入安全。
 *
 * 【作用】分隔缝判定的容差与"按轨分组"是视觉正确性的关键：漏判会让相邻 clip
 * 糊成一块，误判会在 clip 中间画出一道缝。缓冲复用则是滚动帧零分配的前提。
 *
 * 【与其他模块的关系】覆盖 `clipInstances.ts`；复用 `runtime/timelineClipGlRenderer`
 * 的实例布局常量。样式模块在无 DOM 环境走兜底分支，因此本测试无需 jsdom。
 */

import { describe, expect, it } from "vitest";

import { CLIP_INSTANCE_FLOATS } from "../../runtime/timelineClipGlRenderer";
import { createClipInstanceBuilder, type ClipInstanceClip } from "./clipInstances";

/**
 * 实例布局中「分隔缝宽度」的 float 偏移。
 *
 * 来源：`runtime/timelineClipGlRenderer` 的私有常量 `OFF_SEAM`（= 20）。
 * 该常量未导出，这里按布局契约硬编码并在此说明；若布局变更，本断言会失败
 * 并提示同步（比"悄悄测不到"更安全）。
 */
const OFF_SEAM = 20;

function makeClip(overrides: Partial<ClipInstanceClip> = {}): ClipInstanceClip {
    return {
        id: "clip-a",
        trackId: "track-1",
        name: "Clip A",
        leftPx: 0,
        topPx: 0,
        widthPx: 100,
        heightPx: 80,
        headerHeightPx: 18,
        selected: false,
        muted: false,
        gain: 1,
        playbackRate: 1,
        ...overrides,
    };
}

/** 读取第 index 个实例的分隔缝宽度。 */
function seamWidthOf(instances: Float32Array, index: number): number {
    return instances[index * CLIP_INSTANCE_FLOATS + OFF_SEAM];
}

function build(clips: ClipInstanceClip[]) {
    return createClipInstanceBuilder().build({
        clips,
        darkMode: false,
        seamColor: "rgb(31, 31, 31)",
    });
}

describe("clipInstances", () => {
    it("实例数 = clip 数，缓冲容量足够", () => {
        const result = build([makeClip({ id: "a" }), makeClip({ id: "b", leftPx: 200 })]);
        expect(result.count).toBe(2);
        expect(result.instances.length).toBeGreaterThanOrEqual(2 * CLIP_INSTANCE_FLOATS);
    });

    it("紧贴的相邻 clip 画分隔缝，孤立的 clip 不画", () => {
        const result = build([
            makeClip({ id: "a", leftPx: 0, widthPx: 100 }),
            makeClip({ id: "b", leftPx: 100, widthPx: 100 }),
        ]);
        // a 的右缘紧贴 b 的左缘 → a 画缝；b 右侧无邻居 → 不画。
        expect(seamWidthOf(result.instances, 0)).toBeGreaterThan(0);
        expect(seamWidthOf(result.instances, 1)).toBe(0);
    });

    it("有间隔的相邻 clip 不画分隔缝", () => {
        const result = build([
            makeClip({ id: "a", leftPx: 0, widthPx: 100 }),
            makeClip({ id: "b", leftPx: 120, widthPx: 100 }),
        ]);
        expect(seamWidthOf(result.instances, 0)).toBe(0);
    });

    it("不同轨道不参与彼此的分隔缝判定", () => {
        const result = build([
            makeClip({ id: "a", trackId: "t1", leftPx: 0, widthPx: 100 }),
            makeClip({ id: "b", trackId: "t2", leftPx: 100, widthPx: 100 }),
        ]);
        expect(seamWidthOf(result.instances, 0)).toBe(0);
    });

    it("缓冲跨帧复用（两次 build 返回同一缓冲）", () => {
        const builder = createClipInstanceBuilder();
        const args = {
            clips: [makeClip()],
            darkMode: false,
            seamColor: "rgb(31, 31, 31)",
        };
        const first = builder.build(args);
        const second = builder.build(args);
        expect(second.instances).toBe(first.instances);
    });

    it("空输入返回 0 个实例", () => {
        const result = build([]);
        expect(result.count).toBe(0);
    });

    it("分隔缝容差边界：0.4px 命中、0.6px 不命中", () => {
        const hit = build([
            makeClip({ id: "a", leftPx: 0, widthPx: 100 }),
            makeClip({ id: "b", leftPx: 100.4, widthPx: 100 }),
        ]);
        expect(seamWidthOf(hit.instances, 0)).toBeGreaterThan(0);

        const miss = build([
            makeClip({ id: "a", leftPx: 0, widthPx: 100 }),
            makeClip({ id: "b", leftPx: 100.6, widthPx: 100 }),
        ]);
        expect(seamWidthOf(miss.instances, 0)).toBe(0);
    });

    it("BOX 实例的 mode 槽为 0（与 sdf-box program 的 BOX 模式约定一致）", () => {
        const result = build([makeClip()]);
        // OFF_MODE = 24：非 0 会被着色器当作平面矩形，块面将失去圆角与分区着色。
        expect(result.instances[24]).toBe(0);
    });

    it("输出每个 clip 的文字色（长度与 clip 数一致，供文字层使用）", () => {
        const result = build([makeClip({ id: "a" }), makeClip({ id: "b", leftPx: 200 })]);
        expect(result.textFills).toHaveLength(2);
        // 样式模块产出 rgba(...) 文本；文字层用 parseRgbaColor 解析。
        expect(result.textFills[0]).toMatch(/^rgba?\(/);
    });

    it("实例缓冲按倍增策略增长（复用容量策略）", () => {
        const builder = createClipInstanceBuilder();
        const small = builder.build({
            clips: [makeClip()],
            darkMode: false,
            seamColor: "rgb(31, 31, 31)",
        });
        const smallLength = small.instances.length;
        const large = builder.build({
            clips: Array.from({ length: 8 }, (_unused, index) =>
                makeClip({ id: `clip-${index}`, leftPx: index * 200 }),
            ),
            darkMode: false,
            seamColor: "rgb(31, 31, 31)",
        });
        expect(large.count).toBe(8);
        expect(large.instances.length).toBeGreaterThanOrEqual(8 * CLIP_INSTANCE_FLOATS);
        expect(large.instances.length).toBeGreaterThan(smallLength);
    });
});
