/**
 * `timelineClipGlRenderer` 的纯逻辑测试。
 *
 * 【测什么】GL 路径无法在 Node 下真跑（无 WebGL2），但两处**纯 CPU 逻辑**
 * 一旦出错，真机上要么颜色全错、要么块面错位，且都很难一眼定位：
 *
 * 1. `parseRgbaColor`：颜色字符串 → 归一化分量；
 * 2. 实例数据布局：各字段偏移不得重叠、不得越界（布局错位在真机上表现为
 *    clip 尺寸/颜色张冠李戴）。
 *
 * 【不测什么】着色器与 GL 调用本身——需要真机目视确认，见模块头注释。
 */

import { describe, expect, it } from "vitest";

import {
    buildClipBodyInstance,
    buildGuideInstance,
    CLIP_INSTANCE_FLOATS,
    INSTANCE_MODE_BOX,
    INSTANCE_MODE_FLAT,
    parseRgbaColor,
} from "./timelineClipGlRenderer.js";

describe("parseRgbaColor", () => {
    it("解析 timelineCanvasStyle 产出的 rgba 格式", () => {
        // buildTimelineClipVisualStyle 生成的正是这种带空格的 rgba()。
        expect(parseRgbaColor("rgba(120, 140, 160, 0.82)")).toEqual([
            120 / 255,
            140 / 255,
            160 / 255,
            0.82,
        ]);
    });

    it("省略 alpha 时按 1 处理", () => {
        expect(parseRgbaColor("rgb(255, 0, 0)")[3]).toBe(1);
    });

    it("分量夹到 [0,1]，不会因脏数据让整块变透明或过曝", () => {
        const [r, g, b, a] = parseRgbaColor("rgba(999, -50, 128, 5)");
        expect(r).toBe(1);
        expect(g).toBe(0);
        expect(b).toBeCloseTo(128 / 255, 6);
        expect(a).toBe(1);
    });

    it("解析失败时回退到洋红（故意刺眼，让漏解析立刻可见）", () => {
        expect(parseRgbaColor("#4f8ef7")).toEqual([1, 0, 1, 1]);
        expect(parseRgbaColor("var(--qt-border)")).toEqual([1, 0, 1, 1]);
    });
});

describe("clip 实例数据布局", () => {
    const style = {
        headerFill: "rgba(10, 20, 30, 0.8)",
        bodyFill: "rgba(40, 50, 60, 0.7)",
        borderStroke: "rgba(255, 255, 255, 0.6)",
        borderLineWidth: 2,
        mutedAlpha: 1,
    };

    it("写入的字段能按各自偏移原样读回（无重叠、无越界）", () => {
        const buffer = new Float32Array(CLIP_INSTANCE_FLOATS);
        buildClipBodyInstance(
            buffer,
            0,
            {
                leftPx: 120,
                topPx: 48,
                widthPx: 200,
                heightPx: 90,
                headerHeightPx: 18,
                leadingOverlapPx: 40,
            },
            style,
            "rgb(31, 31, 31)",
        );

        // 几何
        expect(buffer[0]).toBe(120);
        expect(buffer[1]).toBe(48);
        expect(buffer[2]).toBe(200);
        expect(buffer[3]).toBe(90);
        // 圆角按 clip 尺寸收敛到常量 1.5
        expect(buffer[4]).toBeCloseTo(1.5, 6);
        expect(buffer[5]).toBe(18);
        // body 色（含 alpha）
        expect(buffer[6]).toBeCloseTo(40 / 255, 6);
        expect(buffer[9]).toBeCloseTo(0.7, 6);
        // header 色
        expect(buffer[10]).toBeCloseTo(10 / 255, 6);
        expect(buffer[13]).toBeCloseTo(0.8, 6);
        // 描边色
        expect(buffer[14]).toBe(1);
        expect(buffer[17]).toBeCloseTo(0.6, 6);
        // 描边宽度
        expect(buffer[18]).toBe(2);
        // 前导重叠
        expect(buffer[19]).toBe(40);
        // 分隔缝
        expect(buffer[20]).toBeCloseTo(0.5, 6);
        expect(buffer[21]).toBeCloseTo(31 / 255, 6);

        // 全部槽位都被覆盖（没有未初始化的空洞）
        expect(buffer.length).toBe(CLIP_INSTANCE_FLOATS);
        // clip 圆角盒的模式标记
        expect(buffer[24]).toBe(INSTANCE_MODE_BOX);
    });

    it("极小 clip：圆角按尺寸收敛，重叠宽度被钳到宽度以内", () => {
        const buffer = new Float32Array(CLIP_INSTANCE_FLOATS);
        buildClipBodyInstance(
            buffer,
            0,
            {
                leftPx: 0,
                topPx: 0,
                widthPx: 1,
                heightPx: 2,
                headerHeightPx: 18,
                leadingOverlapPx: 50,
            },
            style,
            null,
        );
        // radius = min(1.5, w/2=0.5, h/2=1) = 0.5
        expect(buffer[4]).toBeCloseTo(0.5, 6);
        // overlap = min(w - 1 = 0, 50) = 0 → 不足 0.5px 时按无重叠处理
        expect(buffer[19]).toBe(0);
        // 无分隔缝
        expect(buffer[20]).toBe(0);
    });

    it("平面矩形实例（行分界线）：模式为 FLAT，颜色落在 border 槽", () => {
        const buffer = new Float32Array(CLIP_INSTANCE_FLOATS);
        buildGuideInstance(buffer, 0, 0, 96, 1500, 1 / 2, "rgba(148, 163, 184, 0.22)");

        // 几何
        expect(buffer[0]).toBe(0);
        expect(buffer[1]).toBe(96);
        expect(buffer[2]).toBe(1500);
        expect(buffer[3]).toBeCloseTo(0.5, 6);
        // 无圆角、无分区、无描边、无重叠、无分隔缝
        expect(buffer[4]).toBe(0);
        expect(buffer[5]).toBe(0);
        expect(buffer[18]).toBe(0);
        expect(buffer[19]).toBe(0);
        expect(buffer[20]).toBe(0);
        // 颜色在 border 槽
        expect(buffer[14]).toBeCloseTo(148 / 255, 6);
        expect(buffer[17]).toBeCloseTo(0.22, 6);
        // 模式 = FLAT
        expect(buffer[24]).toBe(INSTANCE_MODE_FLAT);
    });

    it("多个实例按固定步长排布，互不覆盖", () => {
        const count = 5;
        const buffer = new Float32Array(count * CLIP_INSTANCE_FLOATS);
        for (let index = 0; index < count; index += 1) {
            buildClipBodyInstance(
                buffer,
                index,
                {
                    leftPx: index * 100,
                    topPx: 0,
                    widthPx: 90,
                    heightPx: 40,
                    headerHeightPx: 18,
                },
                style,
                null,
            );
        }
        for (let index = 0; index < count; index += 1) {
            // 每个实例的 x 都是自己写入的值，说明步长正确、未互相踩踏。
            expect(buffer[index * CLIP_INSTANCE_FLOATS]).toBe(index * 100);
        }
    });
});
