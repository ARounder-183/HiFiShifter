/**
 * 参数编辑器内核 · 字形渲染适配层单测。
 *
 * 【本测试守护什么】字形管线依赖离屏 Canvas2D，在 node 环境（本工程 Vitest 无
 * jsdom）下构造必定返回 null。因此这里守住两条契约：
 * 1. **无 DOM 时不崩**：`createPianoRollGlyphs` 返回 null，调用方据此跳过文字
 *    （这是产品要求：无 WebGL2 / 无 Canvas2D 时面板其余部分必须照常工作）；
 * 2. **字号解析**：垂直基准换算依赖字号，必须与光栅化器同口径（不锚定行首，
 *    因此 `bold 9px …` 也要能取到 9）。这条在真实渲染里直接影响文字是否垂直居中。
 *
 * 【为什么其余行为测不到】对齐 / 截断 / 四边形几何都在真实光栅化之后才能验证，
 * 只能在浏览器里做像素比对（阶段 2 的验证记录里有对应证据）。
 */
import { describe, expect, it } from "vitest";

import { createPianoRollGlyphs, PIANO_ROLL_ATLAS_PAGE_SIZE_PX } from "./pianoRollGlyphs";

describe("createPianoRollGlyphs", () => {
    it("无 DOM 环境返回 null（不抛错）", () => {
        expect(createPianoRollGlyphs({ dpr: 2 })).toBeNull();
    });

    it("图集页边长的默认值是 2048（与字形 program 的契约）", () => {
        expect(PIANO_ROLL_ATLAS_PAGE_SIZE_PX).toBe(2048);
    });
});
