import { describe, expect, it, vi } from "vitest";

import type { GlRasterTarget } from "../../../renderKernel/gl/glRaster";
import type { GlyphProgram } from "../../../renderKernel/gl/glyphProgram";
import type { GlyphQuad } from "../../../renderKernel/gl/glyphQuads";
import type { PianoRollGlyphs } from "../glyph/pianoRollGlyphs";
import { createGlyphSurface } from "./glyphSurface";

function fakeTarget(): GlRasterTarget {
    return {
        physicalWidthPx: 200,
        physicalHeightPx: 100,
        cssWidthPx: 200,
        cssHeightPx: 100,
        dpr: 1,
    };
}

function fakeProgram() {
    const uploadAtlas = vi.fn();
    const render = vi.fn();
    const dispose = vi.fn();
    const program = { uploadAtlas, render, repaint: vi.fn(), dispose } as unknown as GlyphProgram;
    return { program, uploadAtlas, render, dispose };
}

function fakeGlyphs() {
    const quad = { x: 1, y: 2, w: 3, h: 4, u0: 0, v0: 0, u1: 1, v1: 1, rgba: [1, 1, 1, 1] } as GlyphQuad;
    const build = vi.fn(() => [quad]);
    const consumeDirtyPages = vi.fn(() => [0]);
    const readPage = vi.fn(() => new Uint8ClampedArray(16));
    const dispose = vi.fn();
    const glyphs = {
        build,
        atlasPageSizePx: () => 64,
        consumeDirtyPages,
        readPage,
        dispose,
    } as unknown as PianoRollGlyphs;
    return { glyphs, build, consumeDirtyPages, dispose, quad };
}

describe("createGlyphSurface", () => {
    it("渲染时上传自己的脏页、设置自己的 viewport 并绘制", () => {
        const viewport = vi.fn();
        const gl = { viewport } as unknown as WebGL2RenderingContext;
        const { program, uploadAtlas, render } = fakeProgram();
        const { glyphs, quad } = fakeGlyphs();
        const surface = createGlyphSurface(gl, 1, {
            createProgram: () => program,
            createGlyphs: () => glyphs,
        });
        expect(surface).not.toBeNull();
        if (surface === null) return;

        surface.render([quad], fakeTarget(), 0, 0, gl);
        expect(viewport).toHaveBeenCalledWith(0, 0, 200, 100);
        expect(uploadAtlas).toHaveBeenCalledTimes(1);
        expect(render).toHaveBeenCalledWith([quad], expect.anything(), 0, 0);
        expect(surface.uploadedAtlasPages.has(0)).toBe(true);
    });

    it("两个 surface 的 atlas 上传页互不共享", () => {
        const glA = { viewport: vi.fn() } as unknown as WebGL2RenderingContext;
        const glB = { viewport: vi.fn() } as unknown as WebGL2RenderingContext;
        const a = createGlyphSurface(glA, 1, {
            createProgram: () => fakeProgram().program,
            createGlyphs: () => fakeGlyphs().glyphs,
        });
        const b = createGlyphSurface(glB, 1, {
            createProgram: () => fakeProgram().program,
            createGlyphs: () => fakeGlyphs().glyphs,
        });
        expect(a).not.toBeNull();
        expect(b).not.toBeNull();
        if (a === null || b === null) return;
        a.render([fakeGlyphs().quad], fakeTarget(), 0, 0, glA);
        expect(a.uploadedAtlasPages.has(0)).toBe(true);
        expect(b.uploadedAtlasPages.has(0)).toBe(false);
    });

    it("dev 下拒绝把 surface 用于另一个 WebGL 上下文", () => {
        const gl = { viewport: vi.fn() } as unknown as WebGL2RenderingContext;
        const other = {} as WebGL2RenderingContext;
        const surface = createGlyphSurface(gl, 1, {
            createProgram: () => fakeProgram().program,
            createGlyphs: () => fakeGlyphs().glyphs,
        });
        if (surface === null) throw new Error("surface");
        expect(() => surface.render([fakeGlyphs().quad], fakeTarget(), 0, 0, other)).toThrow(
            /foreign WebGL context/,
        );
    });

    it("释放时同时释放 program 与 rasterizer", () => {
        const gl = { viewport: vi.fn() } as unknown as WebGL2RenderingContext;
        const p = fakeProgram();
        const g = fakeGlyphs();
        const surface = createGlyphSurface(gl, 1, {
            createProgram: () => p.program,
            createGlyphs: () => g.glyphs,
        });
        surface?.dispose();
        expect(p.dispose).toHaveBeenCalledTimes(1);
        expect(g.dispose).toHaveBeenCalledTimes(1);
    });
});
