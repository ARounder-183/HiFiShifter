/**
 * 参数编辑器 · 按 WebGL 上下文归属的字形表面。
 *
 * glyph program 与创建它的 WebGL 上下文绑定；GlRasterTarget 只描述尺寸，
 * 不会切换上下文。主画布与轴画布必须各自持有 program、rasterizer、atlas
 * 上传页集合，否则块标签会被画进轴画布并被压缩成白色碎片。
 */

import { createGlyphProgram, type GlyphProgram } from "../../../renderKernel/gl/glyphProgram";
import type { GlRasterTarget } from "../../../renderKernel/gl/glRaster";
import type { GlyphQuad } from "../../../renderKernel/gl/glyphQuads";
import {
    createPianoRollGlyphs,
    type PianoRollGlyphs,
    type TextRequest,
} from "../glyph/pianoRollGlyphs";

export interface GlyphSurface {
    readonly gl: WebGL2RenderingContext;
    readonly program: GlyphProgram;
    readonly glyphs: PianoRollGlyphs;
    readonly uploadedAtlasPages: Set<number>;
    build(requests: readonly TextRequest[]): GlyphQuad[];
    render(
        quads: readonly GlyphQuad[],
        target: GlRasterTarget,
        viewOriginX: number,
        viewOriginY: number,
        expectedGl?: WebGL2RenderingContext,
    ): void;
    dispose(): void;
}

export interface GlyphSurfaceFactories {
    readonly createProgram?: (gl: WebGL2RenderingContext) => GlyphProgram;
    readonly createGlyphs?: (dpr: number) => PianoRollGlyphs | null;
}

/**
 * 创建一个只服务指定上下文的字形表面。
 *
 * @returns program 或 rasterizer 不可用时返回 null；调用方只跳过文字，不影响
 * 同一画布上的几何图层。
 */
export function createGlyphSurface(
    gl: WebGL2RenderingContext,
    dpr: number,
    factories: GlyphSurfaceFactories = {},
): GlyphSurface | null {
    let program: GlyphProgram | null = null;
    let glyphs: PianoRollGlyphs | null = null;
    try {
        program = (factories.createProgram ?? createGlyphProgram)(gl);
        glyphs = (factories.createGlyphs ?? ((value) => createPianoRollGlyphs({ dpr: value })))(
            dpr,
        );
        if (glyphs === null) {
            program.dispose();
            return null;
        }
    } catch {
        program?.dispose();
        glyphs?.dispose();
        return null;
    }

    const uploadedAtlasPages = new Set<number>();
    const surface: GlyphSurface = {
        gl,
        program,
        glyphs,
        uploadedAtlasPages,
        build(requests) {
            return glyphs.build(requests);
        },
        render(quads, target, viewOriginX, viewOriginY, expectedGl) {
            if (quads.length === 0) return;
            if (import.meta.env.DEV && expectedGl !== undefined && gl !== expectedGl) {
                throw new Error("glyph surface used with a foreign WebGL context");
            }
            for (const page of glyphs.consumeDirtyPages()) {
                const data = glyphs.readPage(page);
                if (data !== null) {
                    program.uploadAtlas(data, glyphs.atlasPageSizePx());
                    uploadedAtlasPages.add(page);
                }
            }
            if (uploadedAtlasPages.size === 0) return;
            // resize 已设置过 viewport；这里再设一次，保证只更新视口原点/尺寸的
            // 调用也不会沿用另一块画布的 viewport。
            gl.viewport(0, 0, target.physicalWidthPx, target.physicalHeightPx);
            program.render(quads, target, viewOriginX, viewOriginY);
        },
        dispose() {
            program.dispose();
            glyphs.dispose();
            uploadedAtlasPages.clear();
        },
    };
    return surface;
}
