import { describe, expect, it } from "vitest";

import { isOverlapEditRegion, resolveOverlapPressTarget } from "./overlapPress";

const clip = (id: string) => ({ id });

describe("resolveOverlapPressTarget", () => {
    it("A 的淡出线命中保留 A 为控件归属，body 前景 B 不覆盖它", () => {
        const hit = {
            kind: "clip",
            region: "fade-out-corner" as const,
            fromOverlapRegion: true,
            clip: clip("A"),
            foregroundClip: clip("B"),
        };
        const press = resolveOverlapPressTarget(hit);
        expect(press.controlOwner).toBe(true);
        expect(press.selectionClipId).toBe("A");
        expect(press.hit.clip.id).toBe("A");
    });

    it("B 的淡入线在 A 更短时仍选 B", () => {
        const press = resolveOverlapPressTarget({
            kind: "clip",
            region: "fade-in-corner" as const,
            fromOverlapRegion: true,
            clip: clip("B"),
            foregroundClip: clip("A"),
        });
        expect(press.controlOwner).toBe(true);
        expect(press.selectionClipId).toBe("B");
        expect(press.hit.clip.id).toBe("B");
    });

    it("交叉点抓手选后一个 Clip，并保留 partner 信息", () => {
        const press = resolveOverlapPressTarget({
            kind: "clip",
            region: "crossfade-grip" as const,
            fromOverlapRegion: true,
            clip: clip("B"),
            foregroundClip: clip("B"),
        });
        expect(press.controlOwner).toBe(true);
        expect(press.selectionClipId).toBe("B");
    });

    it("body 命中选前景 Clip", () => {
        const press = resolveOverlapPressTarget({
            kind: "clip",
            region: "body" as const,
            fromOverlapRegion: false,
            clip: clip("B"),
            foregroundClip: clip("B"),
        });
        expect(press.controlOwner).toBe(false);
        expect(press.selectionClipId).toBe("B");
    });

    it("empty 没有选择目标", () => {
        expect(resolveOverlapPressTarget({ kind: "empty" }).selectionClipId).toBeNull();
    });

    it("反向长度组合：B 的淡入线 owner=B、foreground=A，仍选 B", () => {
        const press = resolveOverlapPressTarget({
            kind: "clip",
            region: "fade-in-corner" as const,
            fromOverlapRegion: true,
            clip: clip("B"),
            foregroundClip: clip("A"),
        });
        expect(press.selectionClipId).toBe("B");
        expect(press.hit.clip.id).toBe("B");
    });

    it("所有重叠区编辑控件都算控件归属", () => {
        for (const region of [
            "left-edge",
            "right-edge",
            "fade-in-corner",
            "fade-out-corner",
            "crossfade-grip",
        ] as const) {
            expect(isOverlapEditRegion(region)).toBe(true);
        }
        expect(isOverlapEditRegion("body")).toBe(false);
        expect(isOverlapEditRegion("header")).toBe(false);
        expect(isOverlapEditRegion("snap-offset-handle")).toBe(false);
    });
});
