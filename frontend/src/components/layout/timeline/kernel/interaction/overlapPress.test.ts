import { describe, expect, it } from "vitest";

import {
    isOverlapEditRegion,
    resolveOverlapPressTarget,
    type OverlapPressModifiers,
} from "./overlapPress";

const clip = (id: string) => ({ id });
const noModifiers: OverlapPressModifiers = {
    altKey: false,
    ctrlKey: false,
    metaKey: false,
    shiftKey: false,
};

describe("resolveOverlapPressTarget", () => {
    it.each([
        "fade-in-corner",
        "fade-out-corner",
        "crossfade-grip",
        "left-edge",
        "right-edge",
    ] as const)("%s 的普通单击选前景，但拖拽仍保留控件归属", (region) => {
        const hit = {
            kind: "clip",
            region,
            fromOverlapRegion: true,
            clip: clip("owner"),
            foregroundClip: clip("foreground"),
        };
        const press = resolveOverlapPressTarget(hit, noModifiers);
        expect(press.controlOwner).toBe(true);
        expect(press.plainControlClick).toBe(true);
        expect(press.selectionClipId).toBe("foreground");
        expect(press.controlClipId).toBe("owner");
        expect(press.hit.clip.id).toBe("owner");
        expect(press.hit.region).toBe(region);
    });

    it("任一编辑修饰键 → 单击直接选控件归属", () => {
        const hit = {
            kind: "clip",
            region: "fade-out-corner" as const,
            fromOverlapRegion: true,
            clip: clip("owner"),
            foregroundClip: clip("foreground"),
        };
        for (const key of ["altKey", "ctrlKey", "metaKey", "shiftKey"] as const) {
            const press = resolveOverlapPressTarget(hit, { ...noModifiers, [key]: true });
            expect(press.controlOwner).toBe(true);
            expect(press.plainControlClick).toBe(false);
            expect(press.selectionClipId).toBe("owner");
            expect(press.controlClipId).toBe("owner");
            expect(press.hit.clip.id).toBe("owner");
        }
    });

    it("A 的淡出线命中保留 A 为控件归属，body 前景 B 不覆盖它", () => {
        const press = resolveOverlapPressTarget(
            {
                kind: "clip",
                region: "fade-out-corner",
                fromOverlapRegion: true,
                clip: clip("A"),
                foregroundClip: clip("B"),
            },
            noModifiers,
        );
        expect(press.selectionClipId).toBe("B");
        expect(press.controlClipId).toBe("A");
        expect(press.hit.clip.id).toBe("A");
    });

    it("B 的淡入线在 A 更短时单击选 A、拖动仍编辑 B", () => {
        const press = resolveOverlapPressTarget(
            {
                kind: "clip",
                region: "fade-in-corner",
                fromOverlapRegion: true,
                clip: clip("B"),
                foregroundClip: clip("A"),
            },
            noModifiers,
        );
        expect(press.selectionClipId).toBe("A");
        expect(press.controlClipId).toBe("B");
        expect(press.hit.clip.id).toBe("B");
    });

    it("交叉点抓手选后一个 Clip，并保留 partner 信息", () => {
        const press = resolveOverlapPressTarget(
            {
                kind: "clip",
                region: "crossfade-grip",
                fromOverlapRegion: true,
                clip: clip("B"),
                foregroundClip: clip("B"),
            },
            noModifiers,
        );
        expect(press.controlOwner).toBe(true);
        expect(press.selectionClipId).toBe("B");
        expect(press.controlClipId).toBe("B");
    });

    it("body 命中选前景 Clip", () => {
        const press = resolveOverlapPressTarget(
            {
                kind: "clip",
                region: "body",
                fromOverlapRegion: false,
                clip: clip("B"),
                foregroundClip: clip("B"),
            },
            noModifiers,
        );
        expect(press.controlOwner).toBe(false);
        expect(press.plainControlClick).toBe(false);
        expect(press.selectionClipId).toBe("B");
        expect(press.controlClipId).toBeNull();
    });

    it("header 不是重叠区编辑控件", () => {
        const press = resolveOverlapPressTarget(
            {
                kind: "clip",
                region: "header",
                fromOverlapRegion: true,
                clip: clip("later"),
                foregroundClip: clip("short"),
            },
            noModifiers,
        );
        expect(press.controlOwner).toBe(false);
        expect(press.selectionClipId).toBe("short");
        expect(press.controlClipId).toBeNull();
    });

    it("非重叠区命中不改变普通淡变角手感", () => {
        const press = resolveOverlapPressTarget(
            {
                kind: "clip",
                region: "fade-in-corner",
                fromOverlapRegion: false,
                clip: clip("A"),
                foregroundClip: clip("B"),
            },
            noModifiers,
        );
        expect(press.controlOwner).toBe(false);
        expect(press.selectionClipId).toBe("B");
        expect(press.controlClipId).toBeNull();
    });

    it("empty 没有选择目标", () => {
        expect(
            resolveOverlapPressTarget({ kind: "empty" }, noModifiers).selectionClipId,
        ).toBeNull();
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
