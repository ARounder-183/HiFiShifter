import { describe, expect, it } from "vitest";
import {
    resolveParamBlockMarqueeSelection,
    resolveParamBlockRightClickSelection,
    resolveParamBlockRightPointerEnd,
    shouldStartParamBlockRightDragMarquee,
    toggleParamBlockEditingTool,
    type ParamBlockMarqueeHitRect,
} from "./paramBlockMarquee";

const rects: readonly ParamBlockMarqueeHitRect[] = [
    {
        id: "a",
        startSec: 10,
        endSec: 11,
        topPx: 20,
        heightPx: 18,
    },
    {
        id: "b",
        startSec: 11,
        endSec: 12,
        topPx: 22,
        heightPx: 18,
    },
    {
        id: "c",
        startSec: 20,
        endSec: 21,
        topPx: 20,
        heightPx: 18,
    },
];

describe("parameter block right-button gestures", () => {
    it("右键不拖动时打开块菜单，拖动后提交框选", () => {
        expect(resolveParamBlockRightPointerEnd({ marqueeActive: false })).toEqual({
            kind: "context-menu",
        });
        expect(resolveParamBlockRightPointerEnd({ marqueeActive: true })).toEqual({
            kind: "marquee-selection",
        });
    });

    it("右键已选块保留整组多选，右键未选块改为单选", () => {
        expect(
            resolveParamBlockRightClickSelection({
                selectedIds: ["a", "b"],
                hitId: "a",
            }),
        ).toEqual(["a", "b"]);
        expect(
            resolveParamBlockRightClickSelection({
                selectedIds: ["a", "b"],
                hitId: "c",
            }),
        ).toEqual(["c"]);
    });

    it("块工具中的右键拖拽可从空白处开始框选", () => {
        expect(
            shouldStartParamBlockRightDragMarquee({
                blockToolActive: true,
                contextMenu: true,
            }),
        ).toBe(true);
        expect(
            shouldStartParamBlockRightDragMarquee({
                blockToolActive: false,
                contextMenu: true,
            }),
        ).toBe(false);
    });

    it("~ 在块工具与音高绘制工具之间双向切换", () => {
        expect(toggleParamBlockEditingTool("block")).toBe("draw");
        expect(toggleParamBlockEditingTool("select")).toBe("block");
        expect(toggleParamBlockEditingTool("draw")).toBe("block");
    });

    it("框选坐标包含横向滚动偏移，并只返回相交块", () => {
        expect(
            resolveParamBlockMarqueeSelection({
                rects,
                marquee: {
                    leftPx: 20,
                    rightPx: 180,
                    topPx: 10,
                    bottomPx: 50,
                },
                pxPerSec: 100,
                scrollLeftPx: 950,
            }),
        ).toEqual(["a", "b"]);
    });

    it("框选会对宿主重复实例去重并按时间稳定排序", () => {
        expect(
            resolveParamBlockMarqueeSelection({
                rects: [rects[1]!, rects[0]!, rects[1]!],
                marquee: {
                    leftPx: 0,
                    rightPx: 250,
                    topPx: 0,
                    bottomPx: 60,
                },
                pxPerSec: 100,
                scrollLeftPx: 950,
            }),
        ).toEqual(["a", "b"]);
    });
});
