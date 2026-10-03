/**
 * 文件浏览器拖拽的手势判定（`./fileBrowserDragGesture`）。
 *
 * 【这里钉住的是"打断"这条规则】左键拖拽中右键**点击** = 打断，右键拖拽中左键
 * 点击 = 打断。三处容易写错的地方各有用例：
 *   1. "点击"不是"按下"（按下后移动超过阈值就不再是点击）；
 *   2. 打断只认与发起键相反的那个键；
 *   3. `cancel` 载荷必须带全 `dirPaths`（此前 `pointercancel` 漏了这个字段）。
 */

import { describe, expect, it } from "vitest";

import {
    FILE_DRAG_THRESHOLD_PX,
    buildFileDragFinishDetail,
    dragButtonOf,
    interruptCandidateMoved,
    interruptCandidateOnDown,
    isInterruptRelease,
    type FileDragSource,
} from "./fileBrowserDragGesture";

const source: FileDragSource = {
    filePath: "C:\\music\\Takes\\vocal.wav",
    fileName: "vocal.wav",
    allFilePaths: ["C:\\music\\Takes\\vocal.wav", "C:\\music\\Takes\\Sub"],
    dirPaths: ["C:\\music\\Takes\\Sub"],
    isRightDrag: false,
};

describe("dragButtonOf", () => {
    it("右键拖拽的发起键是 2，其余是 0", () => {
        expect(dragButtonOf(true)).toBe(2);
        expect(dragButtonOf(false)).toBe(0);
    });
});

describe("buildFileDragFinishDetail", () => {
    it("drop 与 cancel 形状一致，且都带全 dirPaths", () => {
        const drop = buildFileDragFinishDetail(source, "drop", 10, 20);
        const cancel = buildFileDragFinishDetail(source, "cancel", 30, 40);
        expect(drop).toEqual({
            type: "drop",
            filePath: source.filePath,
            fileName: source.fileName,
            filePaths: source.allFilePaths,
            dirPaths: source.dirPaths,
            clientX: 10,
            clientY: 20,
            isRightDrag: false,
        });
        // 【回归】cancel 此前走 `drop + canceled`，detail 里没有 dirPaths。
        expect(cancel.dirPaths).toEqual(source.dirPaths);
        expect(cancel.type).toBe("cancel");
        expect(cancel.clientX).toBe(30);
        expect(cancel.clientY).toBe(40);
    });
});

describe("interruptCandidateOnDown", () => {
    it("拖拽未激活时不记录（那只是点击）", () => {
        expect(
            interruptCandidateOnDown({
                active: false,
                isRightDrag: false,
                button: 2,
                x: 1,
                y: 2,
            }),
        ).toBeNull();
    });

    it("按下的是发起键本身时不记录（那是'再加一个键'，不是后悔）", () => {
        expect(
            interruptCandidateOnDown({
                active: true,
                isRightDrag: false,
                button: 0,
                x: 1,
                y: 2,
            }),
        ).toBeNull();
        expect(
            interruptCandidateOnDown({
                active: true,
                isRightDrag: true,
                button: 2,
                x: 1,
                y: 2,
            }),
        ).toBeNull();
    });

    it("中键等其它键不是打断信号", () => {
        expect(
            interruptCandidateOnDown({
                active: true,
                isRightDrag: false,
                button: 1,
                x: 1,
                y: 2,
            }),
        ).toBeNull();
    });

    it("左键拖拽中按下右键 → 记录候选；右键拖拽中按下左键 → 记录候选", () => {
        expect(
            interruptCandidateOnDown({
                active: true,
                isRightDrag: false,
                button: 2,
                x: 5,
                y: 6,
            }),
        ).toEqual({ button: 2, x: 5, y: 6 });
        expect(
            interruptCandidateOnDown({
                active: true,
                isRightDrag: true,
                button: 0,
                x: 7,
                y: 8,
            }),
        ).toEqual({ button: 0, x: 7, y: 8 });
    });
});

describe("interruptCandidateMoved", () => {
    const candidate = { button: 2, x: 100, y: 100 };

    it("阈值内的移动仍算点击（打断成立）", () => {
        expect(interruptCandidateMoved(candidate, 102, 101)).toBe(false);
        expect(interruptCandidateMoved(candidate, 104, 100)).toBe(false);
    });

    it("恰好达到阈值即算移动（打断作废）", () => {
        expect(interruptCandidateMoved(candidate, 100 + FILE_DRAG_THRESHOLD_PX, 100)).toBe(true);
        expect(interruptCandidateMoved(candidate, 103, 104)).toBe(true);
    });

    it("阈值外的移动（如 20px）算拖动，不是点击", () => {
        expect(interruptCandidateMoved(candidate, 120, 100)).toBe(true);
    });

    it("非有限位移保守判为已移动（不误判成点击）", () => {
        expect(interruptCandidateMoved(candidate, Number.NaN, 100)).toBe(true);
    });
});

describe("isInterruptRelease", () => {
    it("没有候选 → 不是打断（正常松手）", () => {
        expect(isInterruptRelease(null, 0)).toBe(false);
        expect(isInterruptRelease(null, 2)).toBe(false);
    });

    it("松开的键与候选一致 → 打断", () => {
        expect(isInterruptRelease({ button: 2, x: 0, y: 0 }, 2)).toBe(true);
        expect(isInterruptRelease({ button: 0, x: 0, y: 0 }, 0)).toBe(true);
    });

    it("松开的键与候选不一致 → 不是打断（发起键松开才是收尾）", () => {
        expect(isInterruptRelease({ button: 2, x: 0, y: 0 }, 0)).toBe(false);
        expect(isInterruptRelease({ button: 0, x: 0, y: 0 }, 2)).toBe(false);
    });
});
