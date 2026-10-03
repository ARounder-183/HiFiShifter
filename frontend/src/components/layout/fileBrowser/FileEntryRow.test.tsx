// @vitest-environment jsdom
/**
 * 文件浏览器行原语的展示与拖拽契约。
 *
 * 【为什么单独测行而不是整块渲染 `FileBrowserPanel`】行是纯展示组件：给它条目与
 * 几个布尔量即可，不认识 Redux / Tauri / i18n。整块面板要同时伪造 store、后端与
 * `AudioContext`，测试会因为与行无关的依赖而失败，从而不再可信。
 *
 * 【这里钉住的两条不变量】
 * 1. 目录名后**没有**尾随 `/`。`isDir` 已由图标、foldersFirst 排序与属性对话框的
 *    类型行表达；尾随斜杠是第四个冗余通道，而且 tooltip / 正则过滤 / type-ahead /
 *    复制文件名四处消费的都是裸 `name` —— 只有渲染带斜杠，于是「名字」有了两种取值。
 * 2. 目录行是拖拽源，但 `allowDrag={false}`（"此电脑"层）时必须不是 —— 那里每行
 *    都是盘符，拖入时间轴等于把整个盘交给递归扫描。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import type { FileEntry } from "../../../services/api/fileBrowser";
import { FileEntryRow } from "./FileEntryRow";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function entry(partial: Partial<FileEntry> & { name: string }): FileEntry {
    return {
        path: `C:/music/${partial.name}`,
        isDir: false,
        size: 1024,
        extension: null,
        modifiedTime: null,
        ...partial,
    };
}

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

function renderRow(props: {
    entry: FileEntry;
    onPointerDownForDrag?: (e: React.PointerEvent<HTMLDivElement>, entry: FileEntry) => void;
    allowDrag?: boolean;
}): void {
    act(() => {
        root.render(
            <FileEntryRow
                entry={props.entry}
                index={0}
                tabIndex={0}
                ariaPosInSet={1}
                ariaSetSize={1}
                active={false}
                onFocus={() => {}}
                registerRowRef={() => {}}
                isPlaying={false}
                onDoubleClickDir={() => {}}
                onRowClick={() => {}}
                onPointerDownForDrag={props.onPointerDownForDrag ?? (() => {})}
                onContextMenu={() => {}}
                isDragging={false}
                allowDrag={props.allowDrag}
            />,
        );
    });
}

function row(): HTMLElement {
    return container.querySelector<HTMLElement>('[role="option"]')!;
}

// ── 目录名不挂尾随斜杠 ──────────────────────────────────────────────────

test("目录行的可见名字不带尾随斜杠", () => {
    renderRow({ entry: entry({ name: "Takes", isDir: true }) });
    expect(row().textContent).toBe("Takes");
    expect(row().textContent).not.toContain("/");
});

test("目录行的 tooltip 与可见名字一致（同一个裸名）", () => {
    renderRow({ entry: entry({ name: "Takes", isDir: true }) });
    // 展示名与 tooltip 必须同源：分叉过一次（渲染带斜杠、tooltip 不带）就是这条测试的由来。
    expect(row().querySelector("[data-tooltip]")?.getAttribute("data-tooltip")).toBe("Takes");
});

test("文件行的名字同样不带斜杠（改动不能误伤文件）", () => {
    renderRow({ entry: entry({ name: "take_01.wav", extension: "wav" }) });
    expect(row().textContent).toBe("take_01.wav");
});

// ── 目录是可拖拽源 ─────────────────────────────────────────────────────

test("目录行可以作为拖拽源（拖入时间轴 = 目录导入）", () => {
    const onDrag = vi.fn();
    renderRow({ entry: entry({ name: "Takes", isDir: true }), onPointerDownForDrag: onDrag });
    act(() => {
        row().dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, cancelable: true, button: 0 }),
        );
    });
    expect(onDrag).toHaveBeenCalledTimes(1);
    expect(onDrag.mock.calls[0][1]).toMatchObject({ name: "Takes", isDir: true });
});

test("allowDrag=false 时目录行不是拖拽源（此电脑层的盘符）", () => {
    const onDrag = vi.fn();
    renderRow({
        entry: entry({ name: "C:", isDir: true, path: "C:\\" }),
        onPointerDownForDrag: onDrag,
        allowDrag: false,
    });
    act(() => {
        row().dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, cancelable: true, button: 0 }),
        );
    });
    expect(onDrag).not.toHaveBeenCalled();
});

test("不可拖拽的文件类型仍不接 pointerdown（回归：只放行目录）", () => {
    const onDrag = vi.fn();
    renderRow({
        entry: entry({ name: "notes.txt", extension: "txt" }),
        onPointerDownForDrag: onDrag,
    });
    act(() => {
        row().dispatchEvent(
            new PointerEvent("pointerdown", { bubbles: true, cancelable: true, button: 0 }),
        );
    });
    expect(onDrag).not.toHaveBeenCalled();
});
