// @vitest-environment jsdom
/**
 * 文件浏览器键盘可达性契约。
 *
 * 【为什么测 helper + 行原语的组合，而不是整块渲染 `FileBrowserPanel`】
 * 面板直接依赖 Redux store 与 Tauri 后端（`fileBrowserApi` 的 `invoke`），
 * 把它整块挂进 jsdom 要同时伪造 store、后端、`AudioContext` 与 Radix 的测量
 * 环境 —— 测试很容易因与键盘无关的依赖而失败，从而不再可信。
 *
 * 因此把"下一个活动行下标"这段算术抽到 `fileBrowserKeyboardNav.ts`：
 *   - 纯函数部分（ArrowDown/ArrowUp/Home/End 的移动与夹紧）直接单测；
 *   - 再用**同一个 helper** 与**同一个 `AppListRow`** 复现面板 listbox 的接线
 *     （同样的 roving tabindex、同样的 `onKeyDown` 分支），在真实 DOM 上断言
 *     "ArrowDown 移动活动行、Enter 激活当前行"。
 */

import { act, useRef, useState, type KeyboardEvent as ReactKeyboardEvent } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { AppListRow } from "../../ui/ListRow";
import {
    isFileListActivationKey,
    isFileListNavKey,
    nextActiveIndex,
} from "./fileBrowserKeyboardNav";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// ── 纯函数：活动行移动 ──────────────────────────────────────────────────

test("ArrowDown 下移活动行，到末行夹紧", () => {
    expect(nextActiveIndex(-1, "ArrowDown", 3)).toBe(0);
    expect(nextActiveIndex(0, "ArrowDown", 3)).toBe(1);
    expect(nextActiveIndex(1, "ArrowDown", 3)).toBe(2);
    expect(nextActiveIndex(2, "ArrowDown", 3)).toBe(2);
});

test("ArrowUp 上移活动行，到首行夹紧", () => {
    expect(nextActiveIndex(-1, "ArrowUp", 3)).toBe(2);
    expect(nextActiveIndex(2, "ArrowUp", 3)).toBe(1);
    expect(nextActiveIndex(1, "ArrowUp", 3)).toBe(0);
    expect(nextActiveIndex(0, "ArrowUp", 3)).toBe(0);
});

test("Home / End 跳到首 / 末行", () => {
    expect(nextActiveIndex(1, "Home", 4)).toBe(0);
    expect(nextActiveIndex(1, "End", 4)).toBe(3);
    expect(nextActiveIndex(-1, "Home", 4)).toBe(0);
    expect(nextActiveIndex(-1, "End", 4)).toBe(3);
});

test("空列表不产生活动行", () => {
    for (const key of ["ArrowDown", "ArrowUp", "Home", "End"]) {
        expect(nextActiveIndex(-1, key, 0)).toBe(-1);
    }
});

test("未知按键不改动活动行", () => {
    expect(nextActiveIndex(1, "PageDown", 5)).toBe(1);
});

test("按键分类：导航键与激活键", () => {
    for (const key of ["ArrowDown", "ArrowUp", "Home", "End"]) {
        expect(isFileListNavKey(key)).toBe(true);
        expect(isFileListActivationKey(key)).toBe(false);
    }
    expect(isFileListActivationKey("Enter")).toBe(true);
    expect(isFileListActivationKey(" ")).toBe(true);
    expect(isFileListActivationKey("Escape")).toBe(false);
    expect(isFileListNavKey("Enter")).toBe(false);
});

// ── DOM：listbox + 行原语的接线 ─────────────────────────────────────────

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

/**
 * 复现 `FileBrowserPanel` listbox 的接线：同一 helper、同一 `AppListRow` 行原语、
 * 同样的 roving tabindex 与键盘分支。只把"进入目录 / 试听"替换成回调。
 */
function ListboxHarness({
    entries,
    onActivate,
}: {
    entries: string[];
    onActivate: (index: number) => void;
}) {
    const [activeIndex, setActiveIndex] = useState(-1);
    const rowRefs = useRef<(HTMLDivElement | null)[]>([]);

    const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
        if (isFileListNavKey(event.key)) {
            event.preventDefault();
            const next = nextActiveIndex(activeIndex, event.key, entries.length);
            if (next < 0) return;
            setActiveIndex(next);
            rowRefs.current[next]?.focus();
            return;
        }
        if (isFileListActivationKey(event.key) && activeIndex >= 0) {
            event.preventDefault();
            onActivate(activeIndex);
        }
    };

    return (
        <div role="listbox" aria-label="files" onKeyDown={handleKeyDown}>
            {entries.map((name, index) => (
                <AppListRow
                    key={name}
                    role="option"
                    selected={index === activeIndex}
                    tabIndex={index === (activeIndex >= 0 ? activeIndex : 0) ? 0 : -1}
                    onFocus={() => setActiveIndex(index)}
                    ref={(el) => {
                        rowRefs.current[index] = el;
                    }}
                    onClick={() => {}}
                >
                    {name}
                </AppListRow>
            ))}
        </div>
    );
}

/** 向 listbox 派发一次按键，返回该事件是否被 `preventDefault`。 */
function press(key: string): boolean {
    const listbox = container.querySelector('[role="listbox"]')!;
    let prevented = false;
    act(() => {
        const event = new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true });
        listbox.dispatchEvent(event);
        prevented = event.defaultPrevented;
    });
    return prevented;
}

function rows(): HTMLElement[] {
    return Array.from(container.querySelectorAll<HTMLElement>('[role="option"]'));
}

function tabIndexes(): number[] {
    return rows().map((row) => row.tabIndex);
}

test("listbox 接线：ArrowDown 移动活动行，Enter 激活该行", () => {
    const onActivate = vi.fn();
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={onActivate} />);
    });

    expect(rows()).toHaveLength(3);
    // 无活动行时首行可 Tab 进入（roving tabindex 的起点）。
    expect(tabIndexes()).toEqual([0, -1, -1]);

    // 方向键必须 preventDefault，否则容器会跟着滚动。
    expect(press("ArrowDown")).toBe(true);
    expect(tabIndexes()).toEqual([0, -1, -1]);
    expect(rows()[0].getAttribute("aria-selected")).toBe("true");

    // 再下移一格：活动行移到第二行。
    press("ArrowDown");
    expect(tabIndexes()).toEqual([-1, 0, -1]);
    expect(rows()[1].getAttribute("aria-selected")).toBe("true");
    expect(rows()[0].getAttribute("aria-selected")).toBe("false");

    // Enter 激活当前活动行（第二行）。
    expect(press("Enter")).toBe(true);
    expect(onActivate).toHaveBeenCalledTimes(1);
    expect(onActivate).toHaveBeenCalledWith(1);
});

test("listbox 接线：End 跳到末行后 Enter 激活末行", () => {
    const onActivate = vi.fn();
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={onActivate} />);
    });

    press("End");
    expect(tabIndexes()).toEqual([-1, -1, 0]);
    press("Enter");
    expect(onActivate).toHaveBeenCalledWith(2);
});

test("无活动行时 Enter 不激活任何行", () => {
    const onActivate = vi.fn();
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={onActivate} />);
    });

    press("Enter");
    expect(onActivate).not.toHaveBeenCalled();
});
