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
    findTypeAheadIndex,
    isFileListActivationKey,
    isFileListNavKey,
    nextActiveIndex,
    nextTypeAhead,
    rangeIndexes,
    selectionIntentOf,
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

// ── 纯函数：选区意图（与资源管理器一致） ─────────────────────────────────

const NO_MODS = { shift: false, ctrl: false, meta: false };

test("普通方向键替换选区（资源管理器模型）", () => {
    for (const key of ["ArrowDown", "ArrowUp", "Home", "End"]) {
        expect(selectionIntentOf(key, NO_MODS)).toBe("replace");
    }
});

test("Shift + 方向键从锚点扩展", () => {
    for (const key of ["ArrowDown", "ArrowUp", "Home", "End"]) {
        expect(selectionIntentOf(key, { ...NO_MODS, shift: true })).toBe("extend");
    }
});

test("Ctrl / Cmd + 方向键只移动光标，不动选区", () => {
    for (const key of ["ArrowDown", "ArrowUp", "Home", "End"]) {
        expect(selectionIntentOf(key, { ...NO_MODS, ctrl: true })).toBe("moveOnly");
        expect(selectionIntentOf(key, { ...NO_MODS, meta: true })).toBe("moveOnly");
    }
});

test("Ctrl + 空格是加/减选；裸空格与字母不改变选区", () => {
    expect(selectionIntentOf(" ", { ...NO_MODS, ctrl: true })).toBe("toggle");
    expect(selectionIntentOf(" ", { ...NO_MODS, meta: true })).toBe("toggle");
    // 裸空格是"试听"、字母是快速跳转，都不该动选区。
    expect(selectionIntentOf(" ", NO_MODS)).toBeNull();
    expect(selectionIntentOf("a", NO_MODS)).toBeNull();
    expect(selectionIntentOf("Enter", NO_MODS)).toBeNull();
    // Ctrl + 字母（全选 / 复制路径）由面板的组合键分支处理，不属于选区意图。
    expect(selectionIntentOf("a", { ...NO_MODS, ctrl: true })).toBeNull();
});

test("rangeIndexes：闭区间，两个方向都成立", () => {
    expect(rangeIndexes(2, 5)).toEqual([2, 3, 4, 5]);
    expect(rangeIndexes(5, 2)).toEqual([2, 3, 4, 5]);
    expect(rangeIndexes(3, 3)).toEqual([3]);
    // 任一端为负（尚无锚点 / 尚无光标）返回空数组 —— 不能把 -1 当成第 0 行。
    expect(rangeIndexes(-1, 2)).toEqual([]);
    expect(rangeIndexes(2, -1)).toEqual([]);
    expect(rangeIndexes(-1, -1)).toEqual([]);
});

// ── 纯函数：输入字母快速跳转（type-ahead） ────────────────────────────────

const NAMES = ["apple.txt", "banana.wav", "fa1.wav", "fa2.wav", "melon.wav"];

test("type-ahead：无选中时跳到第一个前缀匹配", () => {
    expect(nextTypeAhead(NAMES, "", "f", -1)).toEqual({ index: 2, buffer: "f" });
    expect(nextTypeAhead(NAMES, "", "b", -1)).toEqual({ index: 1, buffer: "b" });
});

test("type-ahead：有选中时从下一行找，末尾绕回开头", () => {
    // 选中 fa1（下标 2），输入 f → 下一个 f 开头的是 fa2。
    expect(nextTypeAhead(NAMES, "", "f", 2)).toEqual({ index: 3, buffer: "f" });
    // 选中 melon（末行），输入 a → 后面没有 a 开头，绕回 apple。
    expect(nextTypeAhead(NAMES, "", "a", 4)).toEqual({ index: 0, buffer: "a" });
});

test("type-ahead：连续输入累积前缀（f → fa）", () => {
    // 第一击 f 落在 fa1；第二击 a 用 fa 从 fa1 之后找 → fa2。
    expect(nextTypeAhead(NAMES, "f", "a", 2)).toEqual({ index: 3, buffer: "fa" });
    // 只有一个 fa 候选时，从它自己之后找会绕回它自己。
    expect(nextTypeAhead(["fab.wav"], "f", "a", 0)).toEqual({ index: 0, buffer: "fa" });
});

test("type-ahead：同字母连按在前缀匹配项之间循环", () => {
    // 已在 fa1（f 的缓冲），再按 f →「ff」无匹配 → 退回单字母 f 从下一行找。
    expect(nextTypeAhead(NAMES, "f", "f", 2)).toEqual({ index: 3, buffer: "f" });
});

test("type-ahead：无匹配不跳转，失败的前缀不污染缓冲", () => {
    expect(nextTypeAhead(NAMES, "", "z", -1)).toEqual({ index: null, buffer: "" });
    // 扩展成无匹配前缀（fa 不存在）同样不跳转，缓冲回退到 f。
    expect(nextTypeAhead(["fb.wav"], "f", "a", 0)).toEqual({ index: null, buffer: "f" });
    // fa 无匹配则 fab 亦不可能存在，失败后的下一键从 f 重新组合：f+b 命中。
    expect(nextTypeAhead(["fb.wav"], "f", "b", 0)).toEqual({ index: 0, buffer: "fb" });
});

test("type-ahead：大小写不敏感", () => {
    expect(nextTypeAhead(NAMES, "", "F", -1)).toEqual({ index: 2, buffer: "F" });
    expect(findTypeAheadIndex(["README.md"], "read", 0)).toBe(0);
});

test("type-ahead：空列表与空查询不产生跳转", () => {
    expect(findTypeAheadIndex([], "f", 0)).toBe(-1);
    expect(findTypeAheadIndex(NAMES, "", 0)).toBe(-1);
    // start 超界自动回绕。
    expect(findTypeAheadIndex(NAMES, "a", 99)).toBe(0);
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
 * 同样的 roving tabindex 与键盘分支（含 `nextTypeAhead` 的 type-ahead 接线），
 * 以及同样的 `focusRow`（`scrollIntoView` + `focus({preventScroll})`）。
 * 只把"进入目录 / 试听"替换成回调。
 */
function ListboxHarness({
    entries,
    onActivate,
    selectedIndexes = [],
}: {
    entries: string[];
    onActivate: (index: number) => void;
    /** 初始选区（模拟"鼠标已经点过这些行"）。 */
    selectedIndexes?: number[];
}) {
    const [activeIndex, setActiveIndex] = useState(-1);
    const [selected, setSelected] = useState<Set<number>>(() => new Set(selectedIndexes));
    const anchorRef = useRef(-1);
    const rowRefs = useRef<(HTMLDivElement | null)[]>([]);
    const typeAheadBufferRef = useRef("");

    const focusRow = (index: number) => {
        const el = rowRefs.current[index];
        if (!el) return;
        try {
            el.scrollIntoView({ block: "nearest" });
        } catch {
            /* jsdom 无布局 */
        }
        el.focus({ preventScroll: true });
    };

    /** 与 FileBrowserPanel.applySelectionIntent 同一套规则。 */
    const applyIntent = (intent: ReturnType<typeof selectionIntentOf>, index: number) => {
        if (intent === null || intent === "moveOnly") return;
        if (intent === "extend") {
            const anchor = anchorRef.current >= 0 ? anchorRef.current : index;
            setSelected(new Set(rangeIndexes(anchor, index)));
            return;
        }
        if (intent === "toggle") {
            setSelected((prev) => {
                const next = new Set(prev);
                if (next.has(index)) next.delete(index);
                else next.add(index);
                return next;
            });
            anchorRef.current = index;
            return;
        }
        anchorRef.current = index;
        setSelected(new Set([index]));
    };

    const handleKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
        const intent = selectionIntentOf(event.key, {
            shift: event.shiftKey,
            ctrl: event.ctrlKey,
            meta: event.metaKey,
        });
        if (isFileListNavKey(event.key)) {
            event.preventDefault();
            const next = nextActiveIndex(activeIndex, event.key, entries.length);
            if (next < 0) return;
            setActiveIndex(next);
            focusRow(next);
            applyIntent(intent, next);
            return;
        }
        if (intent === "toggle" && activeIndex >= 0) {
            event.preventDefault();
            applyIntent(intent, activeIndex);
            return;
        }
        if (isFileListActivationKey(event.key) && activeIndex >= 0) {
            event.preventDefault();
            onActivate(activeIndex);
            return;
        }
        // 与 FileBrowserPanel.handlePanelKeyDown 相同的 type-ahead 接线：
        // 可打印单字符进入增量搜索，命中即移动活动行并聚焦。
        const key = event.key;
        if (key.length !== 1 || key === " ") return;
        const result = nextTypeAhead(entries, typeAheadBufferRef.current, key, activeIndex);
        typeAheadBufferRef.current = result.buffer;
        if (result.index == null) return;
        event.preventDefault();
        setActiveIndex(result.index);
        focusRow(result.index);
    };

    return (
        <div role="listbox" aria-label="files" onKeyDown={handleKeyDown}>
            {entries.map((name, index) => (
                <AppListRow
                    key={name}
                    role="option"
                    selected={selected.has(index)}
                    active={index === activeIndex}
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

/**
 * 向 listbox 派发一次按键，返回该事件是否被 `preventDefault`。
 *
 * 修饰键与"行焦点"可指定：`Shift` / `Ctrl` 扩展选区时事件必须落在列表容器内
 * （真实场景里焦点在行上，事件冒泡到容器）。
 */
function press(
    key: string,
    modifiers: { shift?: boolean; ctrl?: boolean; meta?: boolean } = {},
): boolean {
    const listbox = container.querySelector('[role="listbox"]')!;
    let prevented = false;
    act(() => {
        const event = new KeyboardEvent("keydown", {
            key,
            bubbles: true,
            cancelable: true,
            shiftKey: modifiers.shift ?? false,
            ctrlKey: modifiers.ctrl ?? false,
            metaKey: modifiers.meta ?? false,
        });
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

/**
 * 每行是否携带键盘光标标记。
 *
 * 【为什么断言 `data-active` 而不是 `aria-selected`】多选 listbox 里"键盘光标行"
 * 与"选中行"是两件事：光标恒为一个、选中可为 0..N 个。把光标行报成
 * `aria-selected="true"` 会让读屏把未选中的行念成已选中。光标有自己的属性。
 */
function activeFlags(): boolean[] {
    return rows().map((row) => row.dataset.active === "true");
}

test("listbox 接线：ArrowDown 移动活动行，Enter 激活该行", () => {
    const onActivate = vi.fn();
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={onActivate} />);
    });

    expect(rows()).toHaveLength(3);
    // 无活动行时首行可 Tab 进入（roving tabindex 的起点）。
    expect(tabIndexes()).toEqual([0, -1, -1]);
    expect(activeFlags()).toEqual([false, false, false]);

    // 方向键必须 preventDefault，否则容器会跟着滚动。
    expect(press("ArrowDown")).toBe(true);
    expect(tabIndexes()).toEqual([0, -1, -1]);
    expect(activeFlags()).toEqual([true, false, false]);

    // 再下移一格：活动行移到第二行。
    press("ArrowDown");
    expect(tabIndexes()).toEqual([-1, 0, -1]);
    expect(activeFlags()).toEqual([false, true, false]);

    // Enter 激活当前活动行（第二行）。
    expect(press("Enter")).toBe(true);
    expect(onActivate).toHaveBeenCalledTimes(1);
    expect(onActivate).toHaveBeenCalledWith(1);
});

test("键盘光标与多选选中是两条独立通道（各自有独立属性）", () => {
    act(() => {
        root.render(
            <ListboxHarness
                entries={["a", "b", "c"]}
                onActivate={vi.fn()}
                // 鼠标 Ctrl+点击选了首末两行。
                selectedIndexes={[0, 2]}
            />,
        );
    });

    // 选中态由 `data-selected` 表达，光标态由 `data-active` 表达 —— 两个独立通道。
    expect(rows().map((row) => row.dataset.selected === "true")).toEqual([true, false, true]);
    expect(activeFlags()).toEqual([false, false, false]);

    /*
     * 用 `Ctrl` + 方向键移动光标：这是"只移动光标、不动选区"的那一条。
     * （普通方向键会替换选区 —— 资源管理器模型，见下面的选区测试。）
     */
    press("ArrowDown", { ctrl: true });
    press("ArrowDown", { ctrl: true });
    expect(activeFlags()).toEqual([false, true, false]);
    expect(rows().map((row) => row.dataset.selected === "true")).toEqual([true, false, true]);

    // 光标行的视觉来自描边通道（`index.css` 的 `[data-active]:focus`），而不是
    // 再叠一层背景色；未成为光标的选中行不应带光标标记。
    expect(rows()[1].dataset.active).toBe("true");
    expect(rows()[0].dataset.active).toBeUndefined();
});

test("listbox 接线：End 跳到末行后 Enter 激活末行", () => {
    const onActivate = vi.fn();
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={onActivate} />);
    });

    press("End");
    expect(tabIndexes()).toEqual([-1, -1, 0]);
    expect(activeFlags()).toEqual([false, false, true]);
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

test("listbox 接线：输入字母跳到前缀匹配的行，无匹配不动", () => {
    act(() => {
        root.render(
            <ListboxHarness
                entries={["apple.txt", "banana.wav", "fa1.wav", "fa2.wav"]}
                onActivate={vi.fn()}
            />,
        );
    });

    // 输入 f：跳到第一个 f 开头的行并聚焦（focus 会同步活动行）。
    expect(press("f")).toBe(true);
    expect(document.activeElement).toBe(rows()[2]);
    expect(activeFlags()).toEqual([false, false, true, false]);

    // 无匹配（fz）不跳转，也不消费按键，光标停在原地。
    expect(press("z")).toBe(false);
    expect(document.activeElement).toBe(rows()[2]);
    expect(activeFlags()).toEqual([false, false, true, false]);

    // 再按 f：同字母连按 → 退回单字母从下一行找 → fa2。
    expect(press("f")).toBe(true);
    expect(document.activeElement).toBe(rows()[3]);
    expect(activeFlags()).toEqual([false, false, false, true]);
});

test("focusRow 在无布局环境下不抛（jsdom 没有 scrollIntoView 实现）", () => {
    const scrollIntoView = Element.prototype.scrollIntoView;
    // 模拟 jsdom：`scrollIntoView` 未实现 / 抛错。
    Element.prototype.scrollIntoView = () => {
        throw new Error("not implemented");
    };
    try {
        act(() => {
            root.render(<ListboxHarness entries={["a", "b"]} onActivate={vi.fn()} />);
        });
        // 滚动失败不得阻断聚焦 —— 焦点移动才是语义要求。
        expect(() => press("ArrowDown")).not.toThrow();
        expect(document.activeElement).toBe(rows()[0]);
    } finally {
        Element.prototype.scrollIntoView = scrollIntoView;
    }
});

// ── DOM：键盘选区模型（与资源管理器一致） ─────────────────────────────────

/** 每行是否被选中（读 `aria-selected`，即 AppListRow 的 `selected`）。 */
function selectedFlags(): boolean[] {
    return rows().map((row) => row.getAttribute("aria-selected") === "true");
}

test("普通方向键：移动光标并选中该行（资源管理器模型）", () => {
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={vi.fn()} />);
    });

    press("ArrowDown");
    expect(activeFlags()).toEqual([true, false, false]);
    expect(selectedFlags()).toEqual([true, false, false]);

    press("ArrowDown");
    expect(activeFlags()).toEqual([false, true, false]);
    // 关键：选区跟着光标走，因此随后 Delete / F2 / Ctrl+C 作用的就是这一行，
    // 而不是"之前用鼠标选过的那一行"。
    expect(selectedFlags()).toEqual([false, true, false]);
});

test("Shift + 方向键：从锚点扩展选区", () => {
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={vi.fn()} />);
    });

    press("ArrowDown"); // 光标与锚点都到第 0 行
    press("ArrowDown", { shift: true });
    press("ArrowDown", { shift: true });

    expect(activeFlags()).toEqual([false, false, true]);
    expect(selectedFlags()).toEqual([true, true, true]);
});

test("Ctrl + 方向键：只移动光标，选区不动", () => {
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={vi.fn()} />);
    });

    press("ArrowDown"); // 选中第 0 行
    press("ArrowDown", { shift: true }); // 选区扩到 0..1
    press("ArrowDown", { ctrl: true }); // 光标到第 2 行，选区仍是 0..1

    expect(activeFlags()).toEqual([false, false, true]);
    expect(selectedFlags()).toEqual([true, true, false]);
});

test("Ctrl + 空格：把光标行加入 / 移出选区", () => {
    act(() => {
        root.render(<ListboxHarness entries={["a", "b", "c"]} onActivate={vi.fn()} />);
    });

    press("ArrowDown");
    press("ArrowDown", { ctrl: true }); // 光标到第 1 行，选区仍是第 0 行
    expect(selectedFlags()).toEqual([true, false, false]);

    press(" ", { ctrl: true }); // 加入第 1 行
    expect(selectedFlags()).toEqual([true, true, false]);

    press(" ", { ctrl: true }); // 再按一次移出
    expect(selectedFlags()).toEqual([true, false, false]);
});
