/**
 * 快捷键设置窗口的搜索与分类导航。
 *
 * 【为什么需要这个文件】搜索引擎本身有 `keybindingSearch.test.ts` 覆盖，但"输入
 * 查询后界面上真的只剩几条"、"点分类后真的只剩该组"属于**接线**，纯函数测不到。
 * 这两层之间出过实际情况颠倒的错误 —— 例如滤镜的结果是对的，但渲染的是另一份数组。
 *
 * 【为什么 DOM 断言走 `data-hs-kb-row`】`KeybindingsActionRow` 把分组放在这个
 * 属性上，测试据此判断"当前渲染了哪些组"，不依赖文案与语系。
 */
// @vitest-environment jsdom
import { describe, expect, it, vi } from "vitest";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { configureStore } from "@reduxjs/toolkit";

import { KeybindingsDialog } from "./KeybindingsDialog";
import { I18nProvider } from "../../i18n/I18nProvider";
import keybindingsReducer from "../../features/keybindings/keybindingsSlice";
import sessionReducer from "../../features/session/sessionSlice";
import { ALL_ACTION_IDS } from "../../features/keybindings/defaultKeybindings";

/*
 * jsdom 没有 `ResizeObserver`，而 Radix 的 ScrollArea 在布局副作用里会用它。
 * 与 `ExportAudioDialog.cancel.test.tsx` 同一处处理。
 */
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

/**
 * 极简 store：只挂本组件真正读的两个切片。
 *
 * 【为什么不复用真实 store】真实 store 要拉起 playback / project 等一整套设备状态
 * 与副作用；本测试只关心"绑定 → 行"这一段，用最小 store 才是诚实的依赖。
 *
 * `session` 是搜索匹配设置（转写开关与宽严）的所在 —— 搜索框要把动作名交给后端
 * 转写，这份设置决定转写成什么形态，所以它是本组件的**真实依赖**，不是顺手加的。
 */
function createTestStore() {
    return configureStore({
        reducer: { keybindings: keybindingsReducer, session: sessionReducer },
    });
}

/*
 * 【为什么所有查询都走 `document.body` 而不是容器】`AppDialog` 用 Radix Dialog，
 * 它把内容 portal 到 `<body>` —— 容器里始终只有 0 长度 HTML。
 */
/** 渲染对话框；返回值里的 `root` 为 `document.body`（portal 目的地）。 */
function renderDialog(onOpenChange: (open: boolean) => void = () => {}) {
    const container = document.createElement("div");
    document.body.appendChild(container);
    const root = createRoot(container);
    act(() => {
        root.render(
            <Provider store={createTestStore()}>
                <I18nProvider>
                    <KeybindingsDialog open onOpenChange={onOpenChange} />
                </I18nProvider>
            </Provider>,
        );
    });
    return {
        root: document.body,
        cleanup: () => {
            act(() => root.unmount());
            container.remove();
        },
    };
}

function visibleRowGroups(): string[] {
    return Array.from(document.body.querySelectorAll<HTMLElement>("[data-hs-kb-row]")).map(
        (row) => row.dataset.hsKbRow!,
    );
}

function searchInput(): HTMLInputElement {
    const input = document.body.querySelector<HTMLInputElement>(
        "input[type='text'], input:not([type])",
    );
    if (!input) throw new Error("搜索框未渲染");
    return input;
}

function applyQuery(value: string) {
    const input = searchInput();
    act(() => {
        // React 受控输入：直接赋 value 不会被 onChange 捕获，必须走原生 setter。
        const setter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype,
            "value",
        )!.set!;
        setter.call(input, value);
        input.dispatchEvent(new Event("input", { bubbles: true }));
    });
}

function pressKey(key: string) {
    act(() => {
        searchInput().dispatchEvent(new KeyboardEvent("keydown", { key, bubbles: true }));
    });
}

describe("KeybindingsDialog — 搜索过滤", () => {
    it("无查询时渲染全部动作", () => {
        const { cleanup } = renderDialog();
        try {
            expect(visibleRowGroups().length).toBe(ALL_ACTION_IDS.length);
        } finally {
            cleanup();
        }
    });

    it("输入查询后只渲染命中的条目", () => {
        const { cleanup } = renderDialog();
        try {
            applyQuery("ctrl z");
            const rows = visibleRowGroups();
            expect(rows.length).toBeGreaterThan(0);
            expect(rows.length).toBeLessThan(ALL_ACTION_IDS.length);
        } finally {
            cleanup();
        }
    });

    it("搜赴不到的查询给出空态而不是空白", () => {
        const { cleanup } = renderDialog();
        try {
            applyQuery("zzzzzznope");
            expect(visibleRowGroups()).toEqual([]);
        } finally {
            cleanup();
        }
    });

    it("Esc 清空查询而不关闭窗口", () => {
        const onOpenChange = vi.fn();
        const { cleanup } = renderDialog(onOpenChange);
        try {
            applyQuery("undo");
            expect(visibleRowGroups().length).toBeLessThan(ALL_ACTION_IDS.length);

            pressKey("Escape");
            // 查询清空 → 回到全量；窗口没有被关闭。
            expect(onOpenChange).not.toHaveBeenCalled();
            expect(visibleRowGroups().length).toBe(ALL_ACTION_IDS.length);
        } finally {
            cleanup();
        }
    });

    it("搜索框里按 ↓ 把焦点交到第一个结果行的按键按钮", () => {
        const { cleanup } = renderDialog();
        try {
            pressKey("ArrowDown");
            const firstRow = document.body.querySelector<HTMLElement>("[data-hs-kb-row]");
            const button = firstRow?.querySelector("[data-hs-kb-bind]");
            expect(button).toBeTruthy();
            expect(document.activeElement).toBe(button);
        } finally {
            cleanup();
        }
    });

    it("搜索框里按 ↑ 跳到最后一个结果行", () => {
        const { cleanup } = renderDialog();
        try {
            pressKey("ArrowUp");
            const rows = document.body.querySelectorAll<HTMLElement>("[data-hs-kb-row]");
            const button = rows[rows.length - 1]?.querySelector("[data-hs-kb-bind]");
            expect(button).toBeTruthy();
            expect(document.activeElement).toBe(button);
        } finally {
            cleanup();
        }
    });

    it("Enter 不触发对话框的默认动作（不关窗）", () => {
        const onOpenChange = vi.fn();
        const { cleanup } = renderDialog(onOpenChange);
        try {
            pressKey("Enter");
            expect(onOpenChange).not.toHaveBeenCalled();
        } finally {
            cleanup();
        }
    });
});

describe("KeybindingsDialog — 分类导航", () => {
    it("列出「全部」与全部 14 个分组", () => {
        const { cleanup } = renderDialog();
        try {
            expect(document.body.querySelectorAll('[role="option"]').length).toBe(15);
        } finally {
            cleanup();
        }
    });

    it("点击某一分类后只渲染该组的条目", () => {
        const { cleanup } = renderDialog();
        try {
            const options = document.body.querySelectorAll<HTMLElement>('[role="option"]');
            // 第 0 项是"全部"，第 1 项起是真实分组。
            act(() => {
                options[1].dispatchEvent(new MouseEvent("click", { bubbles: true }));
            });
            const groups = new Set(visibleRowGroups());
            expect(groups.size).toBe(1);
            expect(visibleRowGroups().length).toBeGreaterThan(0);
        } finally {
            cleanup();
        }
    });

    it("分类与搜索同时生效：结果必须是两者的交集", () => {
        const { cleanup } = renderDialog();
        try {
            // 先选一个具体分类，再搜一个该分类内的词。
            const options = document.body.querySelectorAll<HTMLElement>('[role="option"]');
            act(() => {
                options[1].dispatchEvent(new MouseEvent("click", { bubbles: true }));
            });
            const groupOnly = new Set(visibleRowGroups());
            expect(groupOnly.size).toBe(1);
            const targetGroup = [...groupOnly][0];

            applyQuery("zzzzzznope");
            expect(visibleRowGroups()).toEqual([]);

            applyQuery("");
            const afterClear = visibleRowGroups();
            expect(new Set(afterClear)).toEqual(new Set([targetGroup]));
        } finally {
            cleanup();
        }
    });
});

/**
 * 一个功能可以绑多个快捷键（见 `features/keybindings/types.ts` 的 `KeybindingMap`）。
 *
 * 这里只钉住**接线**：默认表里重做有两个绑定 → 界面上就该有两颗 chip；
 * 能追加槽位的动作要有 `+` 按钮，不能追加的（修饰键手势）不能有。
 */
describe("多绑定：一行的 chip 列表与追加按钮", () => {
    it("「重做」行渲染两颗 chip（默认绑了 Ctrl+Shift+Z 与 Ctrl+Y）", () => {
        const { cleanup } = renderDialog();
        try {
            // 默认表里只有 edit.redo 有第二个槽位，因此 slot=1 全窗口唯一。
            const secondSlots =
                document.body.querySelectorAll<HTMLElement>('[data-hs-kb-slot="1"]');
            expect(secondSlots).toHaveLength(1);
            const row = secondSlots[0].closest("[data-hs-kb-row]");
            expect(row).not.toBeNull();
            expect(row!.querySelectorAll("[data-hs-kb-slot]")).toHaveLength(2);
        } finally {
            cleanup();
        }
    });

    it("普通动作行有追加按钮，修饰键手势行没有", () => {
        const { cleanup } = renderDialog();
        try {
            expect(document.body.querySelectorAll("[data-hs-kb-add]").length).toBeGreaterThan(0);
            // 修饰键手势只有一个槽位：按下哪一个组合算触发无法解释，因此不提供追加。
            const modifierRows = Array.from(
                document.body.querySelectorAll<HTMLElement>('[data-hs-kb-row="modClip"]'),
            );
            expect(modifierRows.length).toBeGreaterThan(0);
            for (const row of modifierRows) {
                expect(row.querySelector("[data-hs-kb-add]")).toBeNull();
            }
        } finally {
            cleanup();
        }
    });

    it("点击 chip 进入录入态（chip 变成按键提示）", () => {
        const { cleanup } = renderDialog();
        try {
            const chip = document.body.querySelector<HTMLElement>('[data-hs-kb-slot="0"]');
            expect(chip).not.toBeNull();
            const before = chip!.textContent;
            act(() => {
                chip!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
            });
            const after = document.body.querySelector<HTMLElement>('[data-hs-kb-slot="0"]');
            expect(after!.textContent).not.toBe(before);
        } finally {
            cleanup();
        }
    });
});

/**
 * 行的布局稳定性：录入态不得移动任何已有元素。
 *
 * 【回归背景】chip 的文字会在录入时变成"请按键…"，`+` 也会因为不能追加而消失 ——
 * 两者都会让整行左右跳动（用户报告的"按钮不对齐"）。修法是 chip 固定宽度 +
 * `+` 的槽位常驻。jsdom 没有布局引擎，因此这里锁**结构性保证**：
 * 行内元素数量在进入录入态前后不变、`+` 仍在（只是禁用）、chip 宽度是固定值。
 */
describe("布局稳定性：录入态不移动已有元素", () => {
    it("进入录入态不改变行内元素数量，且追加按钮仍占位（禁用）", () => {
        const { cleanup } = renderDialog();
        try {
            const chip = document.body.querySelector<HTMLElement>('[data-hs-kb-slot="0"]');
            expect(chip).not.toBeNull();
            const row = chip!.closest<HTMLElement>("[data-hs-kb-row]");
            expect(row).not.toBeNull();
            const before = row!.querySelectorAll("[data-hs-kb-slot], [data-hs-kb-add]").length;

            act(() => {
                chip!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
            });

            const rowAfter = document.body
                .querySelector<HTMLElement>('[data-hs-kb-slot="0"]')!
                .closest<HTMLElement>("[data-hs-kb-row]")!;
            expect(rowAfter.querySelectorAll("[data-hs-kb-slot], [data-hs-kb-add]").length).toBe(
                before,
            );
            const add = rowAfter.querySelector<HTMLButtonElement>("[data-hs-kb-add]");
            expect(add).not.toBeNull();
            expect(add!.disabled).toBe(true);
        } finally {
            cleanup();
        }
    });

    it("每个 chip 都是固定宽度 + 省略号（文字长短不改变布局）", () => {
        const { cleanup } = renderDialog();
        try {
            const chips = Array.from(
                document.body.querySelectorAll<HTMLElement>("[data-hs-kb-slot]"),
            );
            expect(chips.length).toBeGreaterThan(0);
            const widths = new Set<string>();
            for (const chip of chips) {
                expect(chip.style.width).not.toBe("");
                /*
                 * 截断走 `truncate` 工具类（`overflow:hidden` + `text-overflow:ellipsis`
                 * + `white-space:nowrap`），不再是三个内联属性 —— chip 已改用
                 * `AppButton`，内联样式只保留"固定宽度"这一项（它没有对应令牌）。
                 */
                expect(chip.classList.contains("truncate"), chip.className).toBe(true);
                widths.add(chip.style.width);
            }
            // 全表同一个宽度 —— 跨行也齐。
            expect(widths.size).toBe(1);
        } finally {
            cleanup();
        }
    });

    it("chip 与追加按钮都走设计系统原语", () => {
        /*
         * 【回归背景】这一行控件曾经直用 Radix 的 `Button`/`IconButton`，还带写死的
         * 调色板色（录入中蓝、自定义绿）与内联 `fontFamily: monospace`。项目里
         * `AppButton`/`AppIconButton` 是按钮外观的唯一来源，且"自定义绑定"已经由
         * 分组标题的提示说明，不需要再上一次色。
         */
        const { cleanup } = renderDialog();
        try {
            const chips = Array.from(
                document.body.querySelectorAll<HTMLElement>("[data-hs-kb-slot]"),
            );
            expect(chips.length).toBeGreaterThan(0);
            for (const chip of chips) {
                expect(chip.className, chip.textContent ?? "").toContain("app-button");
                expect(chip.style.fontFamily, "字体走 hs-type-mono 类").toBe("");
            }
            const addButtons = Array.from(
                document.body.querySelectorAll<HTMLElement>("[data-hs-kb-add]"),
            );
            expect(addButtons.length).toBeGreaterThan(0);
            for (const add of addButtons) {
                expect(add.className).toContain("app-icon-button");
            }
        } finally {
            cleanup();
        }
    });
});

/**
 * 追加槽位在**每一行**都占位 —— 搜索把常规快捷键与修饰键手势混排时，
 * chip 列必须仍然对齐。
 *
 * 【回归背景】修饰键手势不能绑多个键，因此它那一行不画追加按钮；此前连槽位
 * 也一起省掉，于是它的 chip 比其他行往右多出「按钮 + 间距」那一段，混排时
 * 参差不齐。现在槽位无条件渲染，按钮画不画由能力决定。
 */
describe("追加槽位：每行占位，按钮可选", () => {
    it("每一行都恰好有一个追加槽位", () => {
        const { cleanup } = renderDialog();
        try {
            const rows = document.body.querySelectorAll("[data-hs-kb-row]");
            expect(rows.length).toBeGreaterThan(0);
            for (const row of rows) {
                expect(row.querySelectorAll("[data-hs-kb-add-slot]")).toHaveLength(1);
            }
        } finally {
            cleanup();
        }
    });

    it("修饰键手势行有槽位但没有按钮；常规行两者都有", () => {
        const { cleanup } = renderDialog();
        try {
            const modifierRows = Array.from(
                document.body.querySelectorAll<HTMLElement>('[data-hs-kb-row="modClip"]'),
            );
            expect(modifierRows.length).toBeGreaterThan(0);
            for (const row of modifierRows) {
                expect(row.querySelector("[data-hs-kb-add-slot]")).not.toBeNull();
                expect(row.querySelector("[data-hs-kb-add]")).toBeNull();
            }

            const regularRows = Array.from(
                document.body.querySelectorAll<HTMLElement>('[data-hs-kb-row="edit"]'),
            );
            expect(regularRows.length).toBeGreaterThan(0);
            for (const row of regularRows) {
                expect(row.querySelector("[data-hs-kb-add-slot]")).not.toBeNull();
                expect(row.querySelector("[data-hs-kb-add]")).not.toBeNull();
            }
        } finally {
            cleanup();
        }
    });
});
