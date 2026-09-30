/**
 * `keybindingSearch` 的单元测试 —— 检索索引的构成、AND 语义、结果的稳定排序。
 *
 * 【为什么用恒等解析而不是真实词典】纯 ASCII 的假文案让断言能写成字面量，且不受
 * 真实五语系词典改动的影响。这不是偷懒：索引的职责是「把哪些文本纳入检索」，
 * 与词典里写的是哪国语言无关。
 */
import { describe, expect, it } from "vitest";

import {
    buildKeybindingSearchEntries,
    matchKeybindingEntries,
    type KeybindingSearchEntry,
} from "./keybindingSearch";
import type { ActionId } from "./types";

/** 恒等解析：让每个 labelKey 用它自己当文案，索引里就是纯 ASCII 的键名片段。 */
const identity = (key: string) => key;

const ENTRIES = buildKeybindingSearchEntries(identity);

function idsOf(entries: KeybindingSearchEntry[]): ActionId[] {
    return entries.map((entry) => entry.id);
}

describe("buildKeybindingSearchEntries — 词条构成", () => {
    it("覆盖全部动作，且每条都有可检索的词条", () => {
        expect(ENTRIES.length).toBeGreaterThan(100);
        for (const entry of ENTRIES) {
            const total =
                entry.sources.label.length +
                entry.sources.group.length +
                entry.sources.idParts.length +
                entry.sources.primaryKeys.length;
            expect(total, `${entry.id} 没有可检索词条`).toBeGreaterThan(0);
        }
    });

    it("操作名与分组名各自独立成源", () => {
        const entry = ENTRIES.find((item) => item.id === "clip.delete")!;
        // 恒等解析下 labelKey 本身就是 "kb_clip_delete"。
        expect(entry.sources.label).toContain("kb_clip_delete");
        expect(entry.sources.group.length).toBeGreaterThan(0);
    });

    it("修饰键同时产出两套叫法（ctrl 与 cmd 都能搜到 ⌘/Ctrl 绑定）", () => {
        const entry = ENTRIES.find((item) => item.id === "edit.undo")!;
        expect(entry.sources.modifiers).toContain("ctrl");
        expect(entry.sources.modifiers).toContain("control");
        // macOS 上同一个 ctrl 字段渲染成 ⌘：cmd / command 也要能命中它。
        expect(entry.sources.modifiers).toContain("cmd");
        expect(entry.sources.modifiers).toContain("command");
        expect(entry.sources.primaryKeys).toContain("z");
    });

    it("纯修饰键手势的 key 归入修饰键（不与真正的主键同档）", () => {
        const entry = ENTRIES.find((item) => item.id === "modifier.clipStretch")!;
        expect(entry.sources.modifiers).toContain("alt");
        expect(entry.sources.modifiers).toContain("option");
        // 它的 key 就是 alt —— 不能在 primaryKeys 里冒充主键拿到最高分。
        expect(entry.sources.primaryKeys).not.toContain("alt");
    });

    it("方向键补上 up / arrow 别名（UI 里它显示成 ↑）", () => {
        const keys = ENTRIES.find((item) => item.id === "track.selectUp")!.sources.primaryKeys;
        expect(keys).toContain("arrowup");
        expect(keys).toContain("up");
        expect(keys).toContain("arrow");
    });

    it("escape 补上 esc 别名", () => {
        const keys = ENTRIES.find((item) => item.id === "quickSearch.close")!.sources.primaryKeys;
        expect(keys).toContain("esc");
    });

    it("动作 id 的片段进入 idParts（clip.split → clip / split）", () => {
        const entry = ENTRIES.find((item) => item.id === "clip.split")!;
        expect(entry.sources.idParts).toEqual(["clip", "split"]);
    });

    it("无绑定的动作不产生 __none__ 按键词条", () => {
        const entry = ENTRIES.find((item) => item.id === "timeline.zoomIn")!;
        for (const source of Object.values(entry.sources)) {
            expect(source).not.toContain("__none__");
        }
    });
});

describe("matchKeybindingEntries — 分词与 AND 语义", () => {
    it("'ctrl z' 命中 Ctrl+Z（分 token 后各自命中）", () => {
        expect(idsOf(matchKeybindingEntries(ENTRIES, "ctrl z"))).toContain("edit.undo");
    });

    it("'ctrl+z' 与 'ctrl z' 等价（+ 是分隔符）", () => {
        const withSpace = idsOf(matchKeybindingEntries(ENTRIES, "ctrl z"));
        const withPlus = idsOf(matchKeybindingEntries(ENTRIES, "ctrl+z"));
        expect(withPlus).toEqual(withSpace);
    });

    it("逗号也是分隔符", () => {
        expect(idsOf(matchKeybindingEntries(ENTRIES, "ctrl,z"))).toContain("edit.undo");
    });

    it("大小写不敏感", () => {
        expect(idsOf(matchKeybindingEntries(ENTRIES, "CTRL Z"))).toEqual(
            idsOf(matchKeybindingEntries(ENTRIES, "ctrl z")),
        );
    });

    it("空查询返回全量且保持原顺序", () => {
        for (const blank of ["", "   ", "\t"]) {
            expect(idsOf(matchKeybindingEntries(ENTRIES, blank))).toEqual(idsOf(ENTRIES));
        }
    });

    it("单个不存在的 token 使整条不命中（AND 而非 OR）", () => {
        const hits = idsOf(matchKeybindingEntries(ENTRIES, "undo zzzznope"));
        expect(hits).toEqual([]);
    });

    it("AND 把 Ctrl 系绑缩小到远少于全部", () => {
        const ctrlOnly = matchKeybindingEntries(ENTRIES, "ctrl");
        const ctrlZ = matchKeybindingEntries(ENTRIES, "ctrl z");
        expect(ctrlZ.length).toBeGreaterThan(0);
        expect(ctrlZ.length).toBeLessThan(ctrlOnly.length);
    });

    it("方向键可以用 up 搜到（真实 UI 里它显示成 ↑）", () => {
        expect(idsOf(matchKeybindingEntries(ENTRIES, "up"))).toContain("track.selectUp");
    });
});

describe("matchKeybindingEntries — 排序与稳定性", () => {
    it("操作名前缀命中排在按键命中之前", () => {
        // "delete" 既是 clip.delete 的名字、也是它的按键；这里验证存在明确的序：
        // 名字以查询串开头的条目不应被仅"按键含查询串"的条目压下。
        const hits = matchKeybindingEntries(ENTRIES, "clip_delete");
        expect(hits.length).toBeGreaterThan(0);
        expect(hits[0].id).toBe("clip.delete");
    });

    it("同分保持原有分组顺序（结果不跳动）", () => {
        const first = idsOf(matchKeybindingEntries(ENTRIES, "v"));
        const second = idsOf(matchKeybindingEntries(ENTRIES, "v"));
        expect(second).toEqual(first);
    });

    it("名称里恰好含该字母的条目不会压过真正绑在该键上的条目", () => {
        /*
         * 【回归】用恒等解析时标签就是 `kb_*` 形式的键名，`modifier.pianoRollVerticalZoom`
         * 的标签含 "zoom"、id 片段含 z、分组名含 z —— 在**每个来源**都泛泛沾边。
         * 而 `edit.undo` 的名称 "undo" 里根本没有 z，它的强信号只有绑定本身。
         * 主键取最高档（4 > 名称 2/3）后，绑在 Z 上的必须排第一。
         */
        const hits = matchKeybindingEntries(ENTRIES, "ctrl z");
        expect(hits[0].id).toBe("edit.undo");
        const zoomIndex = hits.findIndex((e) => e.id === "modifier.pianoRollVerticalZoom");
        expect(zoomIndex).toBeGreaterThan(0);
    });

    it("操作名含查询串的（2 分）排在仅 id 片段命中的（1 分）之前", () => {
        const hits = matchKeybindingEntries(ENTRIES, "clip");
        // label 是 "kb_clip_*"：名字里含 clip 的是 2 分；名字里没有、只有 id 片段
        // "clip" 命中的（例如 group 名带 clip 的修饰键动作）是 1 分。
        const strong = hits.filter((entry) => entry.label.toLowerCase().includes("clip"));
        const weak = hits.filter((entry) => !entry.label.toLowerCase().includes("clip"));
        expect(strong.length).toBeGreaterThan(0);
        expect(weak.length).toBeGreaterThan(0);
        const firstWeakIndex = hits.indexOf(weak[0]);
        for (const entry of strong) {
            expect(hits.indexOf(entry)).toBeLessThan(firstWeakIndex);
        }
    });

    it("多次调用不修改传入数组", () => {
        const snapshot = idsOf(ENTRIES);
        matchKeybindingEntries(ENTRIES, "ctrl z");
        expect(idsOf(ENTRIES)).toEqual(snapshot);
    });
});
