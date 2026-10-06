/*
 * 记事本右键菜单**内容**的规则测试。
 *
 * 全部走纯数据：给一个目标与一组开关，断言菜单项数组。不渲染、不建编辑器 ——
 * 这正是把 builder 抽成纯函数的目的（与 `fileBrowserMenu.test.ts` 同一套）。
 *
 * 【层级也是被断言的对象】这张菜单曾经平铺到 30 行（表格落点 38 行），接近满屏高。
 * 因此这里不只断言"有哪些项"，还断言**顶层有哪几项、哪些项在子菜单里** ——
 * 否则下一次往里加东西时，没人拦得住它重新长回去。
 */

import { describe, expect, it } from "vitest";

import type { MessageKey } from "../../../i18n/messages";
import type { AppMenuItemSpec } from "../../../ui";
import {
    buildNotebookContextMenu,
    type NotebookMenuActionKey,
    type NotebookMenuActions,
    type NotebookMenuContext,
    type NotebookMenuScope,
    type NotebookMenuSurface,
} from "./notebookMenu";
import type {
    NotebookContext,
    NotebookMenuFlags,
    NotebookMenuTarget,
} from "./notebookContextTarget";

/** 词典直通：断言时直接看键名，省去与文案耦合。 */
const translate = (key: MessageKey): string => key;

const BASE_FLAGS: NotebookMenuFlags = {
    selectionEmpty: false,
    editable: true,
    canUndo: true,
    canRedo: true,
    isBold: false,
    isItalic: false,
    isStrike: false,
    isCode: false,
    hasMarks: false,
    headingLevel: null,
    canIndent: true,
};

/** 面板会提供的全部动作（NodeView 只提供子集）。 */
const ALL_ACTIONS: NotebookMenuActionKey[] = [
    "undo",
    "redo",
    "cut",
    "copy",
    "copyMarkdown",
    "copyPlain",
    "paste",
    "pastePlain",
    "pasteMarkdown",
    "pasteImage",
    "stageClipboard",
    "selectAll",
    "bold",
    "italic",
    "strike",
    "code",
    "clearFormatting",
    "heading1",
    "heading2",
    "heading3",
    "paragraph",
    "indent",
    "outdent",
    "linkOpen",
    "linkCopy",
    "linkEdit",
    "linkRemove",
    "tableRowAbove",
    "tableRowBelow",
    "tableColLeft",
    "tableColRight",
    "tableDeleteRow",
    "tableDeleteCol",
    "tableDelete",
    "tableToggleHeader",
    "insertImage",
    "insertTable",
    "insertRule",
    "insertTimecode",
    "insertClipReference",
    "insertProjectInfo",
    "find",
    "imageCopy",
    "imageSaveAs",
    "imageWidthReset",
    "imageEditAlt",
    "imageRemove",
    "clipRestore",
    "clipInsert",
    "clipInsertNewTracks",
    "clipRename",
    "clipCopyMarkdown",
    "clipSavePayload",
    "clipRemove",
];

function allActions(): NotebookMenuActions {
    const actions: NotebookMenuActions = {};
    for (const key of ALL_ACTIONS) actions[key] = () => {};
    return actions;
}

function build(options: {
    target?: NotebookMenuTarget;
    flags?: Partial<NotebookMenuFlags>;
    surface?: NotebookMenuSurface;
    scope?: NotebookMenuScope;
    actions?: NotebookMenuActions;
}): AppMenuItemSpec[] {
    const context: NotebookContext = {
        target: options.target ?? { kind: "text" },
        flags: { ...BASE_FLAGS, ...options.flags },
    };
    const ctx: NotebookMenuContext = {
        surface: options.surface ?? "rich",
        scope: options.scope ?? "full",
        context,
        translate,
        actions: options.actions ?? allActions(),
    };
    return buildNotebookContextMenu(ctx);
}

/** 本层的 key（不含子菜单）。 */
const keys = (items: AppMenuItemSpec[]): string[] => items.map((item) => item.key);

/** 递归展平：本层 + 所有子菜单。 */
function flatten(items: AppMenuItemSpec[]): AppMenuItemSpec[] {
    return items.flatMap((item) => [item, ...(item.items ? flatten(item.items) : [])]);
}

/** 每一层各自的项数组（用于"逐层"断言不变式）。 */
function layers(items: AppMenuItemSpec[]): AppMenuItemSpec[][] {
    return [items, ...items.flatMap((item) => (item.items ? layers(item.items) : []))];
}

/** 按 key 找**本层**的项。 */
const findTop = (items: AppMenuItemSpec[], key: string): AppMenuItemSpec | undefined =>
    items.find((item) => item.key === key);

/** 按 key 递归找（子菜单里的也算）。 */
const findAny = (items: AppMenuItemSpec[], key: string): AppMenuItemSpec | undefined =>
    flatten(items).find((item) => item.key === key);

/** 取某个子菜单的子项（找不到或不是子菜单就抛，让断言更早失败）。 */
function childrenOf(items: AppMenuItemSpec[], key: string): AppMenuItemSpec[] {
    const item = findTop(items, key);
    if (!item) throw new Error(`顶层没有这一项：${key}`);
    if (!item.items) throw new Error(`这一项不是子菜单：${key}`);
    return item.items;
}

describe("buildNotebookContextMenu —— 通用不变式", () => {
    const targets: NotebookMenuTarget[] = [
        { kind: "text" },
        { kind: "empty" },
        { kind: "link", href: "https://example.com", internal: false },
        { kind: "link", href: "hifi://seek/12.5", internal: true },
        { kind: "table", inHeaderRow: false },
        { kind: "list", itemType: "listItem" },
        { kind: "list", itemType: "taskItem" },
        { kind: "image" },
        { kind: "clip", param: false },
    ];
    const combos: Array<{
        surface: NotebookMenuSurface;
        scope: NotebookMenuScope;
        target: NotebookMenuTarget;
        flags: Partial<NotebookMenuFlags>;
    }> = [];
    for (const surface of ["rich", "source", "preview"] as const) {
        for (const scope of ["full", "compact"] as const) {
            for (const target of targets) {
                for (const flags of [{}, { selectionEmpty: true }, { editable: false }]) {
                    combos.push({ surface, scope, target, flags });
                }
            }
        }
    }

    it("项 key 全局唯一（含子菜单：同一张菜单里不得出现两条同名项）", () => {
        for (const combo of combos) {
            const list = flatten(build(combo)).map((item) => item.key);
            expect(
                list.length === new Set(list).size
                    ? []
                    : [`重复 key：${list.join(", ")}`, JSON.stringify(combo)],
            ).toEqual([]);
        }
    });

    it("每一层的首项都不带分隔线（分隔线是组边界，层顶没有边界）", () => {
        for (const combo of combos) {
            for (const layer of layers(build(combo))) {
                if (layer.length === 0) continue;
                expect(layer[0].separatorBefore, JSON.stringify(combo)).toBeFalsy();
            }
        }
    });

    it("子菜单一定有子项（不留一个点开才发现是空的触发项）", () => {
        for (const combo of combos) {
            for (const item of flatten(build(combo))) {
                if (item.items === undefined) continue;
                expect(item.items.length, `${item.key} 是空的子菜单`).toBeGreaterThan(0);
            }
        }
    });

    it("子菜单触发项只负责展开：不带 onSelect / shortcut / danger / checked", () => {
        for (const combo of combos) {
            for (const item of flatten(build(combo))) {
                if (!item.items) continue;
                expect(item.onSelect, item.key).toBeUndefined();
                expect(item.shortcut, item.key).toBeUndefined();
                expect(item.danger, item.key).toBeUndefined();
                expect(item.checked, item.key).toBeUndefined();
            }
        }
    });
});

describe("buildNotebookContextMenu —— 层级（防止菜单重新长回去）", () => {
    it("正文落点：顶层恰好这十二项，长枚举全部收进子菜单", () => {
        expect(keys(build({}))).toEqual([
            "undo",
            "redo",
            "cut",
            "copy",
            "copy-as",
            "paste",
            "paste-as",
            "selectAll",
            "format",
            "block",
            "insert",
            "find",
        ]);
    });

    it("表格落点：行 / 列各自成组，顶层十六项（平铺时是三十八项）", () => {
        const items = build({ target: { kind: "table", inHeaderRow: false } });
        expect(keys(items)).toEqual([
            "table-rows",
            "table-cols",
            "tableToggleHeader",
            "tableDelete",
            "undo",
            "redo",
            "cut",
            "copy",
            "copy-as",
            "paste",
            "paste-as",
            "selectAll",
            "format",
            "block",
            "insert",
            "find",
        ]);
        expect(keys(childrenOf(items, "table-rows"))).toEqual([
            "tableRowAbove",
            "tableRowBelow",
            "tableDeleteRow",
        ]);
        expect(keys(childrenOf(items, "table-cols"))).toEqual([
            "tableColLeft",
            "tableColRight",
            "tableDeleteCol",
        ]);
    });

    it("最常点的三条留在顶层，不藏进子菜单", () => {
        const list = keys(build({}));
        for (const key of ["cut", "copy", "paste"]) {
            expect(list, key).toContain(key);
        }
        // 反向：它们不该同时出现在任何子菜单里（那会变成两份）。
        expect(flatten(build({})).filter((item) => item.key === "cut")).toHaveLength(1);
    });

    it("格式 / 段落 / 插入三个子菜单的内容与顺序", () => {
        const items = build({});
        expect(keys(childrenOf(items, "format"))).toEqual([
            "bold",
            "italic",
            "strike",
            "code",
            "clearFormatting",
        ]);
        expect(keys(childrenOf(items, "block"))).toEqual([
            "heading1",
            "heading2",
            "heading3",
            "paragraph",
            "indent",
            "outdent",
        ]);
        expect(keys(childrenOf(items, "insert"))).toEqual([
            "insertImage",
            "insertTable",
            "insertRule",
            "insertTimecode",
            "insertClipReference",
            "insertProjectInfo",
            "stageClipboard",
        ]);
    });

    it("复制为 / 粘贴为：变体在子菜单里，主干在顶层", () => {
        const items = build({});
        expect(keys(childrenOf(items, "copy-as"))).toEqual(["copyMarkdown", "copyPlain"]);
        expect(keys(childrenOf(items, "paste-as"))).toEqual([
            "pastePlain",
            "pasteMarkdown",
            "pasteImage",
        ]);
        // 主干两条在顶层。
        expect(keys(items)).toContain("copy");
        expect(keys(items)).toContain("paste");
    });

    it("子菜单内部的分隔线落在正确的组首项上", () => {
        const items = build({});
        expect(findTop(childrenOf(items, "format"), "clearFormatting")?.separatorBefore).toBe(true);
        expect(findTop(childrenOf(items, "format"), "italic")?.separatorBefore).toBeFalsy();
        expect(findTop(childrenOf(items, "insert"), "insertTimecode")?.separatorBefore).toBe(true);
        expect(findTop(childrenOf(items, "insert"), "stageClipboard")?.separatorBefore).toBe(true);
        expect(findTop(childrenOf(items, "copy-as"), "copyPlain")?.separatorBefore).toBeFalsy();
    });
});

describe("buildNotebookContextMenu —— 动作缺失即不出现", () => {
    it("NodeView 只给图片动作时，菜单里只有图片项", () => {
        const items = build({
            target: { kind: "image" },
            actions: { imageCopy: () => {}, imageSaveAs: () => {} },
        });
        expect(keys(items)).toEqual(["imageCopy", "imageSaveAs"]);
    });

    it("整组动作都缺时，连子菜单触发项一起省掉", () => {
        const items = build({
            actions: { cut: () => {}, copy: () => {}, selectAll: () => {}, find: () => {} },
        });
        const list = keys(items);
        expect(list).not.toContain("format");
        expect(list).not.toContain("block");
        expect(list).not.toContain("insert");
        expect(list).not.toContain("copy-as");
        expect(list).not.toContain("paste-as");
    });

    it("子菜单里部分动作缺失时，只少那一项，触发项仍在", () => {
        const items = build({
            actions: {
                bold: () => {},
                italic: () => {},
                clearFormatting: () => {},
            },
        });
        expect(keys(items)).toContain("format");
        expect(keys(childrenOf(items, "format"))).toEqual(["bold", "italic", "clearFormatting"]);
    });
});

describe("buildNotebookContextMenu —— 剪贴板可用性", () => {
    it("空选区时剪切 / 复制 / 复制为… 置灰并给出原因", () => {
        const items = build({ flags: { selectionEmpty: true } });
        for (const key of ["cut", "copy"]) {
            const item = findTop(items, key);
            expect(item?.disabled, key).toBe(true);
            expect(item?.tooltip, key).toBe("notebook_ctx_need_selection");
        }
        for (const key of ["copyMarkdown", "copyPlain"]) {
            const item = findAny(items, key);
            expect(item?.disabled, key).toBe(true);
            expect(item?.tooltip, key).toBe("notebook_ctx_need_selection");
        }
    });

    it("有选区时这些项可用且没有 tooltip", () => {
        const items = build({ flags: { selectionEmpty: false } });
        for (const key of ["cut", "copy", "copyMarkdown", "copyPlain"]) {
            const item = findAny(items, key);
            expect(item?.disabled, key).toBe(false);
            expect(item?.tooltip, key).toBeUndefined();
        }
    });

    it("历史为空时撤销 / 重做置灰", () => {
        const items = build({ flags: { canUndo: false, canRedo: false } });
        expect(findTop(items, "undo")?.disabled).toBe(true);
        expect(findTop(items, "undo")?.tooltip).toBe("notebook_ctx_nothing_to_undo");
        expect(findTop(items, "redo")?.tooltip).toBe("notebook_ctx_nothing_to_redo");
    });

    it("清除格式仅在确实有格式时可用", () => {
        const without = build({ flags: { hasMarks: false } });
        expect(findAny(without, "clearFormatting")?.disabled).toBe(true);
        const withMarks = build({ flags: { hasMarks: true } });
        expect(findAny(withMarks, "clearFormatting")?.disabled).toBe(false);
    });

    it("调用方给的置灰原因优先于 builder 的通用规则", () => {
        const items = buildNotebookContextMenu({
            surface: "rich",
            scope: "full",
            context: { target: { kind: "text" }, flags: BASE_FLAGS },
            translate,
            actions: { insertClipReference: () => {} },
            disabled: { insertClipReference: "notebook_clip_ref_none" },
        });
        expect(findAny(items, "insertClipReference")?.tooltip).toBe("notebook_clip_ref_none");
    });
});

describe("buildNotebookContextMenu —— 只读预览", () => {
    const items = build({ surface: "preview", flags: { editable: false } });

    it("保留复制与选择，去掉一切写入型动作", () => {
        const list = flatten(items).map((item) => item.key);
        expect(list).toContain("copy");
        expect(list).toContain("copyMarkdown");
        expect(list).toContain("selectAll");
        for (const key of [
            "cut",
            "paste",
            "undo",
            "redo",
            "bold",
            "heading1",
            "insertImage",
            "format",
            "block",
            "insert",
        ]) {
            expect(list, key).not.toContain(key);
        }
    });

    it("链接保留打开与复制地址，去掉编辑与移除", () => {
        const link = build({
            surface: "preview",
            flags: { editable: false },
            target: { kind: "link", href: "https://example.com", internal: false },
        });
        const list = keys(link);
        expect(list).toContain("linkOpen");
        expect(list).toContain("linkCopy");
        expect(list).not.toContain("linkEdit");
        expect(list).not.toContain("linkRemove");
    });

    it("表格 / 列表落点整组省略（不置灰六项让人以为是暂时的）", () => {
        const table = build({
            surface: "preview",
            flags: { editable: false },
            target: { kind: "table", inHeaderRow: false },
        });
        expect(keys(table)).not.toContain("table-rows");
        const list = build({
            surface: "preview",
            flags: { editable: false },
            target: { kind: "list", itemType: "listItem" },
        });
        expect(keys(list)).not.toContain("indent");
    });
});

describe("buildNotebookContextMenu —— 落点专属组", () => {
    it("链接：四项齐全，内部链接也照常给出（打开由调用方分流）", () => {
        const items = build({
            target: { kind: "link", href: "hifi://seek/12.5", internal: true },
        });
        expect(keys(items).slice(0, 4)).toEqual(["linkOpen", "linkCopy", "linkEdit", "linkRemove"]);
    });

    it("表格：标题行按落点勾选，删除表格是危险项", () => {
        const items = build({ target: { kind: "table", inHeaderRow: true } });
        expect(findTop(items, "tableToggleHeader")?.checked).toBe(true);
        expect(findTop(items, "tableDelete")?.danger).toBe(true);
    });

    it("列表：缩进 / 凸排出现在最前，且块子菜单不再重复它们", () => {
        const items = build({ target: { kind: "list", itemType: "listItem" } });
        expect(keys(items).slice(0, 2)).toEqual(["indent", "outdent"]);
        expect(flatten(items).filter((item) => item.key === "indent")).toHaveLength(1);
        expect(flatten(items).filter((item) => item.key === "outdent")).toHaveLength(1);
    });

    it("非列表落点：缩进 / 凸排留在块子菜单里", () => {
        expect(keys(childrenOf(build({ target: { kind: "text" } }), "block"))).toContain("indent");
    });

    it("暂存块：不发通用『复制为 Markdown』（避免与块自带那条撞标签）", () => {
        const items = build({ target: { kind: "clip", param: false } });
        const list = flatten(items).map((item) => item.key);
        expect(list).toContain("clipCopyMarkdown");
        expect(list).not.toContain("copyMarkdown");
        expect(list).not.toContain("copy-as");
    });

    it("参数线载荷的主操作说『应用到参数编辑器』，不说『插入到时间轴』", () => {
        const param = findTop(build({ target: { kind: "clip", param: true } }), "clipInsert");
        expect(param?.label).toBe("notebook_clip_apply_to_param");
        const timeline = findTop(build({ target: { kind: "clip", param: false } }), "clipInsert");
        expect(timeline?.label).toBe("notebook_clip_insert_timeline");
    });

    it("图片：不落点专属项时也不会出现图片组", () => {
        expect(keys(build({ target: { kind: "text" } }))).not.toContain("imageCopy");
    });
});

describe("buildNotebookContextMenu —— 详略档", () => {
    it("compact 档只留主干：七个顶层项", () => {
        const items = build({ scope: "compact" });
        expect(keys(items)).toEqual(["undo", "redo", "cut", "copy", "paste", "selectAll", "find"]);
    });

    it("compact 档保留落点专属项（那才是右键的价值）", () => {
        const items = build({
            scope: "compact",
            target: { kind: "link", href: "https://example.com", internal: false },
        });
        expect(keys(items)).toContain("linkOpen");
    });

    it("compact 档的表格落点仍然按轴分组", () => {
        const items = build({ scope: "compact", target: { kind: "table", inHeaderRow: false } });
        expect(keys(items).slice(0, 4)).toEqual([
            "table-rows",
            "table-cols",
            "tableToggleHeader",
            "tableDelete",
        ]);
    });
});

describe("buildNotebookContextMenu —— 源码视图", () => {
    it("只有撤销 / 剪贴板 / 选择 / 查找", () => {
        expect(keys(build({ surface: "source" }))).toEqual([
            "undo",
            "redo",
            "cut",
            "copy",
            "paste",
            "selectAll",
            "find",
        ]);
    });

    it("不随落点或详略变化（textarea 没有富文本语义）", () => {
        const compact = build({ surface: "source", scope: "compact" });
        expect(keys(compact)).toEqual(keys(build({ surface: "source" })));
    });
});

describe("buildNotebookContextMenu —— 快捷键提示", () => {
    it("编辑器自带键位都有提示，且格式为『修饰键+键』", () => {
        const items = build({});
        for (const [key, shortcut] of [
            ["undo", "Z"],
            ["cut", "X"],
            ["copy", "C"],
            ["paste", "V"],
            ["selectAll", "A"],
            ["find", "F"],
        ] as const) {
            const item = findAny(items, key);
            expect(item?.shortcut, key).toBeTruthy();
            expect(item?.shortcut?.endsWith(shortcut), `${key} → ${item?.shortcut}`).toBe(true);
        }
    });

    it("子菜单里的格式项也带提示（键盘用户看得见）", () => {
        const items = build({});
        expect(findAny(items, "bold")?.shortcut?.endsWith("B")).toBe(true);
        expect(findAny(items, "italic")?.shortcut?.endsWith("I")).toBe(true);
    });

    it("无默认键位的动作不占快捷键列", () => {
        expect(findAny(build({}), "copyMarkdown")?.shortcut).toBeUndefined();
        expect(findAny(build({}), "pastePlain")?.shortcut).toBeUndefined();
    });
});
