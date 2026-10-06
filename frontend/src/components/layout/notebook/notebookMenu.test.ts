/*
 * 记事本右键菜单**内容**的规则测试。
 *
 * 全部走纯数据：给一个目标与一组开关，断言菜单项数组。不渲染、不建编辑器 ——
 * 这正是把 builder 抽成纯函数的目的（与 `fileBrowserMenu.test.ts` 同一套）。
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

/** 面板会提供的全部动作（NodeView 只提供子集，见 `Phase F`）。 */
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

const keys = (items: AppMenuItemSpec[]): string[] => items.map((item) => item.key);
const find = (items: AppMenuItemSpec[], key: string): AppMenuItemSpec | undefined =>
    items.find((item) => item.key === key);

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

    it("项 key 全局唯一（同一张菜单里不得出现两条同名项）", () => {
        for (const combo of combos) {
            const list = keys(build(combo));
            expect(
                list.length === new Set(list).size
                    ? []
                    : [`重复 key：${list.join(", ")}`, JSON.stringify(combo)],
            ).toEqual([]);
        }
    });

    it("首项不带分隔线（分隔线是组边界，菜单顶部没有边界）", () => {
        for (const combo of combos) {
            const items = build(combo);
            if (items.length === 0) continue;
            expect(items[0].separatorBefore, JSON.stringify(combo)).toBeFalsy();
        }
    });

    it("组边界两侧都有分隔线（分隔线挂在组首项上）", () => {
        const items = build({});
        // 选择组之后紧接格式组：两段各自带一条，中间隔着 selectAll。
        expect(find(items, "selectAll")?.separatorBefore).toBe(true);
        expect(find(items, "__group-format")?.separatorBefore).toBe(true);
        // 同一组内部不重复划线。
        expect(find(items, "italic")?.separatorBefore).toBeFalsy();
        expect(find(items, "heading2")?.separatorBefore).toBeFalsy();
    });

    it("分组标题行不带 onSelect（不可选中）", () => {
        for (const combo of combos) {
            for (const item of build(combo)) {
                if (item.heading) expect(item.onSelect).toBeUndefined();
            }
        }
    });
});

describe("buildNotebookContextMenu —— 动作缺失即不出现", () => {
    it("NodeView 只给图片动作时，菜单里没有撤销 / 格式 / 插入", () => {
        const items = build({
            target: { kind: "image" },
            actions: { imageCopy: () => {}, imageSaveAs: () => {} },
        });
        expect(keys(items)).toEqual(["imageCopy", "imageSaveAs"]);
    });

    it("面板给全动作时，文本落点出现完整分组", () => {
        const list = keys(build({}));
        expect(list).toContain("cut");
        expect(list).toContain("bold");
        expect(list).toContain("heading1");
        expect(list).toContain("insertTimecode");
        expect(list).toContain("find");
    });
});

describe("buildNotebookContextMenu —— 剪贴板可用性", () => {
    it("空选区时剪切 / 复制 / 复制为… 置灰并给出原因", () => {
        const items = build({ flags: { selectionEmpty: true } });
        for (const key of ["cut", "copy", "copyMarkdown", "copyPlain"]) {
            const item = find(items, key);
            expect(item, key).toBeDefined();
            expect(item?.disabled, key).toBe(true);
            expect(item?.tooltip, key).toBe("notebook_ctx_need_selection");
        }
    });

    it("有选区时这些项可用且没有 tooltip", () => {
        const items = build({ flags: { selectionEmpty: false } });
        for (const key of ["cut", "copy", "copyMarkdown", "copyPlain"]) {
            const item = find(items, key);
            expect(item?.disabled, key).toBe(false);
            expect(item?.tooltip, key).toBeUndefined();
        }
    });

    it("历史为空时撤销 / 重做置灰", () => {
        const items = build({ flags: { canUndo: false, canRedo: false } });
        expect(find(items, "undo")?.disabled).toBe(true);
        expect(find(items, "undo")?.tooltip).toBe("notebook_ctx_nothing_to_undo");
        expect(find(items, "redo")?.tooltip).toBe("notebook_ctx_nothing_to_redo");
    });

    it("清除格式仅在确实有格式时可用", () => {
        expect(find(build({ flags: { hasMarks: false } }), "clearFormatting")?.disabled).toBe(true);
        expect(find(build({ flags: { hasMarks: true } }), "clearFormatting")?.disabled).toBe(false);
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
        expect(find(items, "insertClipReference")?.tooltip).toBe("notebook_clip_ref_none");
    });
});

describe("buildNotebookContextMenu —— 只读预览", () => {
    const items = build({ surface: "preview", flags: { editable: false } });

    it("保留复制与选择，去掉一切写入型动作", () => {
        const list = keys(items);
        expect(list).toContain("copy");
        expect(list).toContain("copyMarkdown");
        expect(list).toContain("selectAll");
        for (const key of ["cut", "paste", "undo", "redo", "bold", "heading1", "insertImage"]) {
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
        expect(keys(table)).not.toContain("tableDeleteRow");
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

    it("表格：八项齐全，标题行按落点勾选", () => {
        const items = build({ target: { kind: "table", inHeaderRow: true } });
        for (const key of [
            "tableRowAbove",
            "tableRowBelow",
            "tableColLeft",
            "tableColRight",
            "tableDeleteRow",
            "tableDeleteCol",
            "tableDelete",
            "tableToggleHeader",
        ]) {
            expect(keys(items), key).toContain(key);
        }
        expect(find(items, "tableToggleHeader")?.checked).toBe(true);
        expect(find(items, "tableDelete")?.danger).toBe(true);
    });

    it("列表：缩进 / 凸排出现在最前，且块分组不再重复它们", () => {
        const items = build({ target: { kind: "list", itemType: "listItem" } });
        expect(keys(items).slice(0, 2)).toEqual(["indent", "outdent"]);
        expect(keys(items).filter((key) => key === "indent")).toHaveLength(1);
        expect(keys(items).filter((key) => key === "outdent")).toHaveLength(1);
    });

    it("非列表落点：缩进 / 凸排留在块分组里", () => {
        const items = build({ target: { kind: "text" } });
        expect(keys(items)).toContain("indent");
        expect(keys(items)).toContain("outdent");
    });

    it("暂存块：不发通用『复制为 Markdown』（避免与块自带那条撞标签）", () => {
        const items = build({ target: { kind: "clip", param: false } });
        const list = keys(items);
        expect(list).toContain("clipCopyMarkdown");
        expect(list).not.toContain("copyMarkdown");
        expect(list).not.toContain("copyPlain");
    });

    it("参数线载荷的主操作说『应用到参数编辑器』，不说『插入到时间轴』", () => {
        const param = find(build({ target: { kind: "clip", param: true } }), "clipInsert");
        expect(param?.label).toBe("notebook_clip_apply_to_param");
        const timeline = find(build({ target: { kind: "clip", param: false } }), "clipInsert");
        expect(timeline?.label).toBe("notebook_clip_insert_timeline");
    });

    it("图片：不落点专属项时也不会出现图片组", () => {
        expect(keys(build({ target: { kind: "text" } }))).not.toContain("imageCopy");
    });
});

describe("buildNotebookContextMenu —— 分组与详略", () => {
    it("full 档含三个分组标题，且每组标题都在自己的项之前", () => {
        const items = build({});
        const list = keys(items);
        expect(list).toContain("__group-format");
        expect(list).toContain("__group-block");
        expect(list).toContain("__group-insert");
        expect(list.indexOf("__group-format")).toBeLessThan(list.indexOf("bold"));
        expect(list.indexOf("__group-block")).toBeLessThan(list.indexOf("heading1"));
        expect(list.indexOf("__group-insert")).toBeLessThan(list.indexOf("insertImage"));
    });

    it("compact 档去掉格式 / 段落 / 插入三组，也去掉粘贴的三种变体", () => {
        const items = build({ scope: "compact" });
        const list = keys(items);
        for (const key of [
            "__group-format",
            "__group-block",
            "__group-insert",
            "bold",
            "heading1",
            "insertImage",
            "pastePlain",
            "pasteMarkdown",
            "pasteImage",
            "stageClipboard",
        ]) {
            expect(list, key).not.toContain(key);
        }
        // 核心项一个不少。
        for (const key of ["cut", "copy", "paste", "selectAll", "find", "undo"]) {
            expect(list, key).toContain(key);
        }
    });

    it("compact 档保留落点专属项（那才是右键的价值）", () => {
        const items = build({
            scope: "compact",
            target: { kind: "link", href: "https://example.com", internal: false },
        });
        expect(keys(items)).toContain("linkOpen");
    });

    it("组内项全部缺失时，连分组标题一起省掉", () => {
        const items = build({
            actions: { cut: () => {}, copy: () => {}, selectAll: () => {}, find: () => {} },
        });
        expect(keys(items)).not.toContain("__group-format");
        expect(keys(items)).not.toContain("__group-block");
        expect(keys(items)).not.toContain("__group-insert");
    });
});

describe("buildNotebookContextMenu —— 源码视图", () => {
    it("只有撤销 / 剪贴板 / 选择 / 查找", () => {
        const items = build({ surface: "source" });
        expect(keys(items)).toEqual(["undo", "redo", "cut", "copy", "paste", "selectAll", "find"]);
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
            ["bold", "B"],
            ["italic", "I"],
            ["find", "F"],
        ] as const) {
            const item = find(items, key);
            expect(item?.shortcut, key).toBeTruthy();
            expect(item?.shortcut?.endsWith(shortcut), `${key} → ${item?.shortcut}`).toBe(true);
        }
    });

    it("无默认键位的动作不占快捷键列", () => {
        expect(find(build({}), "copyMarkdown")?.shortcut).toBeUndefined();
        expect(find(build({}), "pastePlain")?.shortcut).toBeUndefined();
    });
});
