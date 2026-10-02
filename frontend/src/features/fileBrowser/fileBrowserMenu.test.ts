import { describe, expect, it, vi } from "vitest";

import type { FileEntry } from "../../services/api/fileBrowser";
import type { MessageKey } from "../../i18n/messages";
import {
    buildFileBrowserContextMenu,
    resolveMenuTargets,
    type FileBrowserMenuActions,
    type FileBrowserMenuContext,
} from "./fileBrowserMenu";
import { DEFAULT_FILE_BROWSER_VIEW_OPTIONS } from "./fileBrowserViewOptions";

function entry(name: string, isDir = false, extension?: string): FileEntry {
    return {
        name,
        path: isDir ? `D:\\music\\${name}` : `D:\\music\\${name}`,
        isDir,
        size: isDir ? null : 1024,
        extension: isDir ? null : (extension ?? name.split(".").pop() ?? null),
        modifiedTime: 1_700_000_000,
    };
}

const WAV = entry("take01.wav");
const WAV2 = entry("take02.wav");
const MID = entry("melody.mid");
const PROJ = entry("song.hshp");
const RPP = entry("song.rpp");
const TXT = entry("notes.txt");
const DIR = entry("子目录", true);

/** 全部动作都是 mock，便于断言"点哪一条调用了什么"。 */
function makeActions(): FileBrowserMenuActions {
    return {
        openEntry: vi.fn(),
        insertAtPlayhead: vi.fn(),
        insertOnNewTrack: vi.fn(),
        insertMultiple: vi.fn(),
        importFolder: vi.fn(),
        togglePreview: vi.fn(),
        reveal: vi.fn(),
        openWithDefaultApp: vi.fn(),
        copyPaths: vi.fn(),
        copyName: vi.fn(),
        openProject: vi.fn(),
        importProject: vi.fn(),
        importMidi: vi.fn(),
        openContainingFolder: vi.fn(),
        rename: vi.fn(),
        remove: vi.fn(),
        showProperties: vi.fn(),
        newFolder: vi.fn(),
        refresh: vi.fn(),
        openFolderDialog: vi.fn(),
        selectAll: vi.fn(),
        clearSelection: vi.fn(),
        setSortMode: vi.fn(),
        setSortDescending: vi.fn(),
        patchView: vi.fn(),
        openViewOptions: vi.fn(),
    };
}

function makeContext(overrides: Partial<FileBrowserMenuContext> = {}): FileBrowserMenuContext {
    return {
        // 测试只关心"哪些项存在"，标签直接用键名，断言时读起来就是键名。
        t: (key: MessageKey) => key,
        // 复数形式带上数量：这样能断言"数量确实被回填了" —— 只传 t 的写法
        // 会把 `{count}` 模板原样漏到界面上（此前正是如此）。
        plural: (key: MessageKey, count: number) => `${key}:${count}`,
        view: { ...DEFAULT_FILE_BROWSER_VIEW_OPTIONS },
        isComputerLevel: false,
        isSearchMode: false,
        previewingPath: null,
        selected: [],
        currentPath: "D:\\music",
        actions: makeActions(),
        ...overrides,
    };
}

function keys(items: ReturnType<typeof buildFileBrowserContextMenu>): string[] {
    return items.map((item) => item.key);
}

describe("resolveMenuTargets", () => {
    it("右键未选中项：只作用于它自己", () => {
        expect(resolveMenuTargets(WAV, [WAV2])).toEqual([WAV]);
    });

    it("右键已选中项：作用于整个选中集合（Explorer 语义）", () => {
        expect(resolveMenuTargets(WAV, [WAV, WAV2])).toEqual([WAV, WAV2]);
    });

    it("没有选中集合时只作用于点中的那一项", () => {
        expect(resolveMenuTargets(WAV, [])).toEqual([WAV]);
    });
});

describe("条目菜单", () => {
    it("音频文件：主操作是试听，并给出两种插入落点", () => {
        const items = buildFileBrowserContextMenu(WAV, makeContext());
        expect(keys(items)).toEqual(
            expect.arrayContaining([
                "primary",
                "insert-playhead",
                "insert-new-track",
                "preview",
                "reveal",
                "copy-paths",
                "copy-name",
                "open-default-app",
                "rename",
                "delete",
                "properties",
            ]),
        );
        // 单个文件不该出现"多文件插入"分组标题。
        expect(keys(items)).not.toContain("insert-heading");
    });

    it("正在试听的音频文件：主操作与试听项都变成「停止试听」", () => {
        const items = buildFileBrowserContextMenu(WAV, makeContext({ previewingPath: WAV.path }));
        expect(items.find((item) => item.key === "primary")?.label).toBe("fb_ctx_preview");
        expect(items.find((item) => item.key === "preview")?.label).toBe("fb_ctx_stop_preview");
    });

    it("目录：主操作是打开，没有插入与试听项", () => {
        const items = buildFileBrowserContextMenu(DIR, makeContext());
        expect(items.find((item) => item.key === "primary")?.label).toBe("fb_ctx_open");
        expect(keys(items)).not.toContain("insert-playhead");
        expect(keys(items)).not.toContain("preview");
    });

    it("目录：给出「导入文件夹」，且它打开的是选项对话框", () => {
        const actions = makeActions();
        const items = buildFileBrowserContextMenu(DIR, makeContext({ actions }));
        const item = items.find((candidate) => candidate.key === "import-folder");
        expect(item?.label).toBe("fb_ctx_import_folder");
        item?.onSelect?.();
        expect(actions.importFolder).toHaveBeenCalledWith([DIR]);
    });

    it("目录：没有三种排布方式的条目（目录走对话框，模式在里面选）", () => {
        const items = buildFileBrowserContextMenu(DIR, makeContext());
        expect(keys(items)).not.toContain("insert-across-time");
        expect(keys(items)).not.toContain("insert-as-takes");
    });

    it("文件 + 目录混选：两边的入口各自都在", () => {
        const items = buildFileBrowserContextMenu(WAV, makeContext({ selected: [WAV, DIR] }));
        expect(keys(items)).toContain("import-folder");
        expect(keys(items)).toContain("insert-across-time");
    });

    it("MIDI：主操作是导入 MIDI，不出现音频插入项", () => {
        const items = buildFileBrowserContextMenu(MID, makeContext());
        expect(items.find((item) => item.key === "primary")?.label).toBe("fb_ctx_import_midi");
        expect(keys(items)).not.toContain("insert-playhead");
    });

    it("工程文件：主操作是打开工程，并额外给出「导入工程」", () => {
        const items = buildFileBrowserContextMenu(PROJ, makeContext());
        expect(items.find((item) => item.key === "primary")?.label).toBe("fb_ctx_open_project");
        expect(keys(items)).toContain("import-project");
    });

    it("Reaper 工程同样走「打开工程」主操作", () => {
        const items = buildFileBrowserContextMenu(RPP, makeContext());
        expect(items.find((item) => item.key === "primary")?.label).toBe("fb_ctx_open_project");
    });

    it("普通文本文件：仍可复制路径 / 重命名 / 删除 / 看属性（并非无从操作）", () => {
        const items = buildFileBrowserContextMenu(TXT, makeContext());
        expect(keys(items)).toEqual(
            expect.arrayContaining(["reveal", "copy-paths", "rename", "delete", "properties"]),
        );
    });

    it("搜索模式下多出「转到所在文件夹」", () => {
        const items = buildFileBrowserContextMenu(WAV, makeContext({ isSearchMode: true }));
        expect(keys(items)).toContain("open-containing");
    });

    it("计算机虚拟层：没有写操作（重命名 / 删除）", () => {
        const items = buildFileBrowserContextMenu(DIR, makeContext({ isComputerLevel: true }));
        expect(keys(items)).not.toContain("rename");
        expect(keys(items)).not.toContain("delete");
    });

    it("多选：主操作省略，插入变成三种排布方式，删除项标注数量", () => {
        const items = buildFileBrowserContextMenu(WAV, makeContext({ selected: [WAV, WAV2] }));
        expect(keys(items)).not.toContain("primary");
        expect(keys(items)).toEqual(
            expect.arrayContaining([
                "insert-heading",
                "insert-across-time",
                "insert-across-tracks",
                "insert-as-takes",
            ]),
        );
        expect(items.find((item) => item.key === "delete")?.label).toBe("fb_ctx_delete_items:2");
        // 多选时"复制文件名"没有唯一目标。
        expect(keys(items)).not.toContain("copy-name");
    });

    it("多选插入：三种排布方式各自把模式传给同一个动作", () => {
        const context = makeContext({ selected: [WAV, WAV2] });
        const items = buildFileBrowserContextMenu(WAV, context);
        items.find((item) => item.key === "insert-across-tracks")?.onSelect?.();
        expect(context.actions.insertMultiple).toHaveBeenCalledWith([WAV, WAV2], "across-tracks");
    });
});

describe("背景菜单", () => {
    it("包含新建 / 刷新 / 排序 / 视图 / 选择 / 目录入口", () => {
        const items = buildFileBrowserContextMenu(null, makeContext());
        expect(keys(items)).toEqual(
            expect.arrayContaining([
                "new-folder",
                "refresh",
                "sort-heading",
                "sort-name",
                "sort-date",
                "sort-size",
                "sort-descending",
                "folders-first",
                "show-hidden",
                "media-only",
                "view-options",
                "select-all",
                "clear-selection",
                "reveal-current",
                "open-folder-dialog",
            ]),
        );
    });

    it("排序项勾选当前依据；切换时把新依据交给动作", () => {
        const context = makeContext();
        const items = buildFileBrowserContextMenu(null, context);
        expect(items.find((item) => item.key === "sort-name")?.checked).toBe(true);
        expect(items.find((item) => item.key === "sort-size")?.checked).toBe(false);
        items.find((item) => item.key === "sort-size")?.onSelect?.();
        expect(context.actions.setSortMode).toHaveBeenCalledWith("size");
    });

    it("显示开关勾选当前值，点击取反", () => {
        const context = makeContext({
            view: { ...DEFAULT_FILE_BROWSER_VIEW_OPTIONS, showHiddenFiles: false },
        });
        const items = buildFileBrowserContextMenu(null, context);
        expect(items.find((item) => item.key === "show-hidden")?.checked).toBe(false);
        items.find((item) => item.key === "show-hidden")?.onSelect?.();
        expect(context.actions.patchView).toHaveBeenCalledWith({ showHiddenFiles: true });
    });

    it("没有选中项时「取消选择」禁用", () => {
        const items = buildFileBrowserContextMenu(null, makeContext({ selected: [] }));
        expect(items.find((item) => item.key === "clear-selection")?.disabled).toBe(true);
    });

    it("计算机虚拟层：没有新建文件夹，也没有「显示当前文件夹」", () => {
        const items = buildFileBrowserContextMenu(
            null,
            makeContext({ isComputerLevel: true, currentPath: "computer://" }),
        );
        expect(keys(items)).not.toContain("new-folder");
        expect(keys(items)).not.toContain("reveal-current");
        expect(keys(items)).toContain("refresh");
    });
});
