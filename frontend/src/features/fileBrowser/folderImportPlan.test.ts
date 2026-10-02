/**
 * 目录导入的展开逻辑与选项归一化。
 *
 * 【这里钉住的是"三个入口共用的那一段"】系统拖放、文件浏览器拖拽、右键菜单都走
 * `buildFolderImportPlan`。它一旦分叉，同一个文件夹从资源管理器拖进来和从文件
 * 浏览器拖进来会得到不同的轨道结构 —— 只能靠肉眼发现。
 */

import { describe, expect, test } from "vitest";

import type { FolderMediaGroup } from "../../services/api/fileBrowser";
import {
    buildFolderImportPlan,
    shouldPromptFolderImport,
} from "./folderImportPlan";
import {
    DEFAULT_FOLDER_IMPORT_OPTIONS,
    FOLDER_IMPORT_MODES,
    FOLDER_IMPORT_MODE_LABEL_KEY,
    normalizeFolderImportOptions,
} from "./folderImportOptions";

function group(dir: string, paths: string[], hasSubdirs = false): FolderMediaGroup {
    return { dir, label: dir, paths, hasSubdirs };
}

/** 把计划压成"名字: 文件数"的形状，便于断言结构。 */
function shape(node: {
    name: string;
    files: string[];
    children: { name: string; files: string[]; children: unknown[] }[];
}): unknown {
    return {
        name: node.name,
        files: node.files,
        children: node.children.map((child) =>
            shape(child as Parameters<typeof shape>[0]),
        ),
    };
}

describe("buildFolderImportPlan：目录树", () => {
    test("单个目录：一条根，文件按资源管理器序排好", () => {
        const plan = buildFolderImportPlan([
            group("C:\\music\\Takes", [
                "C:\\music\\Takes\\take10.wav",
                "C:\\music\\Takes\\take2.wav",
                "C:\\music\\Takes\\take1.wav",
            ]),
        ]);
        expect(plan.roots).toHaveLength(1);
        expect(plan.roots[0].name).toBe("Takes");
        // 自然序：take2 在 take10 之前（裸字符串比较会反过来）。
        expect(plan.roots[0].files.map((p) => p.split("\\").pop())).toEqual([
            "take1.wav",
            "take2.wav",
            "take10.wav",
        ]);
        expect(plan.orderedFiles).toHaveLength(3);
        expect(plan.totalFiles).toBe(3);
    });

    test("递归：子目录挂到父目录下，名字取最后一段", () => {
        const plan = buildFolderImportPlan([
            group("C:\\music\\Takes", ["C:\\music\\Takes\\a.wav"], true),
            group("C:\\music\\Takes\\Sub", ["C:\\music\\Takes\\Sub\\b.wav"], true),
            group("C:\\music\\Takes\\Sub\\Deep", ["C:\\music\\Takes\\Sub\\Deep\\c.wav"]),
        ]);
        expect(plan.roots).toHaveLength(1);
        expect(shape(plan.roots[0])).toEqual({
            name: "Takes",
            files: ["C:\\music\\Takes\\a.wav"],
            children: [
                {
                    name: "Sub",
                    files: ["C:\\music\\Takes\\Sub\\b.wav"],
                    children: [
                        {
                            name: "Deep",
                            files: ["C:\\music\\Takes\\Sub\\Deep\\c.wav"],
                            children: [],
                        },
                    ],
                },
            ],
        });
    });

    test("父子关系由**路径**推导，不受 label 影响", () => {
        // 后端的 label 是相对被拖入目录的路径；两个不同的被拖入目录下同名的子目录
        // 会有相同的 label。若按 label 建树，这两组会被错误地合并。
        const plan = buildFolderImportPlan([
            {
                dir: "C:\\a\\Takes",
                label: "Takes",
                paths: ["C:\\a\\Takes\\x.wav"],
                hasSubdirs: false,
            },
            {
                dir: "C:\\b\\Takes",
                label: "Takes",
                paths: ["C:\\b\\Takes\\y.wav"],
                hasSubdirs: false,
            },
        ]);
        expect(plan.roots).toHaveLength(2);
        expect(plan.roots.map((node) => node.dir)).toEqual(["C:\\a\\Takes", "C:\\b\\Takes"]);
        expect(plan.totalFiles).toBe(2);
    });

    test("同时拖入目录与其子目录：按文件系统层级嵌套，不重复导入", () => {
        // 后端会为两个被拖入项各走一趟，`Sub` 因此出现两次；去重后只保留一份。
        const plan = buildFolderImportPlan([
            group("C:\\music\\Takes", ["C:\\music\\Takes\\a.wav"], true),
            group("C:\\music\\Takes\\Sub", ["C:\\music\\Takes\\Sub\\b.wav"]),
            group("C:\\music\\Takes\\Sub", ["C:\\music\\Takes\\Sub\\b.wav"]),
        ]);
        expect(plan.roots).toHaveLength(1);
        expect(plan.roots[0].children).toHaveLength(1);
        expect(plan.totalFiles).toBe(2);
    });

    test("文件顺序：子目录整棵在前，然后是本目录直属文件（与 foldersFirst 一致）", () => {
        const plan = buildFolderImportPlan([
            group("C:\\music\\Takes", ["C:\\music\\Takes\\own.wav"], true),
            group("C:\\music\\Takes\\Sub", ["C:\\music\\Takes\\Sub\\inner.wav"]),
        ]);
        expect(plan.orderedFiles).toEqual([
            "C:\\music\\Takes\\Sub\\inner.wav",
            "C:\\music\\Takes\\own.wav",
        ]);
    });

    test("散文件接在目录展开结果之后，并参与去重", () => {
        const plan = buildFolderImportPlan(
            [group("C:\\music\\Takes", ["C:\\music\\Takes\\a.wav"])],
            ["C:\\loose\\b.wav", "C:\\music\\Takes\\a.wav"],
        );
        expect(plan.looseFiles).toEqual(["C:\\loose\\b.wav"]);
        expect(plan.orderedFiles).toEqual(["C:\\music\\Takes\\a.wav", "C:\\loose\\b.wav"]);
        expect(plan.totalFiles).toBe(2);
    });

    test("空扫描与空目录都安全", () => {
        expect(buildFolderImportPlan([])).toEqual({
            roots: [],
            looseFiles: [],
            orderedFiles: [],
            totalFiles: 0,
        });
        const plan = buildFolderImportPlan([group("C:\\music\\Empty", [])]);
        expect(plan.roots).toHaveLength(1);
        expect(plan.totalFiles).toBe(0);
    });
});

describe("shouldPromptFolderImport", () => {
    test("没有子目录、没被截断 → 直接用记住的选项执行", () => {
        expect(shouldPromptFolderImport({ hasSubdirs: false, truncated: false })).toBe(false);
    });

    test("有子目录 → 弹（递归选项第一次有意义）", () => {
        expect(shouldPromptFolderImport({ hasSubdirs: true, truncated: false })).toBe(true);
    });

    test("被截断 → 必弹（不告知就等于静默少导入）", () => {
        expect(shouldPromptFolderImport({ hasSubdirs: false, truncated: true })).toBe(true);
    });

    test("显式要求 → 必弹", () => {
        expect(
            shouldPromptFolderImport({ hasSubdirs: false, truncated: false, force: true }),
        ).toBe(true);
    });
});

describe("folderImportOptions", () => {
    test("缺省值自洽：默认 mode 是唯一能让建轨道组生效的那个", () => {
        expect(DEFAULT_FOLDER_IMPORT_OPTIONS.mode).toBe("across-tracks");
        expect(DEFAULT_FOLDER_IMPORT_OPTIONS.createFolderTracks).toBe(true);
        // 递归默认关（与 REAPER 一致，也是用户指定的）。
        expect(DEFAULT_FOLDER_IMPORT_OPTIONS.recursive).toBe(false);
    });

    test("归一化：非法值逐字段回退，永不抛错", () => {
        expect(normalizeFolderImportOptions(null)).toEqual(DEFAULT_FOLDER_IMPORT_OPTIONS);
        expect(normalizeFolderImportOptions("nonsense")).toEqual(DEFAULT_FOLDER_IMPORT_OPTIONS);
        expect(
            normalizeFolderImportOptions({ mode: "nope", recursive: "yes", createFolderTracks: 1 }),
        ).toEqual(DEFAULT_FOLDER_IMPORT_OPTIONS);
    });

    test("归一化：合法值被保留", () => {
        expect(
            normalizeFolderImportOptions({
                mode: "as-takes",
                recursive: true,
                createFolderTracks: false,
            }),
        ).toEqual({ mode: "as-takes", recursive: true, createFolderTracks: false });
    });

    test("每种排布方式都有显示名（穷举 Record 的运行时对应）", () => {
        for (const mode of FOLDER_IMPORT_MODES) {
            expect(FOLDER_IMPORT_MODE_LABEL_KEY[mode]).toBeTruthy();
        }
    });
});
