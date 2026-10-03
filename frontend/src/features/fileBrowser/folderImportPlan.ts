/**
 * 目录导入的"展开"：把一次目录扫描变成"轨道树 + 文件顺序"。
 *
 * 【为什么这段逻辑必须是纯函数并单独成文件】它是三个入口（系统拖放 / 文件浏览器
 * 拖拽 / 右键菜单）共用的那一段。写三遍必然分叉，而分叉的症状是"同一个文件夹从
 * 资源管理器拖进来和从文件浏览器拖进来得到不同的轨道结构"—— 只能靠肉眼发现。
 *
 * 【父子关系为什么从**文件系统路径**推导，而不是扫描结果的 label】label 是相对
 * 被拖入目录的路径，两个不同父目录下同名的被拖入目录会产生相同的 label。`dir` 是
 * 绝对路径，天然唯一，用它做父子判定不会歧义。
 */

import type { FolderMediaGroup } from "../../services/api/fileBrowser";
import { compareFileNames } from "./fileNameCompare";

/** 计划中的一棵子树：一个目录对应一条轨道。 */
export interface FolderImportPlanNode {
    /** 该组对应的源目录绝对路径（也是节点身份）。 */
    dir: string;
    /** 轨道名：目录名的最后一段。 */
    name: string;
    /** 该目录**直属**的媒体文件（已按资源管理器序排序）。 */
    files: string[];
    /** 子目录节点。 */
    children: FolderImportPlanNode[];
}

export interface FolderImportPlan {
    /** 顶层节点，按被拖入的顺序。 */
    roots: FolderImportPlanNode[];
    /** 直接拖入的散文件（不属于任何目录），保持拖入顺序。 */
    looseFiles: string[];
    /**
     * 全部媒体文件，DFS 先序：**子目录整棵在前，然后是本目录直属文件**。
     *
     * 【为什么子目录在前】与文件浏览器的 `foldersFirst` 默认一致 —— "你在列表里
     * 看到的顺序，就是导入后的顺序"。用户在浏览器里按名字排好，拖进来的序列就
     * 应该是他看到的那一个。
     */
    orderedFiles: string[];
    /** 展开出的媒体文件总数（含散文件，已去重）。 */
    totalFiles: number;
}

/** 把路径拆成段（同时接受 `/` 与 `\`，并忽略尾随分隔符）。 */
function segments(path: string): string[] {
    return path.split(/[\\/]+/).filter((part) => part.length > 0);
}

function lastSegment(path: string): string {
    const parts = segments(path);
    return parts.length > 0 ? parts[parts.length - 1] : path;
}

/** 父目录（用同一套分隔符规则，避免 `/` 与 `\` 混用导致匹配失败）。 */
function parentOf(path: string): string | null {
    const normalized = path.replace(/[\\/]+$/, "");
    const cut = Math.max(normalized.lastIndexOf("/"), normalized.lastIndexOf("\\"));
    return cut > 0 ? normalized.slice(0, cut) : null;
}

/** 用于比较的路径键：分隔符与大小写归一（Windows 路径不区分大小写）。 */
function dirKey(path: string): string {
    return path
        .replace(/[\\/]+$/, "")
        .replace(/\\/g, "/")
        .toLowerCase();
}

function collectOrdered(node: FolderImportPlanNode, out: string[]): void {
    for (const child of node.children) collectOrdered(child, out);
    out.push(...node.files);
}

/**
 * 把扫描结果展开成导入计划。
 *
 * @param groups 后端 `collect_folder_media` 的分组结果。
 * @param looseFiles 同时拖入的散文件（不属于任何被拖入目录）。
 */
export function buildFolderImportPlan(
    groups: readonly FolderMediaGroup[],
    looseFiles: readonly string[] = [],
): FolderImportPlan {
    const nodesByDir = new Map<string, FolderImportPlanNode>();
    for (const group of groups) {
        const key = dirKey(group.dir);
        // 同一个物理目录可能被两个被拖入项覆盖（例如同时拖了 `A` 与 `A/B`），
        // 后写入的会覆盖前者 —— 去重是有意的：同一个文件不该被导入两次。
        nodesByDir.set(key, {
            dir: group.dir,
            name: lastSegment(group.dir) || group.label,
            files: [...group.paths].sort(compareFileNames),
            children: [],
        });
    }

    const roots: FolderImportPlanNode[] = [];
    for (const group of groups) {
        const node = nodesByDir.get(dirKey(group.dir));
        if (!node) continue;
        const parentDir = parentOf(group.dir);
        const parent = parentDir ? nodesByDir.get(dirKey(parentDir)) : undefined;
        // 父节点存在且不是自己 → 挂进去；否则是根。
        if (parent && parent !== node) {
            // 迭代顺序是 DFS 先序，因此同级子节点天然保持扫描顺序。
            if (!parent.children.includes(node)) parent.children.push(node);
        } else if (!roots.includes(node)) {
            roots.push(node);
        }
    }

    const orderedFiles: string[] = [];
    for (const root of roots) collectOrdered(root, orderedFiles);

    // 全局去重（不同被拖入项可能展开出同一个文件），并接上散文件。
    const seen = new Set<string>();
    const deduped: string[] = [];
    for (const path of orderedFiles) {
        const key = dirKey(path);
        if (seen.has(key)) continue;
        seen.add(key);
        deduped.push(path);
    }
    const dedupedLoose: string[] = [];
    for (const path of looseFiles) {
        const key = dirKey(path);
        if (seen.has(key)) continue;
        seen.add(key);
        dedupedLoose.push(path);
    }

    return {
        roots,
        looseFiles: dedupedLoose,
        orderedFiles: [...deduped, ...dedupedLoose],
        totalFiles: deduped.length + dedupedLoose.length,
    };
}

/**
 * 这次目录导入有没有**任何媒体文件**可导。
 *
 * 【为什么判据必须随 `recursive` 走】递归关闭时子目录里的文件根本不会进
 * `orderedFiles`，把它们算作"这个文件夹有媒体"会得出"能导入但导入后是空的"。
 * 调用方按当前选项扫描后再构造计划，因此这里的 `totalFiles` 天然与 `recursive`
 * 一致 —— 不需要再传一个开关进来。
 *
 * 【为什么只看 `totalFiles`】它就是本次真正会被导入的集合（已去重、已按子目录
 * 优先排好，含散文件）。再数一遍 `roots` 里的文件等于复制一份计数，两处一旦
 * 分叉就是"对话框说 3 个、实际导入 4 个"。
 *
 * 【为什么这么薄还要单独存在】它把"可导入 = 有媒体文件"这条规则**命名**了。
 * 此前三个入口各自用 `scan.totalFiles` / `plan.orderedFiles.length` 表达同一件事，
 * 于是"弹窗路径挡住了空目录、非弹窗路径没挡"这种分叉得以存在。
 */
export function hasImportableMedia(plan: FolderImportPlan): boolean {
    return plan.totalFiles > 0;
}
