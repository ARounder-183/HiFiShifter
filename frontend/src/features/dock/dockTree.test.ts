import { test } from "vitest";

import {
    addFormToTabset,
    clampRatio,
    collectDockedForms,
    collectTabsets,
    collectVisibleForms,
    findParentSplit,
    findTabsetOfForm,
    isFormVisible,
    moveForm,
    nextZoneId,
    pruneTree,
    removeForm,
    setSplitRatio,
    setTabsetCollapsed,
    splitRect,
    splitTabsetWith,
    zoneIdAllocator,
} from "./dockTree.ts";
import type { DockLayout, DockSplitNode, DockTabsetNode } from "./dockTypes.ts";

function assertEqual<T>(actual: T, expected: T, label: string): void {
    const a = JSON.stringify(actual);
    const b = JSON.stringify(expected);
    if (a !== b) throw new Error(`${label}: expected ${b}, received ${a}`);
}

function assert(condition: boolean, label: string): void {
    if (!condition) throw new Error(label);
}

function tabset(id: string, tabs: string[]): DockTabsetNode {
    return { t: "tabset", id, tabs, active: tabs[0] };
}

function split(
    id: string,
    a: DockSplitNode["a"],
    b: DockSplitNode["b"],
    extra: Partial<DockSplitNode> = {},
): DockSplitNode {
    return { t: "split", id, dir: "row", ratio: 0.5, fixed: null, a, b, ...extra };
}

/** 便捷的树形状断言：把树压成紧凑字符串。 */
function shape(node: DockSplitNode["a"]): string {
    if (node.t === "tabset") return `[${node.tabs.join(",")}]`;
    return `(${shape(node.a)}|${shape(node.b)})`;
}

test("features/dock/dockTree.test.ts scripted checks", async () => {
    // ── zoneId 分配：连续分配不撞号 ──────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z5", ["b"]));
        const alloc = zoneIdAllocator(tree);
        assertEqual([alloc(), alloc(), alloc()], ["z6", "z7", "z8"], "allocator increments");
        assertEqual(nextZoneId(tree), "z6", "nextZoneId takes max + 1");
    }

    // ── 比例钳制 ────────────────────────────────────────────────
    {
        assertEqual(clampRatio(0.5), 0.5, "ratio in range");
        assertEqual(clampRatio(-3), 0.05, "ratio clamped low");
        assertEqual(clampRatio(9), 0.95, "ratio clamped high");
        assertEqual(clampRatio(Number.NaN), 0.5, "NaN ratio falls back");
    }

    // ── 摘除最后一个标签 → 组被剪掉 → 父分割塌缩 ─────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        assertEqual(shape(removeForm(tree, "a")!), "[b]", "empty tabset pruned, split collapses");
    }

    // ── 摘除不存在的窗体：树不变（引用相等）───────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        assert(removeForm(tree, "zzz") === tree, "removing unknown form is a no-op");
    }

    // ── 摘到全空 ────────────────────────────────────────────────
    {
        assertEqual(
            removeForm(tabset("z1", ["a"]), "a"),
            null,
            "removing the only form empties the tree",
        );
    }

    // ── 并入标签组（可指定下标）──────────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a", "c"]), tabset("z3", ["b"]));
        assertEqual(
            collectTabsets(addFormToTabset(tree, "z2", "x", 1))[0].tabs,
            ["a", "x", "c"],
            "insert at index",
        );
        assertEqual(
            collectTabsets(addFormToTabset(tree, "z2", "x"))[0].tabs,
            ["a", "c", "x"],
            "append when no index",
        );
    }

    // ── 并入时 active 切到新窗体 ─────────────────────────────────
    {
        const tree = addFormToTabset(tabset("z1", ["a"]), "z1", "b");
        assertEqual((tree as DockTabsetNode).active, "b", "inserted tab becomes active");
    }

    // ── 重复并入只改顺序，不产生重复标签 ──────────────────────────
    {
        const tree = addFormToTabset(tabset("z1", ["a", "b"]), "z1", "a");
        assertEqual(
            (tree as DockTabsetNode).tabs,
            ["b", "a"],
            "re-insert reorders, never duplicates",
        );
    }

    // ── 在某一侧拆分 ────────────────────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        const right = splitTabsetWith(tree, "z3", "x", "right");
        assertEqual(shape(right), "([a]|([b]|[x]))", "split right of target");

        const left = splitTabsetWith(tree, "z2", "x", "left");
        assertEqual(shape(left), "(([x]|[a])|[b])", "split left of target");

        const top = splitTabsetWith(tree, "z3", "x", "top");
        const topSplit = collectTabsets(top);
        assertEqual(topSplit.length, 3, "split creates a third tabset");
        assertEqual((top as DockSplitNode).dir, "row", "outer dir untouched");
    }

    // ── 同组内重排：不摘不插，避免组被剪掉的中间态 ────────────────
    {
        const tree = split("z1", tabset("z2", ["a", "b", "c"]), tabset("z3", ["d"]));
        const moved = moveForm(tree, "a", { kind: "tab", tabsetId: "z2", index: 2 });
        assertEqual(collectTabsets(moved)[0].tabs, ["b", "c", "a"], "reorder within tabset");
        assertEqual(shape(moved), "([b,c,a]|[d])", "structure unchanged by reorder");
    }

    // ── 把源组最后一个标签移到别的组：源组被剪，目标正常 ───────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        const moved = moveForm(tree, "a", { kind: "tab", tabsetId: "z3" });
        assertEqual(shape(moved), "[b,a]", "moving the only tab collapses its group");
    }

    // ── 跨组拆分移动 ────────────────────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b", "c"]));
        const moved = moveForm(tree, "b", { kind: "split", tabsetId: "z2", side: "bottom" });
        // b 从 z3 摘出（z3 只剩 c），再在 z2 下方拆出新组：(([a]|[b])|[c])
        assertEqual(
            shape(moved),
            "(([a]|[b])|[c])",
            "moved tab gets its own group below the target",
        );
        assertEqual(collectTabsets(moved).length, 3, "three groups after the move");
    }

    // ── 目标组因摘除而消失 → 退化为并入第一个组，窗体不丢 ──────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        const moved = moveForm(tree, "a", { kind: "tab", tabsetId: "z-missing" });
        assertEqual(
            collectDockedForms(moved).includes("a"),
            true,
            "form survives a missing target",
        );
    }

    // ── 移动不存在的窗体：树不变 ─────────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        assert(
            moveForm(tree, "zzz", { kind: "tab", tabsetId: "z3" }) === tree,
            "move unknown is a no-op",
        );
    }

    // ── 分割比例与固定像素 ──────────────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        const fixed = setSplitRatio(tree, "z1", 0.3, { side: "b", px: 360 });
        assertEqual((fixed as DockSplitNode).fixed, { side: "b", px: 360 }, "fixed side stored");
        assertEqual((fixed as DockSplitNode).ratio, 0.3, "ratio still recorded");
    }

    // ── 折叠 ────────────────────────────────────────────────────
    {
        const tree = setTabsetCollapsed(tabset("z1", ["a"]), "z1", true, 30);
        assertEqual((tree as DockTabsetNode).collapsed, true, "collapsed flag");
        assertEqual((tree as DockTabsetNode).collapsedPx, 30, "collapsed size");
    }

    // ── pruneTree：修正 active、去重、剪空 ───────────────────────
    {
        const dirty = {
            t: "tabset",
            id: "z1",
            tabs: ["a", "a", "b"],
            active: "gone",
        } as DockTabsetNode;
        const pruned = pruneTree(dirty) as DockTabsetNode;
        assertEqual(pruned.tabs, ["a", "b"], "duplicate tabs removed");
        assertEqual(pruned.active, "a", "stale active repaired");
        assertEqual(pruneTree({ ...dirty, tabs: [] }), null, "empty tabset pruned to null");
    }

    // ── 单子分割塌缩 ────────────────────────────────────────────
    {
        const tree: DockSplitNode = split(
            "z1",
            { t: "tabset", id: "z2", tabs: [], active: "" },
            tabset("z3", ["b"]),
        );
        assertEqual(shape(pruneTree(tree)!), "[b]", "single-child split collapses");
    }

    // ── findParentSplit ─────────────────────────────────────────
    {
        const tree = split("z1", tabset("z2", ["a"]), tabset("z3", ["b"]));
        assertEqual(findParentSplit(tree, "z3")?.id, "z1", "finds parent split");
        assertEqual(findParentSplit(tree, "z1"), null, "root has no parent");
    }

    // ── splitRect：比例与固定像素两种模式 ────────────────────────
    {
        const rect = { x: 0, y: 0, w: 1000, h: 400 };
        const [a1, b1] = splitRect(rect, { dir: "row", ratio: 0.6, fixed: null }, 4);
        assertEqual(Math.round(a1.w), 598, "ratio split side A");
        assertEqual(Math.round(b1.w), 398, "ratio split side B");
        assertEqual(b1.x, a1.w + 4, "splitter gap honoured");

        const [a2, b2] = splitRect(
            rect,
            { dir: "row", ratio: 0.6, fixed: { side: "b", px: 360 } },
            4,
        );
        assertEqual(Math.round(b2.w), 360, "fixed side B keeps its pixels");
        assertEqual(Math.round(a2.w), 636, "free side absorbs the remainder");

        const [a3, b3] = splitRect(
            rect,
            { dir: "row", ratio: 0.6, fixed: { side: "a", px: 360 } },
            4,
        );
        assertEqual(Math.round(a3.w), 360, "fixed side A keeps its pixels");
        assertEqual(Math.round(b3.w), 636, "free side absorbs the remainder");

        const [a4, b4] = splitRect(
            { x: 0, y: 0, w: 400, h: 300 },
            { dir: "col", ratio: 0.5, fixed: null },
            4,
        );
        assertEqual(Math.round(a4.h), 148, "column split side A height");
        assertEqual(Math.round(b4.h), 148, "column split side B height");
    }

    // ── 固定像素超过可用空间时被钳制，不产生负宽度 ────────────────
    {
        const [a, b] = splitRect(
            { x: 0, y: 0, w: 200, h: 100 },
            { dir: "row", ratio: 0.5, fixed: { side: "b", px: 9999 } },
            4,
        );
        assertEqual(Math.round(a.w), 0, "free side collapses instead of going negative");
        assertEqual(Math.round(b.w), 196, "fixed side clamped to available space");
    }

    // ── 可见性判定 ──────────────────────────────────────────────
    {
        const layout: DockLayout = {
            schema: 1,
            tree: split("z1", tabset("z2", ["a"]), tabset("z3", ["b"])),
            forms: {
                a: { id: "a", panelId: "a", float: null },
                b: { id: "b", panelId: "b", float: { x: 0, y: 0, w: 300, h: 200 } },
                c: { id: "c", panelId: "c", float: null },
            },
            order: ["a", "b", "c"],
            floatOrder: ["b"],
            gutters: { timelineTrackHeaderPx: 256 },
        };
        assertEqual(isFormVisible(layout, "a"), true, "docked form is visible");
        assertEqual(isFormVisible(layout, "c"), false, "closed form is not visible");
        assertEqual(collectVisibleForms(layout).sort(), ["a", "b"], "visible = docked + floating");
        assertEqual(findTabsetOfForm(layout.tree, "a")?.id, "z2", "locates owning tabset");
        assertEqual(findTabsetOfForm(layout.tree, "c"), null, "closed form has no tabset");
    }
});
