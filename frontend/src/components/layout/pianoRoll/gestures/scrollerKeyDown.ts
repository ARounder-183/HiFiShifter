/**
 * 滚动容器的键盘动作。
 *
 * 【为什么单独成文件】它原本是 `usePianoRollInteractions` 里的一个 `useCallback`
 * （69 行）。这些回调只做一件事，却和另外几十个回调挤在同一个 hook 里。
 *
 * 【依赖怎么拿】上下文由调用方打包传入（`Pick<PointerDeps, …>`，字段清单由 AST
 * 扫描得出），函数体原样搬过来，开头解构回局部变量。
 */
import { ACTION_META } from "../../../../features/keybindings/defaultKeybindings";
import { matchesAnyKeybinding } from "../../../../features/keybindings/useKeybindings";
import type { ActionId, Keybinding } from "../../../../features/keybindings/types";
import type { KeyboardEvent } from "react";

import { emitWindowEvent } from "../../../../features/dock/windowEventBridge";
import type { PointerDeps } from "./pointerDeps";

/** 本回调需要的上下文。 */
type Deps = Pick<
    PointerDeps,
    "editParam" | "keybindingMap" | "onEditAction" | "pitchEnabled" | "rootTrackId" | "toolMode"
>;

/** 滚动容器的键盘动作。 */
export function handleScrollerKeyDown(deps: Deps, e: KeyboardEvent<HTMLDivElement>): void {
    const { editParam, keybindingMap, onEditAction, pitchEnabled, rootTrackId, toolMode } = deps;

    const key = e.key.toLowerCase();
    if (key === "arrowup" || key === "arrowdown" || key === "arrowleft" || key === "arrowright") {
        e.preventDefault();
    }

    if (!rootTrackId) return;
    if (editParam === "pitch" && !pitchEnabled) return;

    // pianoRoll.shiftParamUp / shiftParamDown 已移至全局 handleKeybindingAction 处理

    // Handle edit.* keybindings passed through from useKeybindings global handler
    // (must be before the selectionRef guard since selectAll/deselect work without selection)
    if (keybindingMap && onEditAction) {
        const editActionEntries = (
            Object.entries(keybindingMap) as [ActionId, readonly Keybinding[]][]
        ).filter(([id]) => id.startsWith("edit."));
        // 需要弹出对话框的操作列表
        const dialogOps = new Set([
            "transposeCents",
            "transposeDegrees",
            "setPitch",
            "average",
            "smooth",
            "addVibrato",
            "quantize",
            "meanQuantize",
        ]);
        for (const [actionId, bindings] of editActionEntries) {
            // 动作可绑多个键：任一槽位命中即触发（与全局路由同一约定）。
            if (bindings.some((kb) => kb.modifierOnly)) continue;
            if (!matchesAnyKeybinding(e.nativeEvent, bindings)) continue;
            {
                const meta = ACTION_META[actionId];
                // paramEditorSelect 在选择族（range / block）下都可用。
                // block 已由 toolMode === "select" 包含，不再有第三工具族。
                if (meta?.scopedContext === "paramEditorSelect" && toolMode !== "select") {
                    continue;
                }
                const op = actionId.replace("edit.", "");
                // undo/redo handled globally, skip here
                if (op === "undo" || op === "redo") continue;
                e.preventDefault();
                // 需要弹窗的操作 → 派发 openEditDialog 事件打开对话框
                if (dialogOps.has(op)) {
                    if (op === "addVibrato") {
                        // 颤音窗口的宿主在本面板（要读当前选区），卫星窗口必须本地打开。
                        window.dispatchEvent(
                            new CustomEvent("hifi:openEditDialog", { detail: { dialog: op } }),
                        );
                    } else {
                        // 其它编辑器对话框由主窗口 MenuBar 承载；从卫星窗口请求时
                        // 必须跨窗口送达，否则按快捷键毫无反应。
                        emitWindowEvent("main", "hifi:openEditDialog", { dialog: op });
                    }
                } else {
                    onEditAction(op);
                }
                return;
            }
        }
    }

    // 复制/粘贴（pianoRoll.copy / pianoRoll.paste）已上移：由全局路由
    // （useKeybindings → focusRouting.resolveEditOpRoute）按活动编辑
    // 表面定向派发到 PianoRollPanel.handleEditOp 唯一执行，本地不再
    // 维护平行的键位匹配分支（旧分支与 handleEditOp 重复且行为已
    // 分叉，是复制/粘贴冲突的放大器）。
}
