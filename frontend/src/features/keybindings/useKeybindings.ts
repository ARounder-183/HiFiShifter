import { useEffect, useRef } from "react";
import { useAppSelector } from "../../app/hooks";
import { selectMergedKeybindings } from "./keybindingsSlice";
import { ACTION_META } from "./defaultKeybindings";
import type { ActionId, Keybinding, KeybindingMap } from "./types";
import type { RootState } from "../../app/store";

/**
 * 判断当前焦点是否在可编辑元素上（输入框等），此时不拦截快捷键
 */
function isEditableTarget(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el) return false;
    const tag = (el.tagName ?? "").toLowerCase();
    if (tag === "input" || tag === "textarea" || tag === "select") {
        return true;
    }
    if (el.isContentEditable) return true;
    if (el.closest?.('input,textarea,select,[contenteditable="true"]'))
        return true;
    return false;
}

/**
 * 将 KeyboardEvent 的按键信息规范化为小写 key 字符串
 */
export function normalizeEventKey(e: KeyboardEvent): string {
    // 对 Space 按键特殊处理
    if (e.key === " " || e.code === "Space") return "space";
    return e.key.toLowerCase();
}

/**
 * 判断按下的按键是否匹配某个 Keybinding 定义
 */
export function matchesKeybinding(e: KeyboardEvent, kb: Keybinding): boolean {
    const key = normalizeEventKey(e);
    if (key !== kb.key) return false;

    const isMac =
        typeof navigator !== "undefined" &&
        navigator.platform?.toLowerCase().includes("mac");

    const modKey = isMac ? e.metaKey : e.ctrlKey;
    const wantCtrl = Boolean(kb.ctrl);
    const wantShift = Boolean(kb.shift);
    const wantAlt = Boolean(kb.alt);

    if (modKey !== wantCtrl) return false;
    if (e.shiftKey !== wantShift) return false;
    if (e.altKey !== wantAlt) return false;
    return true;
}

/**
 * 在给定映射表中查找匹配的 actionId
 * @param excludeScopedContexts 排除特定 scopedContext 的操作（用于非 PianoRoll 上下文过滤掉 paramEditorSelect 级别的操作）
 */
function findMatchingAction(
    e: KeyboardEvent,
    keybindings: KeybindingMap,
    excludeScopedContexts?: Set<string>,
): ActionId | null {
    // 优先匹配含修饰键的绑定（避免裸键误触）
    const entries = Object.entries(keybindings) as [ActionId, Keybinding][];

    // 先检查有修饰键的绑定
    for (const [actionId, kb] of entries) {
        if (kb.modifierOnly) continue;
        if (kb.ctrl || kb.shift || kb.alt) {
            if (matchesKeybinding(e, kb)) {
                const ctx = ACTION_META[actionId]?.scopedContext;
                if (excludeScopedContexts && ctx && excludeScopedContexts.has(ctx)) continue;
                return actionId;
            }
        }
    }
    // 再检查无修饰键的绑定
    for (const [actionId, kb] of entries) {
        if (kb.modifierOnly) continue;
        if (!kb.ctrl && !kb.shift && !kb.alt) {
            if (matchesKeybinding(e, kb)) {
                const ctx = ACTION_META[actionId]?.scopedContext;
                if (excludeScopedContexts && ctx && excludeScopedContexts.has(ctx)) continue;
                return actionId;
            }
        }
    }
    return null;
}

function mergeExcludes(
    ...sets: Array<Set<string> | undefined>
): Set<string> | undefined {
    const merged = new Set<string>();
    for (const s of sets) {
        if (!s) continue;
        for (const v of s) merged.add(v);
    }
    return merged.size > 0 ? merged : undefined;
}

export type KeybindingActionHandler = (actionId: ActionId) => void;

/**
 * 全局快捷键监听 Hook
 *
 * 从 Redux store 读取合并后的快捷键映射，统一监听 keydown 事件，
 * 匹配到操作后回调 handler。
 *
 * @param handler — 收到匹配的 actionId 后执行具体逻辑的回调
 */
export function useKeybindings(handler: KeybindingActionHandler): void {
    const keybindings = useAppSelector(selectMergedKeybindings);
    const toolMode = useAppSelector((state: RootState) => state.session.toolMode);
    const keybindingsRef = useRef(keybindings);
    const handlerRef = useRef(handler);
    const toolModeRef = useRef(toolMode);

    useEffect(() => {
        keybindingsRef.current = keybindings;
    }, [keybindings]);

    useEffect(() => {
        handlerRef.current = handler;
    }, [handler]);

    useEffect(() => {
        toolModeRef.current = toolMode;
    }, [toolMode]);

    useEffect(() => {
        const excludeParamEditor = new Set(["paramEditorSelect"]);
        const excludeQuickSearch = new Set(["quickSearch"]);

        function onKeyDown(e: KeyboardEvent) {
            if (e.repeat) return;
            if (isEditableTarget(document.activeElement) || isEditableTarget(e.target))
                return;

            // 快捷键设置对话框打开时，阻塞所有快捷键
            if (document.body.hasAttribute("data-keybindings-dialog-open")) return;

            // Quick Search 打开时，交给弹窗自身输入框处理（避免 ↑/↓ 与时间轴缩放冲突）
            if (document.body.hasAttribute("data-quick-search-open")) return;

            // PianoRoll scroller 内的快捷键由其自身 onKeyDown 处理，不拦截
            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll =
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]");
            if (inPianoRoll) {
                // 查找匹配的 actionId，如果属于 pianoRoll.* 则放行给 PianoRoll 自己处理
                // （shiftParamUp/Down 已移至全局 handler，不需要放行）
                const matchedAction = findMatchingAction(e, keybindingsRef.current);
                if (
                    matchedAction?.startsWith("pianoRoll.") &&
                    matchedAction !== "pianoRoll.shiftParamUp" &&
                    matchedAction !== "pianoRoll.shiftParamDown" &&
                    matchedAction !== "pianoRoll.shiftParamUpSelection" &&
                    matchedAction !== "pianoRoll.shiftParamDownSelection"
                ) {
                    return;
                }
                // clip.copy/clip.paste 同样放行，因为 pianoRoll.copy/pianoRoll.paste
                // 共享相同快捷键，PianoRoll 的本地 onKeyDown 处理参数帧的复制粘贴
                if (matchedAction === "clip.copy" || matchedAction === "clip.paste") {
                    return;
                }
                // paramEditorSelect 级别的操作（如 edit.initialize/transposeCents/...）
                // 仅当当前工具为 "select" 时放行给 PianoRoll 处理；
                // 否则跳过此操作，继续查找非 scoped 的匹配（如 clip.delete）
                if (matchedAction && ACTION_META[matchedAction]?.scopedContext === "paramEditorSelect") {
                    if (toolModeRef.current === "select") {
                        return; // PianoRoll 会处理
                    }
                    // 工具不是 select，排除 paramEditorSelect 操作后重新查找
                    const fallbackAction = findMatchingAction(
                        e,
                        keybindingsRef.current,
                        mergeExcludes(excludeParamEditor, excludeQuickSearch),
                    );
                    if (fallbackAction) {
                        e.preventDefault();
                        e.stopPropagation();
                        handlerRef.current(fallbackAction);
                    }
                    return;
                }
                // edit.selectAll / edit.deselect 在 PianoRoll 中也放行
                if (matchedAction === "edit.selectAll" || matchedAction === "edit.deselect") {
                    if (toolModeRef.current === "select") {
                        return;
                    }
                }
            }

            // 非 PianoRoll 上下文：排除 paramEditorSelect 级别操作以避免冲突
            // （如 Delete 应匹配 clip.delete 而非 edit.initialize）
            const actionId = findMatchingAction(
                e,
                keybindingsRef.current,
                inPianoRoll
                    ? excludeQuickSearch
                    : mergeExcludes(excludeParamEditor, excludeQuickSearch),
            );
            if (!actionId) return;

            e.preventDefault();
            e.stopPropagation();
            handlerRef.current(actionId);
        }

        window.addEventListener("keydown", onKeyDown, true);
        return () => window.removeEventListener("keydown", onKeyDown, true);
    }, []);
}

export { isEditableTarget };
