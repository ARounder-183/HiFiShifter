import { useEffect } from "react";
import type { AppDispatch } from "../../../../app/store";
import { useAppSelector, useAppStore } from "../../../../app/hooks";
import { cycleClipTakesRemote, removeClipsRemote } from "../../../../features/session/sessionSlice";
import {
    selectMergedKeybindings,
    beginHoldRepeat,
    consumeHoldRepeatKeyDown,
} from "../../../../features/keybindings";
import type { ActionId, Keybinding, KeybindingMap } from "../../../../features/keybindings/types";
import { shouldRouteClipPasteToParamEditor } from "../clipboardFocusRouting";
import { expandClipIdsWithGroups } from "./useGroupExpansion";
import { IS_MAC } from "../../../../utils/platform";

const CLIP_ACTIONS: ActionId[] = [
    "clip.delete",
    "clip.copy",
    "clip.cut",
    "clip.paste",
    "clip.split",
    "clip.normalize",
    "clip.group",
    "clip.ungroup",
    "clip.cycleTake",
    "clip.cycleTakePrev",
];
/**
 * 判断 KeyboardEvent 是否匹配某个 Keybinding
 */
function matchesKeybinding(e: KeyboardEvent, kb: Keybinding): boolean {
    let key = e.key.toLowerCase();
    if (key === " " || e.code === "Space") key = "space";

    if (key !== kb.key) return false;

    const modKey = IS_MAC ? e.metaKey : e.ctrlKey;
    if (modKey !== Boolean(kb.ctrl)) return false;
    if (e.shiftKey !== Boolean(kb.shift)) return false;
    if (e.altKey !== Boolean(kb.alt)) return false;
    return true;
}

/**
 * 在 keybinding map 中查找匹配的 actionId
 * 只检查 clip.* 操作
 */
function matchClipAction(e: KeyboardEvent, keybindings: KeybindingMap): ActionId | null {
    // 优先匹配含修饰键的
    for (const actionId of CLIP_ACTIONS) {
        const kb = keybindings[actionId];
        if ((kb.ctrl || kb.shift || kb.alt) && matchesKeybinding(e, kb)) {
            return actionId;
        }
    }
    for (const actionId of CLIP_ACTIONS) {
        const kb = keybindings[actionId];
        if (!kb.ctrl && !kb.shift && !kb.alt && matchesKeybinding(e, kb)) {
            return actionId;
        }
    }
    return null;
}

export function useKeyboardShortcuts(deps: {
    dispatch: AppDispatch;
    setMultiSelectedClipIds: (ids: string[]) => void;
    copyClips: (ids: string[]) => Promise<boolean>;
    cutClips: (ids: string[]) => void;
    isEditableTarget: (target: EventTarget | null) => boolean;
    onNormalize: (ids: string[]) => void;
    onPaste: () => void;
    onSplitSelected: () => void;
    onGroup: (ids: string[]) => void;
    onUngroup: (ids: string[]) => void;
}) {
    const {
        dispatch,
        setMultiSelectedClipIds,
        copyClips,
        cutClips,
        isEditableTarget,
        onNormalize,
        onPaste,
        onSplitSelected,
        onGroup,
        onUngroup,
    } = deps;

    const keybindings = useAppSelector(selectMergedKeybindings);
    // 从 store 实时读取会话状态：事件监听器里的 Redux store 是同步、权威的
    // 最新状态，而闭包捕获的 props / sessionRef 要等渲染+effect 提交后才会
    // 更新 —— 点击 Clip 后立刻按 Ctrl+C/X 时，后者仍是旧选区，导致复制/剪切
    // 作用到过期（甚至已删除）的 Clip 上而静默失败。
    const store = useAppStore();

    useEffect(() => {
        function onKeyDown(e: KeyboardEvent) {
            // 长按重复（clip.paste 等，与「添加轨道」等共用 holdRepeat 管理器）：
            // 顺带处理同键自动重复的吞掉与"新按键终止长按"（放在最前，语义
            // 与原内嵌实现一致：其余重复一律忽略、任何新键都视为意图变化）。
            if (consumeHoldRepeatKeyDown(e)) return;
            // 非长按期间的 OS 自动重复：一律忽略（节奏由 holdRepeat 计时器控制）。
            if (e.repeat) return;
            if (isEditableTarget(document.activeElement) || isEditableTarget(e.target)) return;
            // 快捷键设置对话框打开时，阻塞所有快捷键
            if (document.body.hasAttribute("data-keybindings-dialog-open")) return;
            // 先拦截 actionId
            const actionId = matchClipAction(e, keybindings);
            if (!actionId) return;
            // 实时读取 store 中的会话状态（同步、权威）：闭包里的 props /
            // sessionRef 在渲染+effect 提交前是旧值，会导致"刚点选就按
            // Ctrl+C/X"时复制/剪切落到过期选区上而静默失败。
            const s = store.getState().session;
            const rawSelectedIds =
                s.multiSelectedClipIds.length > 0
                    ? [...s.multiSelectedClipIds]
                    : s.selectedClipId
                      ? [s.selectedClipId]
                      : [];
            // 过滤已删除/已不存在的 Clip id：让失效选区（如删除、胶合、
            // 拆分替换 id 后的残留）不再把死 id 传给后端造成静默失败。
            const selectedIds = rawSelectedIds.filter((id) =>
                s.clips.some((clip) => clip.id === id),
            );

            const active = document.activeElement as HTMLElement | null;
            const inPianoRoll = Boolean(
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]"),
            );
            const inTrackHeader = Boolean(
                Boolean(active?.closest?.("[data-track-list-panel]")) ||
                document.body.getAttribute("data-hs-focus-window") === "trackHeader",
            );

            // clip.paste 与 pianoRoll.paste 冲突时：参数编辑器 / 轨道头焦点优先参数粘贴
            if (
                actionId === "clip.paste" &&
                shouldRouteClipPasteToParamEditor({
                    inPianoRoll,
                    inTrackHeader,
                })
            ) {
                e.preventDefault();
                e.stopPropagation();
                window.dispatchEvent(new CustomEvent("hifi:editOp", { detail: { op: "paste" } }));
                return;
            }

            // clip.copy / clip.cut / clip.paste: 焦点在 PianoRoll 时优先交给参数编辑器。
            if (actionId === "clip.copy" || actionId === "clip.cut" || actionId === "clip.paste") {
                if (inPianoRoll) {
                    if (s.toolMode === "select") {
                        e.preventDefault();
                        e.stopPropagation();
                        const op = actionId.replace("clip.", "");
                        window.dispatchEvent(new CustomEvent("hifi:editOp", { detail: { op } }));
                    }
                    return;
                }
            }

            // 不再对 clip.delete 做焦点位于 PianoRoll 的特殊放行。

            switch (actionId) {
                case "clip.delete": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    setMultiSelectedClipIds([]);
                    void dispatch(removeClipsRemote(selectedIds));
                    return;
                }

                case "clip.copy": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const expandedIds = expandClipIdsWithGroups(
                        selectedIds,
                        s.clips,
                        s.ignoreGrouping,
                        s.disabledGroupIds,
                    );
                    void copyClips(expandedIds);
                    return;
                }

                case "clip.cut": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    const expandedIds = expandClipIdsWithGroups(
                        selectedIds,
                        s.clips,
                        s.ignoreGrouping,
                        s.disabledGroupIds,
                    );
                    cutClips(expandedIds);
                    return;
                }

                case "clip.paste": {
                    e.preventDefault();
                    e.stopPropagation();
                    onPaste();
                    // 长按重复：首次立即粘贴，持续按住后按统一节奏连续重复
                    // （holdRepeat 管理器，与「添加轨道」等共用同一套长按逻辑）。
                    // 仅作用于时间轴粘贴路径（参数编辑器粘贴在上方已提前 return）。
                    const pasteKb = keybindings["clip.paste"];
                    if (pasteKb) beginHoldRepeat(pasteKb, onPaste);
                    return;
                }

                case "clip.split": {
                    e.preventDefault();
                    e.stopPropagation();
                    onSplitSelected();
                    return;
                }

                case "clip.normalize": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    onNormalize(selectedIds);
                    return;
                }

                case "clip.group": {
                    if (selectedIds.length < 2) return;
                    e.preventDefault();
                    e.stopPropagation();
                    onGroup(selectedIds);
                    return;
                }

                case "clip.ungroup": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    onUngroup(selectedIds);
                    return;
                }

                case "clip.cycleTake": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    void dispatch(cycleClipTakesRemote({ clipIds: selectedIds, direction: 1 }));
                    return;
                }

                case "clip.cycleTakePrev": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    void dispatch(cycleClipTakesRemote({ clipIds: selectedIds, direction: -1 }));
                    return;
                }
            }
        }
        // 长按重复的终止（keyup / blur / 卸载 / 新按键）由 holdRepeat
        // 管理器内部的全局监听负责，不随本 effect 重建 —— 本 effect 的
        // 依赖会随粘贴结果变化而重建，不能在这里清理长按定时器。
        window.addEventListener("keydown", onKeyDown, true);
        return () => window.removeEventListener("keydown", onKeyDown, true);
    }, [
        dispatch,
        store,
        setMultiSelectedClipIds,
        copyClips,
        cutClips,
        isEditableTarget,
        keybindings,
        onNormalize,
        onPaste,
        onSplitSelected,
        onGroup,
        onUngroup,
    ]);
}
