import { useCallback, useEffect, useRef } from "react";
import type { AppDispatch } from "../../../../app/store";
import { useAppSelector } from "../../../../app/hooks";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { removeClipsRemote } from "../../../../features/session/sessionSlice";
import { selectMergedKeybindings } from "../../../../features/keybindings/keybindingsSlice";
import type { ActionId, Keybinding, KeybindingMap } from "../../../../features/keybindings/types";
import { shouldRouteClipPasteToParamEditor } from "../clipboardFocusRouting";
import { expandClipIdsWithGroups } from "./useGroupExpansion";
import { IS_MAC } from "../../../../utils/platform";

// 长按重复粘贴的节奏：首次按下立即粘贴，持续按住超过初始延迟后，
// 按固定间隔连续粘贴。节奏由前端控制，不依赖系统按键重复速率。
const PASTE_HOLD_INITIAL_DELAY_MS = 400;
const PASTE_REPEAT_INTERVAL_MS = 50;

const CLIP_ACTIONS: ActionId[] = [
    "clip.delete",
    "clip.copy",
    "clip.cut",
    "clip.paste",
    "clip.split",
    "clip.normalize",
    "clip.group",
    "clip.ungroup",
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
    sessionRef: React.RefObject<SessionState>;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
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
        sessionRef,
        dispatch,
        multiSelectedClipIds,
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

    // ── 长按重复粘贴（clip.paste 专用）────────────────────────
    const holdPasteRef = useRef<{
        kb: Keybinding;
        initialTimer: number | null;
        repeatTimer: number | null;
    } | null>(null);

    const stopHoldPasteRepeat = useCallback(() => {
        const held = holdPasteRef.current;
        if (!held) return;
        if (held.initialTimer != null) window.clearTimeout(held.initialTimer);
        if (held.repeatTimer != null) window.clearInterval(held.repeatTimer);
        holdPasteRef.current = null;
    }, []);

    const startHoldPasteRepeat = useCallback(
        (kb: Keybinding, fire: () => void) => {
            stopHoldPasteRepeat();
            const held: {
                kb: Keybinding;
                initialTimer: number | null;
                repeatTimer: number | null;
            } = { kb, initialTimer: null, repeatTimer: null };
            holdPasteRef.current = held;
            held.initialTimer = window.setTimeout(() => {
                held.initialTimer = null;
                held.repeatTimer = window.setInterval(fire, PASTE_REPEAT_INTERVAL_MS);
            }, PASTE_HOLD_INITIAL_DELAY_MS);
        },
        [stopHoldPasteRepeat],
    );

    // 长按终止监听（keyup / blur / 卸载清理）。
    //
    // 注意：这里必须与下方的主快捷键 effect 分开、且只挂载一次。
    // 主 effect 的依赖（multiSelectedClipIds 等）在每次粘贴成功后都会
    // 变化并触发重建；若长按状态跟随主 effect 的 cleanup 被清掉，
    // 定时器会在初始延迟内就被杀死，长按重复粘贴永远不会生效。
    useEffect(() => {
        function onKeyUp(e: KeyboardEvent) {
            const held = holdPasteRef.current;
            if (!held) return;
            const key = e.key.toLowerCase();
            if (
                key === held.kb.key ||
                key === "control" ||
                key === "shift" ||
                key === "alt" ||
                key === "meta"
            ) {
                stopHoldPasteRepeat();
            }
        }
        function onBlur() {
            stopHoldPasteRepeat();
        }
        window.addEventListener("keyup", onKeyUp, true);
        window.addEventListener("blur", onBlur);
        return () => {
            window.removeEventListener("keyup", onKeyUp, true);
            window.removeEventListener("blur", onBlur);
            stopHoldPasteRepeat();
        };
    }, [stopHoldPasteRepeat]);

    useEffect(() => {
        function onKeyDown(e: KeyboardEvent) {
            // 长按期间的 OS 自动重复事件：若与进行中的长按粘贴同键，
            // 仅吞掉事件（重复节奏由自定义定时器控制）；其余重复一律忽略。
            if (e.repeat) {
                const held = holdPasteRef.current;
                if (held && matchesKeybinding(e, held.kb)) {
                    e.preventDefault();
                }
                return;
            }
            // 按下任何其他新按键都视为意图变化，终止进行中的长按粘贴。
            if (holdPasteRef.current) {
                stopHoldPasteRepeat();
            }
            if (isEditableTarget(document.activeElement) || isEditableTarget(e.target)) return;
            // 快捷键设置对话框打开时，阻塞所有快捷键
            if (document.body.hasAttribute("data-keybindings-dialog-open")) return;
            // 先拦截 actionId
            const actionId = matchClipAction(e, keybindings);
            if (!actionId) return;
            const s = sessionRef.current;
            const selectedIds =
                multiSelectedClipIds.length > 0
                    ? [...multiSelectedClipIds]
                    : s.selectedClipId
                      ? [s.selectedClipId]
                      : [];

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
                    const s = sessionRef.current;
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
                    const s = sessionRef.current;
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
                    // 长按重复：首次立即粘贴，持续按住后连续重复粘贴。
                    // 仅作用于时间轴粘贴路径（参数编辑器粘贴在上方已提前 return）。
                    const pasteKb = keybindings["clip.paste"];
                    if (pasteKb) startHoldPasteRepeat(pasteKb, onPaste);
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
            }
        }
        // 长按粘贴的终止（keyup / blur / 卸载）由上方独立的 mount-once
        // effect 负责；本 effect 依赖会随粘贴结果变化而重建，
        // 不能在这里清理长按定时器。
        window.addEventListener("keydown", onKeyDown, true);
        return () => window.removeEventListener("keydown", onKeyDown, true);
    }, [
        dispatch,
        multiSelectedClipIds,
        sessionRef,
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
        startHoldPasteRepeat,
    ]);
}
