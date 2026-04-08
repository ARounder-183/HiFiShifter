import { useEffect } from "react";
import type { AppDispatch } from "../../../../app/store";
import { useAppSelector } from "../../../../app/hooks";
import type { SessionState } from "../../../../features/session/sessionSlice";
import { removeClipsRemote } from "../../../../features/session/sessionSlice";
import type { ClipTemplate } from "../../../../features/session/sessionTypes";
import { selectMergedKeybindings } from "../../../../features/keybindings/keybindingsSlice";
import type { ActionId, Keybinding, KeybindingMap } from "../../../../features/keybindings/types";
import { writeSystemClipboardObject } from "../../../../utils/systemClipboard";

const IS_MAC =
    typeof navigator !== "undefined" && navigator.platform?.toLowerCase().includes("mac");

const CLIP_ACTIONS: ActionId[] = [
    "clip.delete",
    "clip.copy",
    "clip.cut",
    "clip.paste",
    "clip.split",
    "clip.normalize",
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
    clipClipboardRef: React.RefObject<ClipTemplate[] | null>;
    buildClipClipboardTemplates: (ids: string[]) => Promise<ClipTemplate[]>;
    isEditableTarget: (target: EventTarget | null) => boolean;
    onNormalize: (ids: string[]) => void;
    onPaste: () => void;
    onSplitSelected: () => void;
}) {
    const {
        sessionRef,
        dispatch,
        multiSelectedClipIds,
        setMultiSelectedClipIds,
        clipClipboardRef,
        buildClipClipboardTemplates,
        isEditableTarget,
        onNormalize,
        onPaste,
        onSplitSelected,
    } = deps;

    const keybindings = useAppSelector(selectMergedKeybindings);

    useEffect(() => {
        function onKeyDown(e: KeyboardEvent) {
            if (e.repeat) return;
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
            const inPianoRoll =
                active?.hasAttribute("data-piano-roll-scroller") ||
                active?.closest?.("[data-piano-roll-scroller]");
            const inTrackHeader =
                Boolean(active?.closest?.("[data-track-list-panel]")) ||
                document.body.getAttribute("data-hs-focus-window") === "trackHeader";

            // clip.paste 与 pianoRoll.paste 冲突时：参数编辑器 / 轨道头焦点优先参数粘贴
            if (actionId === "clip.paste" && (inPianoRoll || inTrackHeader)) {
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
                    void (async () => {
                        const templates = await buildClipClipboardTemplates(selectedIds);
                        if (templates.length === 0) return;
                        (
                            clipClipboardRef as React.MutableRefObject<ClipTemplate[] | null>
                        ).current = templates;
                        try {
                            await writeSystemClipboardObject({
                                version: 1,
                                kind: "clip",
                                templates,
                            });
                        } catch {
                            // ignore
                        }
                    })();
                    return;
                }

                case "clip.cut": {
                    if (selectedIds.length === 0) return;
                    e.preventDefault();
                    e.stopPropagation();
                    void (async () => {
                        const templates = await buildClipClipboardTemplates(selectedIds);
                        if (templates.length === 0) return;
                        (
                            clipClipboardRef as React.MutableRefObject<ClipTemplate[] | null>
                        ).current = templates;
                        try {
                            await writeSystemClipboardObject({
                                version: 1,
                                kind: "clip",
                                templates,
                            });
                        } catch {
                            // ignore
                        }
                        setMultiSelectedClipIds([]);
                        void dispatch(removeClipsRemote(selectedIds));
                    })();
                    return;
                }

                case "clip.paste": {
                    e.preventDefault();
                    e.stopPropagation();
                    onPaste();
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
            }
        }
        window.addEventListener("keydown", onKeyDown, true);
        return () => window.removeEventListener("keydown", onKeyDown, true);
    }, [
        dispatch,
        multiSelectedClipIds,
        sessionRef,
        setMultiSelectedClipIds,
        clipClipboardRef,
        buildClipClipboardTemplates,
        isEditableTarget,
        keybindings,
        onNormalize,
        onPaste,
        onSplitSelected,
    ]);
}
