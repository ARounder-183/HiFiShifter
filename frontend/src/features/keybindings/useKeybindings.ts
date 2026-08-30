import { useEffect, useRef } from "react";
import { useAppSelector } from "../../app/hooks";
import { selectMergedKeybindings } from "./keybindingsSlice";
import { ACTION_META } from "./defaultKeybindings";
import type { ActionId } from "./types";
import type { RootState } from "../../app/store";
import { resolveActionByFocus, type KeybindingFocusDomain } from "./focusRouting";
import {
    matchesKeybinding,
    matchesKeybindingAllowingFineModifier,
    normalizeEventKey,
} from "./keybindingMatch";
const REPEATABLE_ACTIONS = new Set<ActionId>([
    "playback.seekLeft",
    "playback.seekRight",
    "timeline.zoomIn",
    "timeline.zoomOut",
    "track.selectUp",
    "track.selectDown",
]);
const VIBRATO_DRAG_KEYBOARD_ACTIONS: ActionId[] = [
    "pianoRoll.vibratoDragAmplitudeIncrease",
    "pianoRoll.vibratoDragAmplitudeDecrease",
    "pianoRoll.vibratoDragFrequencyIncrease",
    "pianoRoll.vibratoDragFrequencyDecrease",
];
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
    if (el.closest?.('input,textarea,select,[contenteditable="true"]')) return true;
    return false;
}

/** 计算当前键盘焦点域（与 MenuBar / TimelinePanel / TrackList 的标记一致） */
function computeFocusDomain(active: HTMLElement | null): KeybindingFocusDomain {
    const focusWindow = document.body.getAttribute("data-hs-focus-window");
    const inPianoRoll =
        active?.hasAttribute("data-piano-roll-scroller") ||
        active?.closest?.("[data-piano-roll-scroller]") ||
        focusWindow === "pianoRoll";
    const inTimeline =
        active?.hasAttribute("data-timeline-scroller") ||
        active?.closest?.("[data-timeline-scroller]") ||
        focusWindow === "timeline";
    const inTrackHeader =
        Boolean(active?.closest?.("[data-track-list-panel]")) || focusWindow === "trackHeader";
    if (inPianoRoll) return "pianoRoll";
    if (inTimeline) return "timeline";
    if (inTrackHeader) return "trackHeader";
    return null;
}

export type KeybindingActionHandler = (actionId: ActionId) => void;

/**
 * 全局快捷键监听 Hook
 *
 * 从 Redux store 读取合并后的快捷键映射，统一监听 keydown 事件，
 * 匹配到操作后回调 handler。
 *
 * 冲突解决（焦点路由）：同键值的多个绑定按「激活作用域 > 全局 >
 * 其它作用域 > 硬排除」排序取首位（见 focusRouting.ts），因此例如
 * 「添加轨道（Ctrl+T）」与「音高设置到（Ctrl+T）」可以在不同焦点域
 * 下共存 —— 参数编辑器内 select 工具下按 Ctrl+T 打开音高设置，
 * 时间轴/轨道头下按 Ctrl+T 新建轨道，与复制/粘贴的焦点分发同构。
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
        function onKeyDown(e: KeyboardEvent) {
            if (isEditableTarget(document.activeElement) || isEditableTarget(e.target)) return;

            // 快捷键设置对话框打开时，阻塞所有快捷键
            if (document.body.hasAttribute("data-keybindings-dialog-open")) return;

            // Quick Search 打开时，交给弹窗自身输入框处理（避免 ↑/↓ 与时间轴缩放冲突）
            if (document.body.hasAttribute("data-quick-search-open")) return;

            // 直线/颤音拖拽期间，命中振幅/频率方向键时，交给参数编辑器本地监听处理。
            if (document.body.hasAttribute("data-piano-roll-vibrato-drag-active")) {
                const fineAdjustKb = keybindingsRef.current["modifier.paramFineAdjust"];
                for (const actionId of VIBRATO_DRAG_KEYBOARD_ACTIONS) {
                    const kb = keybindingsRef.current[actionId];
                    if (!kb || kb.modifierOnly) continue;
                    if (matchesKeybindingAllowingFineModifier(e, kb, fineAdjustKb)) {
                        return;
                    }
                }
            }

            const active = document.activeElement as HTMLElement | null;
            const domain = computeFocusDomain(active);

            const key = normalizeEventKey(e);
            const isArrowKey =
                key === "arrowup" ||
                key === "arrowdown" ||
                key === "arrowleft" ||
                key === "arrowright";
            if (isArrowKey && domain) {
                e.preventDefault();
            }

            // ── 焦点感知路由：按「激活作用域 > 全局 > 其它作用域」取最佳匹配 ──
            const matched = resolveActionByFocus(
                e,
                keybindingsRef.current,
                domain,
                toolModeRef.current,
            );
            if (!matched) return;

            // 参数编辑器（钢琴卷帘）内的快捷键由其自身 onKeyDown 处理，不拦截
            if (domain === "pianoRoll") {
                if (ACTION_META[matched]?.scopedContext === "pianoRollVibratoDrag") {
                    return;
                }
                if (
                    matched.startsWith("pianoRoll.") &&
                    matched !== "pianoRoll.shiftParamUp" &&
                    matched !== "pianoRoll.shiftParamDown" &&
                    matched !== "pianoRoll.shiftParamUpSelection" &&
                    matched !== "pianoRoll.shiftParamDownSelection"
                ) {
                    return;
                }
                if (matched === "clip.copy" || matched === "clip.paste") {
                    return;
                }
                // paramEditorSelect 作用域的操作只有在「select 工具」下才会被
                // 路由到这里（见 focusRouting.scopePriority），此时交给 PianoRoll 处理。
                if (ACTION_META[matched]?.scopedContext === "paramEditorSelect") {
                    return;
                }
                if (matched === "edit.selectAll" || matched === "edit.deselect") {
                    if (toolModeRef.current === "select") {
                        return;
                    }
                }
            }

            // clip.* 操作由 TimelinePanel 内部处理，避免全局层提前吞掉事件。
            // （焦点在轨道头且发生键值重叠时，trackHeaderFocus 作用域的轨道
            // 操作会因更高优先级被路由到此处 —— 见 resolveActionByFocus。）
            if (matched.startsWith("clip.")) {
                return;
            }

            if (e.repeat && !REPEATABLE_ACTIONS.has(matched)) {
                e.preventDefault();
                return;
            }

            e.preventDefault();
            e.stopPropagation();
            handlerRef.current(matched);
        }

        window.addEventListener("keydown", onKeyDown, true);
        return () => window.removeEventListener("keydown", onKeyDown, true);
    }, []);
}

export {
    isEditableTarget,
    matchesKeybinding,
    normalizeEventKey,
    matchesKeybindingAllowingFineModifier,
};
