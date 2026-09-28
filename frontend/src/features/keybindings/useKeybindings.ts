import { useEffect, useRef } from "react";
import { useAppSelector } from "../../app/hooks";
import { selectMergedKeybindings } from "./keybindingsSlice";
import { ACTION_META } from "./defaultKeybindings";
import type { ActionId } from "./types";
import type { RootState } from "../../app/store";
import {
    resolveActionByFocus,
    ACTION_TO_EDIT_OP,
    type KeybindingFocusDomain,
} from "./focusRouting";
import { getActiveSurface } from "../uiFocus/focusSurface";
import { consumeHoldRepeatKeyDown } from "./holdRepeat";
import { isShortcutSuppressed } from "../../ui/shortcutScope";
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

/**
 * 复合控件自己拥有方向键 —— 全局快捷键必须让路。
 *
 * 【为什么必须有】`playback.seekLeft/Right` 默认绑定在左右方向键上、
 * `track.selectUp/Down` 绑在上下方向键上，而本分发器运行在 **window 捕获阶段**，
 * 命中后 `stopPropagation()`。于是凡是声明了 ARIA 键盘契约的复合控件，方向键
 * 永远到不了控件自己：焦点停在停靠标签上按 ←/→ 会去 seek，而不是切标签。
 * （实测确认：在标签上派发 `ArrowLeft`，事件在 window 捕获阶段就被本分发器
 * 吞掉，标签自己的处理器收不到。）
 *
 * ARIA 角色是"方向键归我管"的承诺，与 `isEditableTarget` 放行输入框是同一个
 * 道理：**先看谁拥有这组键，再谈全局绑定**。
 *
 * 覆盖的角色是标准里明确拥有方向键的那些。本仓库自己的停靠标签条与手写菜单
 * 都在其中；Radix 的 Select / Slider / Menu 也各自渲染这些角色，因此一并受益。
 */
const ARROW_OWNING_SELECTOR = [
    '[role="tablist"]',
    '[role="menu"]',
    '[role="menubar"]',
    '[role="listbox"]',
    '[role="radiogroup"]',
    '[role="slider"]',
    '[role="spinbutton"]',
    '[role="grid"]',
    '[role="tree"]',
    '[role="treegrid"]',
].join(",");

/** 会被复合控件接管的按键（方向键 + 标准里同组的首尾/翻页键）。 */
const COMPOSITE_WIDGET_KEYS = new Set([
    "arrowup",
    "arrowdown",
    "arrowleft",
    "arrowright",
    "home",
    "end",
    "pageup",
    "pagedown",
]);

function ownsArrowKeys(target: EventTarget | null): boolean {
    const el = target as HTMLElement | null;
    if (!el?.closest) return false;
    return el.closest(ARROW_OWNING_SELECTOR) !== null;
}

/**
 * 计算当前焦点域：即「活动编辑表面」（focusSurface 单一事实源，由最后
 * 一次 pointerdown / focusin 落点驱动）。不读 document.activeElement ——
 * 时间轴/轨道列刻意 preventDefault 自管焦点，DOM 焦点会滞留在上一个
 * 可聚焦元素上，曾导致复制/剪切/粘贴被路由到错误的编辑器。
 */
function computeFocusDomain(): KeybindingFocusDomain {
    return getActiveSurface();
}

export type KeybindingActionHandler = (actionId: ActionId) => void;

/**
 * 全局快捷键监听 Hook（window 上唯一的 keydown 监听器）
 *
 * 从 Redux store 读取合并后的快捷键映射，统一监听 keydown 事件，
 * 匹配到操作后回调 handler。
 *
 * 冲突解决（焦点路由）：同键值的多个绑定按「激活作用域 > 全局 >
 * 其它作用域 > 硬排除」排序取首位（见 focusRouting.ts），因此例如
 * 「添加轨道（Ctrl+T）」与「音高设置到（Ctrl+T）」可以在不同焦点域
 * 下共存。
 *
 * 复制/剪切/粘贴等编辑操作（clip.* 与 pianoRoll.* 共绑 Ctrl+C/X/V）是
 * 「同键异义」：归一为同一编辑 op 后由 handler 按活动编辑表面定向派发
 * （见 focusRouting.resolveEditOpRoute），冲突在结构上消解。
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
            // 长按重复（添加轨道/克隆轨道/粘贴等，见 holdRepeat.ts）：同键自动
            // 重复被吞掉（节奏由计时器控制），其它按键终止长按。放在最前，
            // 与粘贴的长按终止语义一致（新按键 = 意图变化）。
            if (consumeHoldRepeatKeyDown(e)) return;

            if (isEditableTarget(document.activeElement) || isEditableTarget(e.target)) return;

            /*
             * 任一模态表面打开时阻塞所有全局快捷键。
             *
             * 走统一的引用计数作用域（`src/ui/shortcutScope.ts`），
             * 而不是此前三个各自为政的 body 属性：那时 42 个对话框里只有 3 个
             * 接入了抑制，其余 39 个按空格会触发播放、按字母会触发时间轴动作。
             * `AppDialog` 现在自动接入，新对话框默认正确。
             */
            if (isShortcutSuppressed()) return;

            const key = normalizeEventKey(e);

            /*
             * 复合控件（标签条 / 菜单 / 下拉 / 滑杆…）自己处理方向键，全局绑定让路。
             * 必须在下面的 `resolveActionByFocus` 之前返回 —— 那条路径会
             * `stopPropagation()`，让控件再也收不到这个按键。
             */
            if (
                COMPOSITE_WIDGET_KEYS.has(key) &&
                (ownsArrowKeys(e.target) || ownsArrowKeys(document.activeElement))
            ) {
                return;
            }

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

            // 参数线拖拽期间，「切换拖动方向」键交由参数编辑器本地监听统一
            // 处理（与拖拽中右键同义）。这里必须放行且不消费：
            // - 避免同一按键既走全局派发（重复步进设置）、又在本地再切一次；
            // - 避免叠按「精细调整」修饰键时命中其它全局兜底（如 Ctrl+D 克隆轨道）。
            if (document.body.hasAttribute("data-piano-roll-param-drag-active")) {
                const cycleKb = keybindingsRef.current["pianoRoll.cycleDragDirection"];
                const fineAdjustKb = keybindingsRef.current["modifier.paramFineAdjust"];
                if (
                    cycleKb &&
                    !cycleKb.modifierOnly &&
                    matchesKeybindingAllowingFineModifier(e, cycleKb, fineAdjustKb)
                ) {
                    return;
                }
            }

            const domain = computeFocusDomain();

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

            // OS 自动重复：仅 REPEATABLE_ACTIONS（快进/缩放/轨道导航）走系统
            // 重复，其余一律吞掉 —— 复制/剪切等编辑操作若放行重复，会以键率
            // 向后端发送重复请求；粘贴的连续重复由 holdRepeat 计时器管理。
            if (e.repeat && !REPEATABLE_ACTIONS.has(matched)) {
                e.preventDefault();
                return;
            }

            // ── 编辑操作统一路由（复制/剪切/粘贴、全选、clip 专有操作等）──
            // clip.* 与 pianoRoll.* 的同义绑定归一为同一编辑 op，由 handler
            // 按活动编辑表面定向派发到唯一执行者（事件名即契约，消费者不再
            // 自行判断焦点）。表面未知时 handler 按 no-op 处理。剪贴板组
            // （pianoRoll.copy/cut/paste）虽属 paramEditorSelect 作用域，但
            // 必须先于下方的"作用域放行"检查路由，否则会被错误交回卷帘本地。
            if (ACTION_TO_EDIT_OP[matched]) {
                e.preventDefault();
                e.stopPropagation();
                handlerRef.current(matched);
                return;
            }

            // 参数编辑器作用域的其余快捷键（edit.* 对话框类：移调/量化/平均
            // 化…）由卷帘 scroller 本地 onKeyDown 处理 —— 需要打开
            // openEditDialog 弹窗，不经过全局路由（与既有行为一致）。
            const matchedScope = ACTION_META[matched]?.scopedContext;
            if (matchedScope === "paramEditorSelect" || matchedScope === "pianoRollVibratoDrag") {
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
