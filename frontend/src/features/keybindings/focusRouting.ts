/**
 * focusRouting.ts — 快捷键的焦点感知路由（冲突解决核心）
 *
 * 设计目标：不同"焦点域 / 作用域"的快捷键可以善意地共用同一组按键，
 * 由当前焦点决定期望的编辑目标 —— 与复制/粘贴（clip.copy 与
 * pianoRoll.copy 共绑 Ctrl+C，参数编辑器焦点优先参数粘贴）同一套思路。
 *
 * 匹配顺序（同一按键事件命中多个绑定时）：
 *  0. 当前焦点域内激活的作用域操作（如参数编辑器内 select 工具下的
 *     paramEditorSelect 操作；轨道头内的 trackHeaderFocus 操作）—— 最高优先。
 *  1. 全局（无 scopedContext）操作 —— 例如 track.add / clip.* / edit.*。
 *  2. 其它焦点域的作用域操作 —— 仅在没有更优匹配时兜底（例如在时间轴
 *     按 Ctrl+D，轨道头专用的 track.clone 仍是唯一匹配时也照常生效）。
 *  Infinity. 硬排除：当前焦点域下明确不应响应的作用域
 *     （paramEditorSelect 仅在参数编辑器 + select 工具下响应；
 *     quickSearch / pianoRollVibratoDrag 由各自的弹窗与拖拽表面接管）。
 *
 * 这样两个绑定键值相同但作用域不同的操作不再互相"抢占"，也不会被
 * 赋值时的冲突检测拦截（见 keybindingsSlice.findConflicts 的作用域规则）。
 */

import type { ActionId, Keybinding } from "./types";
import { ACTION_META } from "./defaultKeybindings";
import { matchesKeybinding } from "./keybindingMatch";
import type { EditSurfaceId } from "../uiFocus/focusSurface";

/** 全局快捷键路由关心的焦点域。 */
export type KeybindingFocusDomain = "pianoRoll" | "timeline" | "trackHeader" | null;

/** 硬排除的作用域：即使按键唯一匹配也绝不由此全局路由触发。 */
const HARD_EXCLUDED_CONTEXTS = new Set(["quickSearch", "pianoRollVibratoDrag"]);

/**
 * 计算作用域在当前焦点域下的优先级（越小越优先）。
 * - 0：当前焦点域内激活的作用域操作
 * - 1：全局（无作用域）
 * - 2：其它焦点域的作用域操作（兜底）
 * - Infinity：硬排除（不参与匹配）
 */
export function scopePriority(
    scopedContext: string | undefined,
    domain: KeybindingFocusDomain,
    toolMode: string,
): number {
    if (!scopedContext) return 1;
    switch (scopedContext) {
        case "paramEditorSelect":
            // 参数编辑器操作只在钢琴卷帘 + 选择工具下生效；其它焦点域硬排除。
            return domain === "pianoRoll" && toolMode === "select" ? 0 : Infinity;
        case "timelineFocus":
            return domain === "timeline" ? 0 : 2;
        case "trackHeaderFocus":
            return domain === "trackHeader" ? 0 : 2;
        default:
            return HARD_EXCLUDED_CONTEXTS.has(scopedContext) ? Infinity : 2;
    }
}

export const SCOPE_PRIORITY_MAX = Infinity;

/**
 * 在键位映射中查找匹配当前按键事件的最佳 actionId。
 *
 * 该函数替代旧的"按对象插入顺序取第一个匹配"逻辑：改为先收集全部
 * 匹配项，再按「激活作用域 > 全局 > 其它作用域 > 硬排除」排序取首位，
 * 使路由结果与键位表声明顺序无关，键值相同的作用域操作由焦点裁决
 * （与复制/粘贴的焦点分发同构）。
 *
 * @param domain 当前焦点域（pianoRoll / timeline / trackHeader / null）
 * @param toolMode 当前工具（select / draw / vibrato），用于 paramEditorSelect 激活判定
 * @param options.hardExcludeParamEditorSelect 参数编辑器作用域是否硬排除
 *   （默认按 scopePriority 的规则自动判定；无需显式传参）
 */
export function resolveActionByFocus(
    e: KeyboardEvent,
    keybindings: Readonly<Record<string, Keybinding>>,
    domain: KeybindingFocusDomain,
    toolMode: string,
): ActionId | null {
    let best: ActionId | null = null;
    let bestPriority = Infinity;
    for (const [actionId, kb] of Object.entries(keybindings) as [ActionId, Keybinding][]) {
        if (kb.modifierOnly) continue;
        if (!matchesKeybinding(e, kb)) continue;
        const priority = scopePriority(ACTION_META[actionId]?.scopedContext, domain, toolMode);
        if (priority < bestPriority) {
            bestPriority = priority;
            best = actionId;
        }
    }
    return bestPriority === Infinity ? null : best;
}

/**
 * 判断某作用域是否在给定焦点域下"激活"（用于测试与提示文案）。
 */
export function isScopeActive(
    scopedContext: string | undefined,
    domain: KeybindingFocusDomain,
    toolMode: string,
): boolean {
    return scopePriority(scopedContext, domain, toolMode) === 0;
}

// ── 编辑操作路由（复制/剪切/粘贴等冲突的最终裁决） ──────────────────
//
// clip.copy 与 pianoRoll.copy 共绑 Ctrl+C/X/V 只是「同键异义」，裁决依据
// 按操作分派：
// - 复制/剪切：按"当前选中的是什么"路由（选区数据，resolveCopyCutRoute）
//   —— 仅参数线选区 → 参数复制；仅 Clip 选区 → Clip 复制；两者并存按
//   selectionContext（最近被触碰的选区上下文，轨道换轨计入参数侧）仲裁。
// - 粘贴：单剪贴板纪律保证"槽位内容 = 用户最后一次复制的意图"，因此按
//   剪贴板载荷类型定向（resolvePasteRoute，last-copy-wins），点击表面仅
//   在槽位为空/外来数据时兜底 —— 例如"点轨道换轨后在参数编辑器粘贴"
//   这类跨表面工作流不再被点击动作打断。
// 路由把同义 actionId 归一为同一编辑 op 后定向派发 —— 冲突在结构上消解，
// 而不是被特判规则压住。

/** 双表面共享的编辑操作，以及时间轴专有操作。 */
export type EditOp =
    | "copy"
    | "cut"
    | "paste"
    | "selectAll"
    | "deselect"
    | "pasteTracks"
    | "delete"
    | "split"
    | "normalize"
    | "group"
    | "ungroup"
    | "cycleTake"
    | "cycleTakePrev";

/** 事件通道即契约：hifi:editOp 只属于参数编辑器，hifi:timelineEditOp 只属于时间轴。 */
export type EditOpChannel = "hifi:editOp" | "hifi:timelineEditOp";

/**
 * 快捷键 actionId → 编辑操作。clip.* 与 pianoRoll.* 的同义操作归一为同一
 * op（路由只看表面，不看哪个同义绑定赢得了作用域竞争）；参数编辑器专有
 * 操作（shiftParam*、pasteVocalShifter 等）不在此表 —— 它们只有单一归属，
 * 直接走既有派发路径。
 */
export const ACTION_TO_EDIT_OP: Partial<Record<ActionId, EditOp>> = {
    "clip.copy": "copy",
    "clip.cut": "cut",
    "clip.paste": "paste",
    "clip.delete": "delete",
    "clip.split": "split",
    "clip.normalize": "normalize",
    "clip.group": "group",
    "clip.ungroup": "ungroup",
    "clip.cycleTake": "cycleTake",
    "clip.cycleTakePrev": "cycleTakePrev",
    "pianoRoll.copy": "copy",
    "pianoRoll.cut": "cut",
    "pianoRoll.paste": "paste",
    "edit.selectAll": "selectAll",
    "edit.deselect": "deselect",
    "edit.pasteTracks": "pasteTracks",
};

/**
 * 复制/剪切的选择路由 —— 复制按键那一刻内容尚不存在，裁决依据是"当前
 * 选中了什么"，与粘贴的"剪贴板里有什么"（resolvePasteRoute）首尾呼应：
 *
 * - 仅参数线选区存在 → 参数复制（**与点击表面无关**：在时间轴/轨道头换轨
 *   后直接 Ctrl+C，复制的是换轨目标轨道的参数线，无需先点回参数编辑器）；
 * - 仅 Clip 选区存在 → Clip 复制（镜像场景同理）；
 * - 两者并存 → 按 selectionContext（最近被触碰的选区上下文：轨道换轨与
 *   参数选区变化记 "param"，Clip 选中变化记 "clips"）仲裁；
 * - 两者皆空 → 无操作。
 *
 * selectionContext 无记录（启动初期理论上不可达，防御性保留）时退回按
 * 活动表面裁决。
 */
export function resolveCopyCutRoute(args: {
    surface: EditSurfaceId | null;
    clipSelectionActive: boolean;
    paramSelectionActive: boolean;
    selectionContext: "param" | "clips" | null;
}): EditOpChannel | null {
    const { surface, clipSelectionActive, paramSelectionActive, selectionContext } = args;
    if (paramSelectionActive && !clipSelectionActive) return "hifi:editOp";
    if (!paramSelectionActive && clipSelectionActive) return "hifi:timelineEditOp";
    if (!paramSelectionActive && !clipSelectionActive) return null;
    if (selectionContext === "param") return "hifi:editOp";
    if (selectionContext === "clips") return "hifi:timelineEditOp";
    return resolveEditOpRoute(surface, "copy");
}

/**
 * 解析编辑操作的目标通道（事件名）。
 *
 * - copy/cut：**此处仅作兜底** —— 正常路径由 resolveCopyCutRoute 按"当前
 *   选中的是什么"路由（选区数据）；仅双选区并存且无上下文记录时触达。
 * - paste：**此处仅作兜底** —— 正常路径由 resolvePasteRoute 按剪贴板载荷
 *   类型（last-copy-wins）路由；只有槽位为空/外来数据时才落到本函数，
 *   按活动表面选择 REAPER/MIDI 兜底流程的归属。
 * - selectAll/deselect：参数编辑器仅在 select 工具下实现（见
 *   PianoRollPanel.handleEditOp 的工具守卫），draw/vibrato 工具下落回
 *   时间轴全选 Clip（与既有行为一致）。
 * - delete/split/…：时间轴专有操作，固定路由到时间轴通道 —— 无论最后
 *   点击的是哪个表面（裸 Delete 删除选中 Clip 等既有语义保持不变）。
 * - 未收录的 op：返回 null，由调用方走各自的固定通道。
 */
export function resolveEditOpRoute(
    surface: EditSurfaceId | null,
    op: EditOp | string,
    toolMode?: string,
): EditOpChannel | null {
    switch (op) {
        case "copy":
        case "cut":
            if (surface === "pianoRoll") return "hifi:editOp";
            if (surface === "timeline" || surface === "trackHeader") {
                return "hifi:timelineEditOp";
            }
            return null;
        case "paste":
            // 兜底路径：见 resolvePasteRoute（内容路由的主裁决在它之上）。
            if (surface === "pianoRoll") return "hifi:editOp";
            if (surface === "timeline" || surface === "trackHeader") {
                return "hifi:timelineEditOp";
            }
            return null;
        case "selectAll":
        case "deselect":
            if (surface === "pianoRoll" && toolMode === "select") return "hifi:editOp";
            return "hifi:timelineEditOp";
        case "pasteTracks":
        case "delete":
        case "split":
        case "normalize":
        case "group":
        case "ungroup":
        case "cycleTake":
        case "cycleTakePrev":
            return "hifi:timelineEditOp";
        default:
            return null;
    }
}

/**
 * 粘贴的内容路由（last-copy-wins）。
 *
 * 单剪贴板纪律保证"槽位内容 = 用户最后一次复制的意图"，因此粘贴不再按
 * 点击表面裁决，而按载荷类型定向：参数线数据贴回参数编辑器的实时上下文
 * （选区 + 当前轨道 —— 轨道点击换轨后依然成立），Clip 载荷贴到播放头。
 * 槽位为空或承载外来数据（REAPER/MIDI，不携带我们的意图）时，才退回按
 * 活动表面选择兜底流程：参数编辑器表面的 REAPER/MIDI 兜底（含 MIDI 导入
 * 对话框）与时间轴表面的播放头 REAPER 粘贴各归其位。
 *
 * @param clipboardKind 后端 clipboard_kind 探测结果：
 *   "clips" | "tracks" | "project" | "param" | null。
 */
export function resolvePasteRoute(
    clipboardKind: string | null,
    surface: EditSurfaceId | null,
): EditOpChannel | null {
    if (clipboardKind === "param") return "hifi:editOp";
    if (clipboardKind === "clips" || clipboardKind === "tracks" || clipboardKind === "project") {
        return "hifi:timelineEditOp";
    }
    return resolveEditOpRoute(surface, "paste");
}

/** 硬排除是否包含需要被排除的作用域（仅用于单测断言）。 */
export { HARD_EXCLUDED_CONTEXTS };
