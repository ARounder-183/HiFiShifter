/**
 * 预设列表的解析与循环切换。
 *
 * 【分层的理由】系统预设在代码里（`systemPresets.ts`），用户预设在设置里
 * （`app_config.json`）。两者合成一份"可用预设列表"的逻辑只有这里一份 ——
 * 拖拽切换、工具栏选择器、右键菜单、预设编辑器都从这里取。
 */

import {
    DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
    STRAIGHT_VIBRATO_PRESET_ID,
    SYSTEM_VIBRATO_PRESETS,
    builtinVibratoPresetId,
    type BuiltinVibratoId,
} from "./systemPresets";
import {
    dedupeVibratoPresets,
    isBuiltinVibratoPresetId,
    MAX_VIBRATO_PRESETS,
    sanitizeVibratoPreset,
} from "./vibratoPresets";
import type { VibratoPreset } from "./vibratoTypes";

/** 系统预设在前、用户预设在后。 */
export interface ResolvedVibratoPresets {
    system: readonly VibratoPreset[];
    user: VibratoPreset[];
    /** 系统 + 用户，顺序即 UI 顺序。 */
    all: VibratoPreset[];
}

/**
 * 合成可用预设列表。
 *
 * 系统预设复用模块级的那份稳定数组（对象标识不随调用变化，方便 React
 * 依赖比较）；用户预设每次经 `sanitizeVibratoPreset` 之后才返回，因此
 * 上游不会看到未规整的数据。
 *
 * 用户预设里若混进了 `builtin.` 前缀的 id（手改配置），会被剔除 —— 系统
 * 预设只能来自代码，否则"系统预设只读"的保证就有了漏洞。
 */
export function resolveVibratoPresets(
    userPresets: readonly VibratoPreset[] | null | undefined,
    builtinOrder?: readonly string[] | null,
): ResolvedVibratoPresets {
    const system = orderSystemPresets(builtinOrder);
    const systemIds = new Set(system.map((preset) => preset.id));
    const user = dedupeVibratoPresets(
        (userPresets ?? [])
            .map((preset) => sanitizeVibratoPreset(preset))
            .filter((preset) => !isBuiltinVibratoPresetId(preset.id) && !systemIds.has(preset.id)),
        // 手改过的配置可能远超上限：这里截断而不是照单全收，避免
        // 把一份病态配置一路拖进菜单与渲染。
    ).slice(0, MAX_VIBRATO_PRESETS);
    return { system, user, all: [...system, ...user] };
}

/**
 * 系统预设的**有效顺序**（id 列表）。
 *
 * 持久化的顺序可能缺项（新版本新增了出厂预设）、也可能含无效项（旧版本删过），
 * 因此一律以出厂顺序为底：先按持久化顺序取有效项，再把没提到的按出厂顺序补在后面。
 * 这样"新增一个出厂预设"不需要迁移任何配置。
 */
export function effectiveBuiltinPresetOrder(order?: readonly string[] | null): string[] {
    const defaultIds = SYSTEM_VIBRATO_PRESETS.map((preset) => preset.id);
    if (!order || order.length === 0) return defaultIds;
    const known = new Set(defaultIds);
    const result: string[] = [];
    const seen = new Set<string>();
    for (const id of order) {
        if (!known.has(id) || seen.has(id)) continue;
        seen.add(id);
        result.push(id);
    }
    for (const id of defaultIds) {
        if (!seen.has(id)) result.push(id);
    }
    return result;
}

/**
 * 按持久化顺序排列系统预设。
 *
 * 顺序与出厂一致（含"没有自定义顺序"）时返回模块级那份**稳定数组** —— 上游拿它做
 * React 依赖比较，每次返回新数组会让整棵子树白白重渲染。
 */
function orderSystemPresets(order?: readonly string[] | null): readonly VibratoPreset[] {
    const ids = effectiveBuiltinPresetOrder(order);
    if (ids.every((id, index) => id === SYSTEM_VIBRATO_PRESETS[index]?.id)) {
        return SYSTEM_VIBRATO_PRESETS;
    }
    const byId = new Map(SYSTEM_VIBRATO_PRESETS.map((preset) => [preset.id, preset]));
    return ids
        .map((id) => byId.get(id))
        .filter((preset): preset is VibratoPreset => Boolean(preset));
}

/** 取出厂预设（按稳定键）。 */
export function systemVibratoPreset(id: BuiltinVibratoId): VibratoPreset {
    const found = SYSTEM_VIBRATO_PRESETS.find((preset) => preset.id === builtinVibratoPresetId(id));
    // `BUILTIN_VIBRATO_ORDER` 与 specs 表同源，找不到只可能是构造逻辑被改坏了。
    return found ?? sanitizeVibratoPreset({ id: builtinVibratoPresetId(id), builtin: true });
}

/** 按 id 查找；找不到返回 `undefined`。 */
export function findVibratoPreset(
    all: readonly VibratoPreset[],
    id: string | null | undefined,
): VibratoPreset | undefined {
    if (!id) return undefined;
    return all.find((preset) => preset.id === id);
}

/**
 * 解析「当前生效的预设」。
 *
 * 找不到（预设被删、配置来自更旧的版本）时回落到出厂默认，**不写回设置**
 * —— 静默改写用户的活动选择会让"删掉一个预设"顺带改掉当前音色。
 */
export function resolveActiveVibratoPreset(
    all: readonly VibratoPreset[],
    activeId: string | null | undefined,
): VibratoPreset {
    return (
        findVibratoPreset(all, activeId) ??
        findVibratoPreset(all, DEFAULT_ACTIVE_VIBRATO_PRESET_ID) ??
        all[0] ??
        sanitizeVibratoPreset({ id: DEFAULT_ACTIVE_VIBRATO_PRESET_ID, builtin: true })
    );
}

/**
 * 过滤掉被停用的预设。
 *
 * 【停用影响什么】只影响"本机怎么挑预设"：参数编辑器工具栏的列表、以及拖拽中按
 * 快捷键 / 侧键的循环切换。它**不影响**预设本身能否被编辑、能否作为当前预设 ——
 * 因此过滤只发生在取"可选列表"的地方，管理器与活动预设的解析仍看全量列表。
 */
export function enabledVibratoPresets(
    all: readonly VibratoPreset[],
    disabledIds: readonly string[] | null | undefined,
): VibratoPreset[] {
    if (!disabledIds || disabledIds.length === 0) return [...all];
    const disabled = new Set(disabledIds);
    return all.filter((preset) => !disabled.has(preset.id));
}

/**
 * 在预设列表中移动 `delta` 步（环绕），返回新的预设 id。
 *
 * 列表为空时返回 `null`。`delta` 为正表示下一个。
 */
export function cycleVibratoPresetId(
    all: readonly VibratoPreset[],
    activeId: string | null | undefined,
    delta: 1 | -1,
): string | null {
    if (all.length === 0) return null;
    const current = resolveActiveVibratoPreset(all, activeId);
    const index = all.findIndex((preset) => preset.id === current.id);
    const from = index >= 0 ? index : 0;
    const next = (from + delta + all.length) % all.length;
    return all[next].id;
}

/**
 * 一次"选预设"的结果。
 *
 * 【为什么不是 id】直线不再是普通预设，而是与颤音并列的**工具**：选中「直线」
 * 预设的语义是切到直线工具，而不是把活动预设改成一条平线。把这个映射收进类型，
 * 调用方（键盘轮转、侧键、预设子菜单）就无法各自解释错。
 */
export type VibratoChoice = { tool: "line" } | { tool: "vibrato"; presetId: string };

/** 唯一的「切到直线工具」结果，供调用方复用（避免每处新建对象）。 */
export const VIBRATO_LINE_CHOICE: VibratoChoice = { tool: "line" };

/** 是否为直线预设的 id（缺省 / 非字符串一律视为否）。 */
export function isStraightVibratoPresetId(id: string | null | undefined): boolean {
    return typeof id === "string" && id === STRAIGHT_VIBRATO_PRESET_ID;
}

/**
 * 轮转一格，并把结果翻译成"切工具"还是"切预设"。
 *
 * 【环绕规则没有第二套】内部仍走 {@link cycleVibratoPresetId}，只是在落点上做
 * 一次映射 —— 直线预设占着序列里的一格，落在它上面就等于切到直线工具。
 *
 * 列表为空返回 `null`。
 */
export function cycleVibratoChoice(
    all: readonly VibratoPreset[],
    anchorId: string | null | undefined,
    delta: 1 | -1,
): VibratoChoice | null {
    const nextId = cycleVibratoPresetId(all, anchorId, delta);
    if (nextId === null) return null;
    return chooseVibratoPreset(nextId);
}

/** 直接选中某个预设 id：直线预设 → 直线工具，其余 → 颤音工具 + 该预设。 */
export function chooseVibratoPreset(presetId: string | null | undefined): VibratoChoice {
    if (isStraightVibratoPresetId(presetId)) return VIBRATO_LINE_CHOICE;
    return { tool: "vibrato", presetId: String(presetId) };
}

/**
 * 轮转的起点 id。
 *
 * 【直线工具为什么不读活动预设】直线工具不是"用某个预设的颤音工具"，它固定在直线
 * 预设上（见 `STRAIGHT_VIBRATO_PRESET_ID`）。若拿活动预设当起点，用户从直线工具
 * 轮转一步会跳到活动预设的**下一个**而不是直线预设的下一个 —— 换了工具却不换
 * 位置，与"直线工具就是直线预设"这条规则自相矛盾。
 */
export function vibratoCycleAnchorId(args: {
    lineTool: boolean;
    currentPresetId: string | null | undefined;
}): string | null {
    return args.lineTool ? STRAIGHT_VIBRATO_PRESET_ID : (args.currentPresetId ?? null);
}

/**
 * 删除预设后，把活动 id 迁移到一个仍然存在的预设上。
 *
 * @returns 应当写入设置的活动 id；`removedId` 不是当前活动项时原样返回。
 */
export function activeIdAfterRemoval(
    all: readonly VibratoPreset[],
    activeId: string,
    removedId: string,
): string {
    if (activeId !== removedId) return activeId;
    const index = all.findIndex((preset) => preset.id === removedId);
    const remaining = all.filter((preset) => preset.id !== removedId);
    if (remaining.length === 0) return DEFAULT_ACTIVE_VIBRATO_PRESET_ID;
    const fallback = remaining[Math.min(Math.max(index, 0), remaining.length - 1)];
    return fallback?.id ?? DEFAULT_ACTIVE_VIBRATO_PRESET_ID;
}

/**
 * 把第 `from` 项移动到 `toIndex`（**移除之后**的数组下标语义）。
 *
 * 越界索引被钳制；`from` 无效时原样返回。抽出来是为了让"用户预设"与"系统预设
 * 顺序"两份排序共用同一套边界规则 —— 拖到首 / 尾、越界、id 不存在这几处最容易
 * 各自漂移。
 */
export function moveItemToIndex<T>(items: readonly T[], from: number, toIndex: number): T[] {
    if (from < 0 || from >= items.length) return [...items];
    const clamped = Math.min(items.length - 1, Math.max(0, Math.round(toIndex)));
    if (clamped === from) return [...items];
    const next = [...items];
    const [moved] = next.splice(from, 1);
    next.splice(clamped, 0, moved);
    return next;
}

/**
 * 把用户预设数组中的一项移动到新位置（仅用户段内部）。
 *
 * 纯函数：返回新数组，越界索引被钳制。抽出来是为了单测 —— 列表排序
 * 的边界（拖到首/尾、越界、id 不存在）比 UI 更容易出错。
 */
export function reorderUserVibratoPresets(
    user: readonly VibratoPreset[],
    id: string,
    toIndex: number,
): VibratoPreset[] {
    return moveItemToIndex(
        user,
        user.findIndex((preset) => preset.id === id),
        toIndex,
    );
}

/** 把系统预设顺序里的某个 id 移动到新位置（系统预设的顺序以 id 列表持久化）。 */
export function reorderBuiltinPresetIds(
    order: readonly string[],
    id: string,
    toIndex: number,
): string[] {
    return moveItemToIndex(order, order.indexOf(id), toIndex);
}
