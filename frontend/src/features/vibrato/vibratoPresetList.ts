/**
 * 预设列表的解析与循环切换。
 *
 * 【分层的理由】系统预设在代码里（`systemPresets.ts`），用户预设在设置里
 * （`app_config.json`）。两者合成一份"可用预设列表"的逻辑只有这里一份 ——
 * 拖拽切换、工具栏选择器、右键菜单、预设编辑器都从这里取。
 */

import {
    DEFAULT_ACTIVE_VIBRATO_PRESET_ID,
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
): ResolvedVibratoPresets {
    const system = SYSTEM_VIBRATO_PRESETS;
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
    const from = user.findIndex((preset) => preset.id === id);
    if (from < 0) return [...user];
    const clamped = Math.min(user.length - 1, Math.max(0, Math.round(toIndex)));
    if (clamped === from) return [...user];
    const next = [...user];
    const [moved] = next.splice(from, 1);
    next.splice(clamped, 0, moved);
    return next;
}
