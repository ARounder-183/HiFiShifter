/*
 * 停靠系统设置：类型、默认值、归一化。
 *
 * 与 `notebookSettings.ts` 同一套路：独立成文件而不是塞进通用 UI 设置，因为
 * 这一组选项只被停靠内核消费，且每项都需要"非法值退回默认"的收敛逻辑。
 * `UiSettings.dock` 只是一个可缺省的嵌套对象 —— 旧配置文件里没有它，归一化
 * 会补全默认值，因此新增字段永远不需要配置迁移。
 *
 * 【只收录真正被消费的选项】不预留"以后可能有用"的开关：一个改了没反应的
 * 选项比没有这个选项更糟，它会让用户以为功能坏了。
 */

export interface DockSettings {
    /**
     * 停靠修饰键。
     *
     * `primary` = Windows/Linux 的 Ctrl、macOS 的 Cmd（复用 `utils/platform.ts`
     * 的既有判定）。按住它拖拽时**优先尝试停靠**；不按时默认浮动 —— 与 VEGAS
     * 的停靠修饰键行为一致。
     */
    dockModifier?: "primary" | "alt" | "shift" | "none";
    /** 边缘停靠感应带宽度（像素）。 */
    edgeBandPx?: number;
    /** 拖拽时显示落点预览框。 */
    showDropPreview?: boolean;
    /** 单标签时是否仍显示标签条（关 = 更紧凑，REAPER 风格）。 */
    showTabBarWhenSingle?: boolean;
    /** 标签条是否显示图标。 */
    tabIcons?: boolean;
    /** 双击窗体标题/标签的行为。 */
    doubleClickHeaderAction?: "toggleFloat" | "maximize" | "collapse" | "none";
    /** 浮动窗是否互相/向视口边缘吸附。 */
    floatSnapEnabled?: boolean;
    /** 吸附阈值（像素）。 */
    floatSnapThresholdPx?: number;
    /** 启动时是否恢复上次的浮动窗体（关 = 浮动窗体一律回停靠位）。 */
    floatRestoreOnStartup?: boolean;
    /** 布局写入去抖（毫秒）—— 纯尺寸拖动会高频触发。 */
    saveDebounceMs?: number;
    /** 重置布局前是否确认。 */
    confirmResetLayout?: boolean;
    /**
     * 启动时套用的布局。
     *
     * `last` = 恢复上次关闭时的排布；其余值视为预设名。多显示器/多工种用户
     * 可以固定成"混音布局"开机即用。
     */
    startupLayout?: string;
}

export type ResolvedDockSettings = Required<DockSettings>;

/**
 * 落盘形状：行为选项平铺 + 一份完整布局。
 *
 * 两者放在同一个 `UiSettings.dock` 对象里，是为了**一次写入同时落盘** ——
 * 后端 `save_ui_settings` 是"读-改-写整个配置文件"，两个独立键各写一次会
 * 平白多一轮文件往返，也让并发写入的窗口翻倍（该函数没有跨调用锁）。
 * `"dock"` 加入后端的深度合并白名单后，`layout` 作为一个键被整体替换，
 * 语义正好。
 */
export interface DockPersistedSettings extends DockSettings {
    layout?: unknown;
}

export const DEFAULT_DOCK_SETTINGS: ResolvedDockSettings = {
    dockModifier: "primary",
    edgeBandPx: 28,
    showDropPreview: true,
    showTabBarWhenSingle: false,
    tabIcons: true,
    doubleClickHeaderAction: "toggleFloat",
    floatSnapEnabled: true,
    floatSnapThresholdPx: 12,
    floatRestoreOnStartup: true,
    saveDebounceMs: 400,
    confirmResetLayout: true,
    startupLayout: "last",
};

const MODIFIERS: ReadonlyArray<ResolvedDockSettings["dockModifier"]> = [
    "primary",
    "alt",
    "shift",
    "none",
];
const DOUBLE_CLICK_ACTIONS: ReadonlyArray<ResolvedDockSettings["doubleClickHeaderAction"]> = [
    "toggleFloat",
    "maximize",
    "collapse",
    "none",
];

function clampInt(value: unknown, min: number, max: number, fallback: number): number {
    if (typeof value !== "number" || !Number.isFinite(value)) return fallback;
    return Math.min(max, Math.max(min, Math.round(value)));
}

function pickEnum<T extends string>(value: unknown, allowed: readonly T[], fallback: T): T {
    return typeof value === "string" && (allowed as readonly string[]).includes(value)
        ? (value as T)
        : fallback;
}

/** 归一化停靠设置（非法值一律退回默认，永不抛错）。 */
export function normalizeDockSettings(
    input: DockSettings | null | undefined,
): ResolvedDockSettings {
    const raw = (input ?? {}) as DockSettings;
    return {
        dockModifier: pickEnum(raw.dockModifier, MODIFIERS, DEFAULT_DOCK_SETTINGS.dockModifier),
        edgeBandPx: clampInt(raw.edgeBandPx, 8, 120, DEFAULT_DOCK_SETTINGS.edgeBandPx),
        showDropPreview: raw.showDropPreview ?? DEFAULT_DOCK_SETTINGS.showDropPreview,
        showTabBarWhenSingle:
            raw.showTabBarWhenSingle ?? DEFAULT_DOCK_SETTINGS.showTabBarWhenSingle,
        tabIcons: raw.tabIcons ?? DEFAULT_DOCK_SETTINGS.tabIcons,
        doubleClickHeaderAction: pickEnum(
            raw.doubleClickHeaderAction,
            DOUBLE_CLICK_ACTIONS,
            DEFAULT_DOCK_SETTINGS.doubleClickHeaderAction,
        ),
        floatSnapEnabled: raw.floatSnapEnabled ?? DEFAULT_DOCK_SETTINGS.floatSnapEnabled,
        floatSnapThresholdPx: clampInt(
            raw.floatSnapThresholdPx,
            0,
            64,
            DEFAULT_DOCK_SETTINGS.floatSnapThresholdPx,
        ),
        floatRestoreOnStartup:
            raw.floatRestoreOnStartup ?? DEFAULT_DOCK_SETTINGS.floatRestoreOnStartup,
        saveDebounceMs: clampInt(raw.saveDebounceMs, 0, 5000, DEFAULT_DOCK_SETTINGS.saveDebounceMs),
        confirmResetLayout: raw.confirmResetLayout ?? DEFAULT_DOCK_SETTINGS.confirmResetLayout,
        startupLayout:
            typeof raw.startupLayout === "string" && raw.startupLayout
                ? raw.startupLayout
                : DEFAULT_DOCK_SETTINGS.startupLayout,
    };
}
