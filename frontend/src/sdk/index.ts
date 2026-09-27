/**
 * HiFiShifter 扩展 API（`@hs/sdk`）—— **对外唯一入口**。
 *
 * ## 为什么需要这个文件
 *
 * 审查结论：停靠内核本身已经为扩展做好了准备 —— 面板注册中心是运行时可变、
 * 可订阅的，内置面板与未来的第三方面板走**完全相同**的注册路径，
 * `dockApi.ts` 的头部注释也早已写明「将来开放 API 时暴露的就是这几个函数」。
 * 缺的是内核**周围**的一切：没有对外入口、没有可编译的类型、没有 UI 原语、
 * 面板配置只可读不可写。
 *
 * 本文件补上「对外入口」这一件：第三方只需 `import { ... } from "@hs/sdk"`，
 * 不必知道 `src/features/dock/...` 的内部路径。
 *
 * ## 分层
 *
 * | 层 | 内容 | 稳定性 |
 * |---|---|---|
 * | 令牌 | CSS 变量 `--qt-*`（见 `src/index.css`） | **最稳定**，可直接读 `var()` |
 * | 原语 | `AppButton` / `AppDialog` / `AppContextMenu` / … | 按月演进 |
 * | 组合壳 | `AppDialog` 的交互协议（Enter/Esc/快捷键抑制） | 可内部重构 |
 * | 面板 | `registerPanel` / `DockPanelProps` / `dockApi` | 契约，有版本号 |
 *
 * ## 稳定性承诺
 *
 * - `SDK_VERSION` 是**契约版本**，不是应用版本。破坏性变更必须递增它。
 * - 面板 id 一旦发布即不可更改：它会写进用户的布局存档
 *   （`dockSchema` 的归一化会丢弃"面板已不存在"的记录）。
 * - 本文件导出的类型视为契约；未在此导出的内部类型**随时可变**。
 */

// ── 面板注册（扩展的核心入口）────────────────────────────────────────
export {
    registerPanel,
    setPanelComponent,
    getPanel,
    listPanels,
    isPanelRegistered,
    getPanelRegistryVersion,
    subscribePanels,
    resetPanelRegistryForTests,
    type DockPanelLifecycle,
    type DockPanelProps,
    type PanelDefinition,
} from "../features/dock/panelRegistry";

// ── 停靠命令门面（菜单 / 快捷键 / 工具栏按钮共用的同一份行为）─────────
export {
    activeFormId,
    closeFormById,
    cycleFocus,
    deletePreset,
    detachFormToWindow,
    exportLayoutJson,
    getPanelProps,
    importLayoutJson,
    isPanelVisible,
    isRegistered,
    listPanelEntries,
    listPresetNames,
    maximizeActive,
    openPanelById,
    reclaimDetachedForm,
    resetLayout,
    savePreset,
    applyPreset,
    selectPanelVisible,
    setFormPropsById,
    setPanelProps,
    toggleFloatActive,
    togglePanelVisible,
    type DockPanelEntry,
    type GetState,
} from "../features/dock/dockApi";

// ── 布局类型 ────────────────────────────────────────────────────────
export type { DockForm, DockLayout, DockPlacement } from "../features/dock/dockTypes";

// ── UI 原语与组合壳（视觉契约）──────────────────────────────────────
export * from "../ui";

// ── 主题 ────────────────────────────────────────────────────────────
export { useAppTheme } from "../theme/AppThemeProvider";
export {
    QT_COLOR_TOKENS,
    QT_COLOR_TOKEN_LABELS,
    type AppearanceSettings,
    type CustomTheme,
    type QtColorToken,
} from "../theme/themeTypes";

// ── 本地化 ──────────────────────────────────────────────────────────
export { useI18n, translateOutsideReact } from "../i18n/I18nProvider";
export type { Locale } from "../i18n/messages";

// ── 平台工具 ────────────────────────────────────────────────────────
export { IS_LINUX, IS_MAC } from "../utils/platform";

/**
 * 扩展契约版本。
 *
 * 语义化版本，独立于应用版本。第三方可以据此判断自己依赖的表面是否兼容。
 */
export const SDK_VERSION = "0.1.0";
