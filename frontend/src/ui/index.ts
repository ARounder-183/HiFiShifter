/**
 * 设计系统入口（`@hs/ui`）。
 *
 * 【这一层是什么】令牌层（CSS 变量）→ **原语层（本目录）** → 组合壳
 * （`AppDialog` / `AppToolbar` / `AppContextMenu` / `AppPanel`）。
 *
 * 三者按变更频率分层：令牌数月不变，原语按月演进，组合壳可能被推翻。
 * 混成一个"控件基类"后，改任一层都会迫使另外两层跟着动。
 *
 * 【对扩展 API 的意义】本 barrel 就是第三方控件的公开契约。第三方用
 * `AppButton` 即自动跟随主题与尺寸体系；若坚持自造控件，只要消费
 * `--qt-*` 令牌（见 `src/index.css`）也能继承用户的自定义主题。
 * 因此新增原语时请一并在此导出，并保证其视觉取值只来自令牌层。
 */
export { cx, type ClassValue } from "./cx";

export { AppText, type AppTextProps, type AppTextRole } from "./Text";

export {
    AppButton,
    AppIconButton,
    AppToggleButton,
    type AppButtonIntent,
    type AppButtonProps,
    type AppButtonSize,
    type AppIconButtonProps,
    type AppIconSize,
    type AppToggleButtonProps,
} from "./Button";

export {
    AppForm,
    AppField,
    AppFormSection,
    AppSwitchRow,
    type AppFieldLabelWidth,
    type AppFieldProps,
    type AppFormProps,
    type AppFormSectionProps,
    type AppSwitchRowProps,
} from "./Field";

export {
    AppContextMenu,
    type AppContextMenuProps,
    type AppMenuItemSpec,
} from "./Menu";

export {
    AppConfirmDialog,
    AppDialog,
    type AppConfirmDialogProps,
    type AppDialogAction,
    type AppDialogProps,
    type AppDialogSize,
} from "./Dialog";

export { useDialogDraft } from "./useDialogDraft";

export {
    acquireShortcutSuppression,
    isShortcutSuppressed,
    releaseShortcutSuppression,
    resetShortcutScopesForTests,
} from "./shortcutScope";

export { AppListRow, type AppListRowDensity, type AppListRowProps } from "./ListRow";

export { AppBusy, AppEmptyState, type AppBusyProps, type AppEmptyStateProps } from "./State";

export {
    AppStatusChip,
    type AppStatusChipProps,
    type AppStatusTone,
} from "./StatusChip";
