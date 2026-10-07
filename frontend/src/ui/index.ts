/**
 * 设计系统入口（`@hs/ui`）。
 *
 * 【这一层是什么】令牌层（CSS 变量）→ **原语层（本目录）** → 组合壳
 * （`AppDialog` / `AppContextMenu`）。
 *
 * 三者按变更频率分层：令牌数月不变，原语按月演进，组合壳可能被推翻。
 * 混成一个"控件基类"后，改任一层都会迫使另外两层跟着动。
 *
 * 【两个不在契约里的名字】这里曾把 `AppToolbar` / `AppPanel` 一并列为组合壳。
 * 前者只是 `PanelToolbar` 的实现细节（面板条是本仓库自己的用例，不是给第三方的
 * 通用壳），后者**从未实现**。把没交付的东西写进公开契约，会让读注释的人以为
 * 自己漏看了一个模块 —— 因此把它们删掉，而不是补一个空壳凑数。
 *
 * 【对扩展 API 的意义】本 barrel 就是第三方控件的公开契约。第三方用
 * `AppButton` 即自动跟随主题与尺寸体系；若坚持自造控件，只要消费
 * `--qt-*` 令牌（见 `src/index.css`）也能继承用户的自定义主题。
 * 因此新增原语时请一并在此导出，并保证其视觉取值只来自令牌层。
 */
// `cx` / `ClassValue` 刻意不导出：它们只在 src/ui 内部使用（作为原语的实现细节）。
// 对外暴露一个无人使用的类名工具，只会让"公共 API"这个说法变模糊 ——
// 第三方需要类名合并时直接依赖 `clsx` 即可。

export {
    AppButton,
    AppIconButton,
    type AppButtonIntent,
    type AppButtonProps,
    type AppButtonSize,
    type AppIconButtonProps,
    type AppIconSize,
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

export { AppChoiceList, type AppChoiceListProps, type AppChoiceOption } from "./ChoiceList";

export {
    AppSegmentedControl,
    type AppSegmentedControlProps,
    type AppSegmentedOption,
} from "./SegmentedControl";
export {
    AppAnchoredMenu,
    AppContextMenu,
    AppSubMenu,
    type AppAnchoredMenuProps,
    type AppContextMenuProps,
    type AppMenuItemSpec,
    type AppSubMenuProps,
} from "./Menu";

export {
    AppConfirmDialog,
    AppDialog,
    AppNoticeDialog,
    type AppConfirmDialogProps,
    type AppDialogAction,
    type AppDialogProps,
    type AppDialogSize,
    type AppNoticeDialogProps,
} from "./Dialog";

export { useDialogDraft } from "./useDialogDraft";

export {
    useRepeatPress,
    type RepeatPressHandlers,
    type RepeatPressOptions,
} from "./useRepeatPress";

export { useMenuShortcut } from "./useMenuShortcut";

export { isShortcutSuppressed } from "./shortcutScope";

export { AppListRow, type AppListRowDensity, type AppListRowProps } from "./ListRow";

export { AppSelect, type AppSelectEntry, type AppSelectItem, type AppSelectProps } from "./Select";

export { AppNumberField, type AppNumberFieldProps } from "./NumberField";

export { AppSlider, AppSliderReadout, type AppSliderProps } from "./Slider";

export { AppBusy, AppEmptyState, type AppBusyProps, type AppEmptyStateProps } from "./State";

export { AppStatusChip, type AppStatusChipProps, type AppStatusTone } from "./StatusChip";
