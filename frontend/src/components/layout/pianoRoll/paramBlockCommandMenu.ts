/**
 * 参数块**专有命令**的唯一描述表。
 *
 * 这里只放“块”这一层才存在的命令（分割 / 成组 / 还原 / 删除…）。复制、剪切、
 * 粘贴、初始化、移调、平滑、量化、导出等标准参数命令属于 `EditContextMenu`
 * 的共享描述表，绝不在这里再抄一份 —— 两张表迟早会在快捷键文本与禁用条件上
 * 漂移，而且块菜单会重新膨胀成 26 项平铺列表。
 */
import type { ActionId } from "../../../features/keybindings/types";
import type { MessageKey } from "../../../i18n/messages";

/** 块命令只走这一个动作通道；标准参数命令仍由 `EditContextMenu` 走 edit。 */
export type ParamBlockCommandChannel = "paramBlock";

export interface ParamBlockCommandContext {
    /** 实时求值：菜单渲染时读取播放头与当前块视图，不能保存打开瞬间的快照。 */
    readonly canSplit: () => boolean;
    readonly canGroup: boolean;
    readonly canUngroup: boolean;
    readonly canDelete: boolean;
    readonly canRestoreGame: boolean;
    readonly canRestoreDeleted: boolean;
    readonly canNormalizePitch: boolean;
    readonly hasSelection: boolean;
    readonly isPitch: boolean;
}

export interface ParamBlockCommandItem {
    readonly key: string;
    readonly labelKey: MessageKey;
    readonly op: string;
    readonly shortcutAction?: ActionId;
    readonly danger?: boolean;
    readonly separatorBefore?: boolean;
    /** 仅控制“是否出现在菜单里”；可用性由 `enabled` 表达。 */
    readonly visible?: (ctx: ParamBlockCommandContext) => boolean;
    readonly enabled: (ctx: ParamBlockCommandContext) => boolean;
}

export const PARAM_BLOCK_COMMANDS: readonly ParamBlockCommandItem[] = [
    {
        key: "split",
        labelKey: "param_blocks_split_at_playhead",
        op: "split",
        shortcutAction: "paramBlock.split",
        visible: (ctx) => ctx.canSplit(),
        enabled: (ctx) => ctx.canSplit(),
    },
    {
        key: "create-from-selection",
        labelKey: "param_blocks_create_from_selection",
        op: "createFromSelection",
        visible: (ctx) => ctx.hasSelection,
        enabled: (ctx) => ctx.hasSelection,
    },
    {
        key: "group",
        labelKey: "param_blocks_group",
        op: "group",
        shortcutAction: "paramBlock.group",
        visible: (ctx) => ctx.canGroup,
        enabled: (ctx) => ctx.canGroup,
    },
    {
        key: "ungroup",
        labelKey: "param_blocks_ungroup",
        op: "ungroup",
        shortcutAction: "paramBlock.ungroup",
        visible: (ctx) => ctx.canUngroup,
        enabled: (ctx) => ctx.canUngroup,
    },
    {
        key: "restore-game",
        labelKey: "param_blocks_restore_game",
        op: "restoreGame",
        shortcutAction: "paramBlock.restoreGame",
        visible: (ctx) => ctx.canRestoreGame,
        enabled: (ctx) => ctx.canRestoreGame,
    },
    {
        key: "restore-deleted",
        labelKey: "param_blocks_restore_deleted",
        op: "restoreDeleted",
        shortcutAction: "paramBlock.restoreDeleted",
        visible: (ctx) => ctx.canRestoreDeleted,
        enabled: (ctx) => ctx.canRestoreDeleted,
    },
    {
        key: "normalize-pitch",
        labelKey: "param_blocks_normalize_pitch",
        op: "normalizePitch",
        shortcutAction: "paramBlock.normalizePitch",
        visible: (ctx) => ctx.isPitch && ctx.canNormalizePitch,
        enabled: (ctx) => ctx.isPitch && ctx.canNormalizePitch,
    },
    {
        key: "delete",
        labelKey: "param_blocks_delete",
        op: "delete",
        shortcutAction: "paramBlock.delete",
        danger: true,
        separatorBefore: true,
        visible: (ctx) => ctx.canDelete,
        enabled: (ctx) => ctx.canDelete,
    },
];
