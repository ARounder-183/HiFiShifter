/**
 * 参数块命令面的唯一描述表。
 *
 * 参数块右键菜单从这里生成条目；差异只在 `ParamBlockMenuContext` 的瞬时
 * 可用态，不在文案或动作名上。
 */
import type { MessageKey } from "../../../i18n/messages";

export type ParamBlockMenuChannel = "paramBlock" | "edit";

export interface ParamBlockMenuContext {
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
    readonly isVolume: boolean;
    readonly isDyn: boolean;
}

export interface ParamBlockMenuItem {
    readonly key: string;
    readonly labelKey: MessageKey;
    readonly op: string;
    readonly channel: ParamBlockMenuChannel;
    readonly shortcutAction?: string;
    readonly danger?: boolean;
    readonly separatorBefore?: boolean;
    readonly enabled: (ctx: ParamBlockMenuContext) => boolean;
}

const selectionAvailable = (ctx: ParamBlockMenuContext) => ctx.hasSelection;

export const PARAM_BLOCK_MENU_ITEMS: readonly ParamBlockMenuItem[] = [
    {
        key: "split",
        labelKey: "param_blocks_split_at_playhead",
        op: "split",
        channel: "paramBlock",
        shortcutAction: "paramBlock.split",
        enabled: (ctx) => ctx.canSplit(),
    },
    {
        key: "create-from-selection",
        labelKey: "param_blocks_create_from_selection",
        op: "createFromSelection",
        channel: "paramBlock",
        enabled: (ctx) => ctx.hasSelection,
    },
    {
        key: "group",
        labelKey: "param_blocks_group",
        op: "group",
        channel: "paramBlock",
        shortcutAction: "paramBlock.group",
        separatorBefore: true,
        enabled: (ctx) => ctx.canGroup,
    },
    {
        key: "ungroup",
        labelKey: "param_blocks_ungroup",
        op: "ungroup",
        channel: "paramBlock",
        shortcutAction: "paramBlock.ungroup",
        enabled: (ctx) => ctx.canUngroup,
    },
    {
        key: "restore-game",
        labelKey: "param_blocks_restore_game",
        op: "restoreGame",
        channel: "paramBlock",
        shortcutAction: "paramBlock.restoreGame",
        separatorBefore: true,
        enabled: (ctx) => ctx.canRestoreGame,
    },
    {
        key: "restore-deleted",
        labelKey: "param_blocks_restore_deleted",
        op: "restoreDeleted",
        channel: "paramBlock",
        shortcutAction: "paramBlock.restoreDeleted",
        enabled: (ctx) => ctx.canRestoreDeleted,
    },
    {
        key: "normalize-pitch",
        labelKey: "param_blocks_normalize_pitch",
        op: "normalizePitch",
        channel: "paramBlock",
        shortcutAction: "paramBlock.normalizePitch",
        enabled: (ctx) => ctx.isPitch && ctx.canNormalizePitch,
    },
    {
        key: "copy",
        labelKey: "menu_copy",
        op: "copy",
        channel: "edit",
        shortcutAction: "pianoRoll.copy",
        separatorBefore: true,
        // copy/cut/initialize 通过 handleEditOp 支持隐式全选；菜单不应比快捷键更严。
        enabled: () => true,
    },
    {
        key: "cut",
        labelKey: "menu_cut",
        op: "cut",
        channel: "edit",
        shortcutAction: "pianoRoll.cut",
        enabled: () => true,
    },
    {
        key: "paste",
        labelKey: "menu_paste",
        op: "paste",
        channel: "edit",
        shortcutAction: "pianoRoll.paste",
        enabled: () => true,
    },
    {
        key: "initialize",
        labelKey: "menu_initialize",
        op: "initialize",
        channel: "edit",
        separatorBefore: true,
        enabled: () => true,
    },
    {
        key: "transpose-cents",
        labelKey: "menu_transpose_cents",
        op: "transposeCents",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "transpose-degrees",
        labelKey: "menu_transpose_degrees",
        op: "transposeDegrees",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "set-pitch",
        labelKey: "menu_set_pitch",
        op: "setPitch",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "average",
        labelKey: "menu_average",
        op: "average",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "smooth",
        labelKey: "menu_smooth",
        op: "smooth",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "add-vibrato",
        labelKey: "menu_add_vibrato",
        op: "addVibrato",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "quantize",
        labelKey: "menu_quantize",
        op: "quantize",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "mean-quantize",
        labelKey: "menu_mean_quantize",
        op: "meanQuantize",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "save-as-pitch-ref",
        labelKey: "menu_save_as_pitch_ref",
        op: "saveAsPitchRef",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "export-midi",
        labelKey: "menu_export_midi",
        op: "exportMidi",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "convert-volume-to-dyn",
        labelKey: "menu_convert_volume_to_dyn",
        op: "convertVolumeToDyn",
        channel: "edit",
        separatorBefore: true,
        enabled: (ctx) => ctx.isVolume,
    },
    {
        key: "convert-dyn-to-volume",
        labelKey: "menu_convert_dyn_to_volume",
        op: "convertDynToVolume",
        channel: "edit",
        enabled: (ctx) => ctx.isDyn,
    },
    {
        key: "select-all",
        labelKey: "menu_select_all",
        op: "selectAll",
        channel: "edit",
        shortcutAction: "paramBlock.selectAll",
        separatorBefore: true,
        enabled: () => true,
    },
    {
        key: "deselect",
        labelKey: "menu_deselect",
        op: "deselect",
        channel: "edit",
        shortcutAction: "paramBlock.deselect",
        enabled: (ctx) => ctx.hasSelection,
    },
    {
        key: "delete",
        labelKey: "param_blocks_delete",
        op: "delete",
        channel: "paramBlock",
        shortcutAction: "paramBlock.delete",
        danger: true,
        separatorBefore: true,
        enabled: (ctx) => ctx.canDelete,
    },
];
