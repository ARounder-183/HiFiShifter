/**
 * 参数块命令面的唯一描述表。
 *
 * 右键菜单、菜单栏与工具栏溢出菜单都从这里生成条目；差异只在
 * `ParamBlockMenuContext` 的瞬时可用态，不在文案或动作名上。
 */
import type { MessageKey } from "../../../i18n/messages";

export type ParamBlockMenuScope = "block" | "param" | "selection";
export type ParamBlockMenuChannel = "paramBlock" | "edit";

export interface ParamBlockMenuContext {
    readonly canSplit: boolean;
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
    readonly scope: ParamBlockMenuScope;
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
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.split",
        enabled: (ctx) => ctx.canSplit,
    },
    {
        key: "create-from-selection",
        labelKey: "param_blocks_create_from_selection",
        op: "createFromSelection",
        scope: "block",
        channel: "paramBlock",
        enabled: (ctx) => ctx.hasSelection,
    },
    {
        key: "group",
        labelKey: "param_blocks_group",
        op: "group",
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.group",
        separatorBefore: true,
        enabled: (ctx) => ctx.canGroup,
    },
    {
        key: "ungroup",
        labelKey: "param_blocks_ungroup",
        op: "ungroup",
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.ungroup",
        enabled: (ctx) => ctx.canUngroup,
    },
    {
        key: "restore-game",
        labelKey: "param_blocks_restore_game",
        op: "restoreGame",
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.restoreGame",
        separatorBefore: true,
        enabled: (ctx) => ctx.canRestoreGame,
    },
    {
        key: "restore-deleted",
        labelKey: "param_blocks_restore_deleted",
        op: "restoreDeleted",
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.restoreDeleted",
        enabled: (ctx) => ctx.canRestoreDeleted,
    },
    {
        key: "normalize-pitch",
        labelKey: "param_blocks_normalize_pitch",
        op: "normalizePitch",
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.normalizePitch",
        enabled: (ctx) => ctx.isPitch && ctx.canNormalizePitch,
    },
    {
        key: "copy",
        labelKey: "menu_copy",
        op: "copy",
        scope: "param",
        channel: "edit",
        shortcutAction: "clip.copy",
        separatorBefore: true,
        enabled: selectionAvailable,
    },
    {
        key: "cut",
        labelKey: "menu_cut",
        op: "cut",
        scope: "param",
        channel: "edit",
        shortcutAction: "clip.cut",
        enabled: selectionAvailable,
    },
    {
        key: "paste",
        labelKey: "menu_paste",
        op: "paste",
        scope: "param",
        channel: "edit",
        shortcutAction: "clip.paste",
        enabled: () => true,
    },
    {
        key: "initialize",
        labelKey: "menu_initialize",
        op: "initialize",
        scope: "param",
        channel: "edit",
        separatorBefore: true,
        enabled: selectionAvailable,
    },
    {
        key: "transpose-cents",
        labelKey: "menu_transpose_cents",
        op: "transposeCents",
        scope: "param",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "transpose-degrees",
        labelKey: "menu_transpose_degrees",
        op: "transposeDegrees",
        scope: "param",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "set-pitch",
        labelKey: "menu_set_pitch",
        op: "setPitch",
        scope: "param",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "average",
        labelKey: "menu_average",
        op: "average",
        scope: "param",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "smooth",
        labelKey: "menu_smooth",
        op: "smooth",
        scope: "param",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "add-vibrato",
        labelKey: "menu_add_vibrato",
        op: "addVibrato",
        scope: "param",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "quantize",
        labelKey: "menu_quantize",
        op: "quantize",
        scope: "param",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "mean-quantize",
        labelKey: "menu_mean_quantize",
        op: "meanQuantize",
        scope: "param",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "save-as-pitch-ref",
        labelKey: "menu_save_as_pitch_ref",
        op: "saveAsPitchRef",
        scope: "param",
        channel: "edit",
        enabled: (ctx) => ctx.isPitch && selectionAvailable(ctx),
    },
    {
        key: "export-midi",
        labelKey: "menu_export_midi",
        op: "exportMidi",
        scope: "param",
        channel: "edit",
        enabled: selectionAvailable,
    },
    {
        key: "convert-volume-to-dyn",
        labelKey: "menu_convert_volume_to_dyn",
        op: "convertVolumeToDyn",
        scope: "param",
        channel: "edit",
        separatorBefore: true,
        enabled: (ctx) => ctx.isVolume,
    },
    {
        key: "convert-dyn-to-volume",
        labelKey: "menu_convert_dyn_to_volume",
        op: "convertDynToVolume",
        scope: "param",
        channel: "edit",
        enabled: (ctx) => ctx.isDyn,
    },
    {
        key: "select-all",
        labelKey: "menu_select_all",
        op: "selectAll",
        scope: "selection",
        channel: "edit",
        shortcutAction: "edit.selectAll",
        separatorBefore: true,
        enabled: () => true,
    },
    {
        key: "deselect",
        labelKey: "menu_deselect",
        op: "deselect",
        scope: "selection",
        channel: "edit",
        shortcutAction: "edit.deselect",
        enabled: (ctx) => ctx.hasSelection,
    },
    {
        key: "delete",
        labelKey: "param_blocks_delete",
        op: "delete",
        scope: "block",
        channel: "paramBlock",
        shortcutAction: "paramBlock.delete",
        danger: true,
        separatorBefore: true,
        enabled: (ctx) => ctx.canDelete,
    },
];
