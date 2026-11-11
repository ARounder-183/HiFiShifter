/**
 * 参数块命令面的唯一描述表。
 *
 * 参数块右键菜单从这里生成条目；差异只在 `ParamBlockMenuContext` 的瞬时
 * 可用态，不在文案或动作名上。
 *
 * 【菜单布局】块命令 / 选区 / 剪贴板保持平铺（右键后直接可点），参数变换与
 * 文件 IO 收进“更多参数操作”子菜单。这样典型块选择下平铺项不超过 10 项，
 * 同时每个有真实快捷键绑定的命令都补上 `shortcutAction`，由菜单组件统一
 * 通过 `useMenuShortcut` 显示真实按键，不在模型里伪造文本。
 */
import type { ActionId } from "../../../features/keybindings/types";
import type { MessageKey } from "../../../i18n/messages";

export type ParamBlockMenuChannel = "paramBlock" | "edit";

/**
 * 菜单分组：
 * - block / selection / clipboard：平铺；
 * - transform / io：收进子菜单；
 * - delete 虽属 block，但渲染时固定置于最后。
 */
export type ParamBlockMenuGroup = "block" | "selection" | "clipboard" | "transform" | "io";

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
    /** 固定文案，或按瞬时上下文选择文案（如 pitch / value 共用的“设置到…”）。 */
    readonly labelKey: MessageKey | ((ctx: ParamBlockMenuContext) => MessageKey);
    readonly op: string;
    readonly channel: ParamBlockMenuChannel;
    readonly group: ParamBlockMenuGroup;
    readonly shortcutAction?: ActionId;
    readonly danger?: boolean;
    readonly separatorBefore?: boolean;
    /** 仅控制“是否出现在菜单里”；可用性由 `enabled` 表达。 */
    readonly visible?: (ctx: ParamBlockMenuContext) => boolean;
    readonly enabled: (ctx: ParamBlockMenuContext) => boolean;
}

const selectionAvailable = (ctx: ParamBlockMenuContext) => ctx.hasSelection;
const isPitchSelection = (ctx: ParamBlockMenuContext) => ctx.isPitch && selectionAvailable(ctx);

export const PARAM_BLOCK_MENU_ITEMS: readonly ParamBlockMenuItem[] = [
    // ── 块命令（平铺）──────────────────────────────────────────────────────
    {
        key: "split",
        labelKey: "param_blocks_split_at_playhead",
        op: "split",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.split",
        visible: (ctx) => ctx.canSplit(),
        enabled: (ctx) => ctx.canSplit(),
    },
    {
        key: "create-from-selection",
        labelKey: "param_blocks_create_from_selection",
        op: "createFromSelection",
        channel: "paramBlock",
        group: "block",
        visible: (ctx) => ctx.hasSelection,
        enabled: (ctx) => ctx.hasSelection,
    },
    {
        key: "group",
        labelKey: "param_blocks_group",
        op: "group",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.group",
        separatorBefore: true,
        visible: (ctx) => ctx.canGroup,
        enabled: (ctx) => ctx.canGroup,
    },
    {
        key: "ungroup",
        labelKey: "param_blocks_ungroup",
        op: "ungroup",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.ungroup",
        visible: (ctx) => ctx.canUngroup,
        enabled: (ctx) => ctx.canUngroup,
    },
    {
        key: "restore-game",
        labelKey: "param_blocks_restore_game",
        op: "restoreGame",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.restoreGame",
        separatorBefore: true,
        visible: (ctx) => ctx.canRestoreGame,
        enabled: (ctx) => ctx.canRestoreGame,
    },
    {
        key: "restore-deleted",
        labelKey: "param_blocks_restore_deleted",
        op: "restoreDeleted",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.restoreDeleted",
        visible: (ctx) => ctx.canRestoreDeleted,
        enabled: (ctx) => ctx.canRestoreDeleted,
    },
    {
        key: "normalize-pitch",
        labelKey: "param_blocks_normalize_pitch",
        op: "normalizePitch",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.normalizePitch",
        visible: (ctx) => ctx.isPitch && ctx.canNormalizePitch,
        enabled: (ctx) => ctx.isPitch && ctx.canNormalizePitch,
    },
    // ── 选区（平铺）────────────────────────────────────────────────────────
    {
        key: "select-all",
        labelKey: "menu_select_all",
        op: "selectAll",
        channel: "edit",
        group: "selection",
        shortcutAction: "paramBlock.selectAll",
        separatorBefore: true,
        enabled: () => true,
    },
    {
        key: "deselect",
        labelKey: "menu_deselect",
        op: "deselect",
        channel: "edit",
        group: "selection",
        shortcutAction: "paramBlock.deselect",
        visible: (ctx) => ctx.hasSelection,
        enabled: (ctx) => ctx.hasSelection,
    },
    // ── 剪贴板（平铺）──────────────────────────────────────────────────────
    {
        key: "copy",
        labelKey: "menu_copy",
        op: "copy",
        channel: "edit",
        group: "clipboard",
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
        group: "clipboard",
        shortcutAction: "pianoRoll.cut",
        enabled: () => true,
    },
    {
        key: "paste",
        labelKey: "menu_paste",
        op: "paste",
        channel: "edit",
        group: "clipboard",
        shortcutAction: "pianoRoll.paste",
        enabled: () => true,
    },
    // ── 参数变换（子菜单）──────────────────────────────────────────────────
    {
        key: "initialize",
        labelKey: "menu_initialize",
        op: "initialize",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.initialize",
        enabled: () => true,
    },
    {
        key: "transpose-cents",
        labelKey: "menu_transpose_cents",
        op: "transposeCents",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.transposeCents",
        visible: (ctx) => ctx.isPitch,
        enabled: isPitchSelection,
    },
    {
        key: "transpose-degrees",
        labelKey: "menu_transpose_degrees",
        op: "transposeDegrees",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.transposeDegrees",
        visible: (ctx) => ctx.isPitch,
        enabled: isPitchSelection,
    },
    {
        key: "set-pitch",
        labelKey: (ctx) => (ctx.isPitch ? "menu_set_pitch" : "menu_set_value"),
        op: "setPitch",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.setPitch",
        enabled: selectionAvailable,
    },
    {
        key: "average",
        labelKey: "menu_average",
        op: "average",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.average",
        enabled: selectionAvailable,
    },
    {
        key: "smooth",
        labelKey: "menu_smooth",
        op: "smooth",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.smooth",
        enabled: selectionAvailable,
    },
    {
        key: "add-vibrato",
        labelKey: "menu_add_vibrato",
        op: "addVibrato",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.addVibrato",
        visible: (ctx) => ctx.isPitch,
        enabled: isPitchSelection,
    },
    {
        key: "quantize",
        labelKey: "menu_quantize",
        op: "quantize",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.quantize",
        enabled: selectionAvailable,
    },
    {
        key: "mean-quantize",
        labelKey: "menu_mean_quantize",
        op: "meanQuantize",
        channel: "edit",
        group: "transform",
        shortcutAction: "edit.meanQuantize",
        enabled: selectionAvailable,
    },
    {
        key: "convert-volume-to-dyn",
        labelKey: "menu_convert_volume_to_dyn",
        op: "convertVolumeToDyn",
        channel: "edit",
        group: "transform",
        separatorBefore: true,
        visible: (ctx) => ctx.isVolume,
        enabled: (ctx) => ctx.isVolume,
    },
    {
        key: "convert-dyn-to-volume",
        labelKey: "menu_convert_dyn_to_volume",
        op: "convertDynToVolume",
        channel: "edit",
        group: "transform",
        visible: (ctx) => ctx.isDyn,
        enabled: (ctx) => ctx.isDyn,
    },
    // ── 文件 IO（子菜单）───────────────────────────────────────────────────
    {
        key: "save-as-pitch-ref",
        labelKey: "menu_save_as_pitch_ref",
        op: "saveAsPitchRef",
        channel: "edit",
        group: "io",
        separatorBefore: true,
        visible: (ctx) => ctx.isPitch,
        enabled: isPitchSelection,
    },
    {
        key: "export-midi",
        labelKey: "menu_export_midi",
        op: "exportMidi",
        channel: "edit",
        group: "io",
        visible: (ctx) => ctx.isPitch,
        enabled: selectionAvailable,
    },
    // ── 破坏性操作固定最后 ─────────────────────────────────────────────────
    {
        key: "delete",
        labelKey: "param_blocks_delete",
        op: "delete",
        channel: "paramBlock",
        group: "block",
        shortcutAction: "paramBlock.delete",
        danger: true,
        separatorBefore: true,
        visible: (ctx) => ctx.canDelete,
        enabled: (ctx) => ctx.canDelete,
    },
];
