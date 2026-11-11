/**
 * 参数块右键菜单：**一个块命令子菜单 + 标准参数编辑菜单**。
 *
 * 块命令由 `paramBlockCommandMenu` 描述（只有 8 项）；复制、剪切、粘贴、全选、
 * 初始化、移调、平滑、颤音、量化、导出等标准命令全部交给 `EditContextMenu`
 * 的唯一描述表。本组件只负责：
 * 1. 按 `visible` 过滤当前时刻不可用的块命令；
 * 2. 从 `shortcutAction` 读取真实快捷键文本；
 * 3. 把块项收成一个顶部“参数块 ▸”子菜单，标准项仍走 `edit` 动作通道。
 */
import type { AppMenuItemSpec } from "../../../../ui";
import { useMenuShortcut } from "../../../../ui/useMenuShortcut";
import type { ActionId } from "../../../../features/keybindings/types";
import type { MessageKey } from "../../../../i18n/messages";
import { EditContextMenu } from "../../../editDialogs/EditContextMenu";
import {
    PARAM_BLOCK_COMMANDS,
    type ParamBlockCommandContext,
    type ParamBlockCommandItem,
} from "../paramBlockCommandMenu";

export type ParamBlockMenuActionChannel = "paramBlock" | "edit";

export interface ParamBlockContextMenuProps {
    readonly x: number;
    readonly y: number;
    readonly context: ParamBlockCommandContext & {
        readonly isVolume: boolean;
        readonly isDyn: boolean;
    };
    readonly onAction: (channel: ParamBlockMenuActionChannel, op: string) => void;
    readonly onClose: () => void;
    readonly tf: (key: MessageKey) => string;
}

export function ParamBlockContextMenu({
    x,
    y,
    context,
    onAction,
    onClose,
    tf,
}: ParamBlockContextMenuProps) {
    /*
     * 所有 useMenuShortcut 必须无条件调用：菜单项是否可见不能改变 hooks 顺序。
     * 只登记块命令真正用到的 ActionId；没有绑定（或绑定为 __none__）时
     * useMenuShortcut 返回空串，渲染层会省略快捷键列。
     */
    const shortcuts: Partial<Record<ActionId, string>> = {
        "paramBlock.split": useMenuShortcut("paramBlock.split"),
        "paramBlock.group": useMenuShortcut("paramBlock.group"),
        "paramBlock.ungroup": useMenuShortcut("paramBlock.ungroup"),
        "paramBlock.restoreGame": useMenuShortcut("paramBlock.restoreGame"),
        "paramBlock.restoreDeleted": useMenuShortcut("paramBlock.restoreDeleted"),
        "paramBlock.normalizePitch": useMenuShortcut("paramBlock.normalizePitch"),
        "paramBlock.delete": useMenuShortcut("paramBlock.delete"),
    };

    const toSpec = (item: ParamBlockCommandItem): AppMenuItemSpec => {
        const shortcut =
            item.shortcutAction === undefined ? "" : (shortcuts[item.shortcutAction] ?? "");
        return {
            key: item.key,
            label: tf(item.labelKey),
            ...(shortcut.length > 0 ? { shortcut } : {}),
            ...(item.separatorBefore === true ? { separatorBefore: true } : {}),
            ...(item.danger === true ? { danger: true } : {}),
            disabled: !item.enabled(context),
            onSelect: () => onAction("paramBlock", item.op),
        };
    };

    const blockItems = PARAM_BLOCK_COMMANDS.filter((item) => item.visible?.(context) ?? true).map(
        toSpec,
    );
    const blockMenu: AppMenuItemSpec = {
        key: "paramBlock",
        label: tf("param_blocks_submenu"),
        items: blockItems,
    };

    return (
        <EditContextMenu
            x={x}
            y={y}
            leadingItems={[blockMenu]}
            isPitchParam={context.isPitch}
            isVolumeParam={context.isVolume}
            isDynParam={context.isDyn}
            ariaLabel={tf("param_blocks_menu")}
            onClose={onClose}
            onCopy={() => onAction("edit", "copy")}
            onCut={() => onAction("edit", "cut")}
            onPaste={() => onAction("edit", "paste")}
            onSelectAll={() => onAction("edit", "selectAll")}
            onDeselect={() => onAction("edit", "deselect")}
            onInitialize={() => onAction("edit", "initialize")}
            onTransposeCents={() => onAction("edit", "transposeCents")}
            onTransposeDegrees={() => onAction("edit", "transposeDegrees")}
            onSetPitch={() => onAction("edit", "setPitch")}
            onAverage={() => onAction("edit", "average")}
            onSmooth={() => onAction("edit", "smooth")}
            onAddVibrato={() => onAction("edit", "addVibrato")}
            onQuantize={() => onAction("edit", "quantize")}
            onMeanQuantize={() => onAction("edit", "meanQuantize")}
            onSaveAsPitchRef={() => onAction("edit", "saveAsPitchRef")}
            onExportMidi={() => onAction("edit", "exportMidi")}
            onConvertVolumeToDyn={() => onAction("edit", "convertVolumeToDyn")}
            onConvertDynToVolume={() => onAction("edit", "convertDynToVolume")}
        />
    );
}
