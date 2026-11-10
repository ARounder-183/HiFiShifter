/**
 * 参数块右键菜单。
 *
 * 条目由 `paramBlockMenuModel` 唯一描述；本组件只负责把描述表渲染成
 * AppContextMenu，并根据当前瞬时上下文计算禁用态。块命令与参数命令走同一
 * 菜单，不再把块工具当成“另一个编辑器”。
 */
import { AppContextMenu, type AppMenuItemSpec } from "../../../../ui";
import { useMenuShortcut } from "../../../../ui/useMenuShortcut";
import type { MessageKey } from "../../../../i18n/messages";
import {
    PARAM_BLOCK_MENU_ITEMS,
    type ParamBlockMenuChannel,
    type ParamBlockMenuContext,
} from "../paramBlockMenuModel";

export interface ParamBlockContextMenuProps {
    readonly x: number;
    readonly y: number;
    readonly context: ParamBlockMenuContext;
    readonly onAction: (channel: ParamBlockMenuChannel, op: string) => void;
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
    const shortcuts = {
        "paramBlock.split": useMenuShortcut("paramBlock.split"),
        "paramBlock.group": useMenuShortcut("paramBlock.group"),
        "paramBlock.ungroup": useMenuShortcut("paramBlock.ungroup"),
        "paramBlock.restoreGame": useMenuShortcut("paramBlock.restoreGame"),
        "paramBlock.restoreDeleted": useMenuShortcut("paramBlock.restoreDeleted"),
        "paramBlock.normalizePitch": useMenuShortcut("paramBlock.normalizePitch"),
        "paramBlock.delete": useMenuShortcut("paramBlock.delete"),
        "pianoRoll.copy": useMenuShortcut("pianoRoll.copy"),
        "pianoRoll.cut": useMenuShortcut("pianoRoll.cut"),
        "pianoRoll.paste": useMenuShortcut("pianoRoll.paste"),
        "paramBlock.selectAll": useMenuShortcut("paramBlock.selectAll"),
        "paramBlock.deselect": useMenuShortcut("paramBlock.deselect"),
    } as const;
    const items: AppMenuItemSpec[] = PARAM_BLOCK_MENU_ITEMS.map((item) => {
        const action = item.shortcutAction as keyof typeof shortcuts | undefined;
        return {
            key: item.key,
            label: tf(item.labelKey),
            ...(action === undefined ? {} : { shortcut: shortcuts[action] }),
            ...(item.separatorBefore === true ? { separatorBefore: true } : {}),
            ...(item.danger === true ? { danger: true } : {}),
            disabled: !item.enabled(context),
            onSelect: () => onAction(item.channel, item.op),
        };
    });
    return (
        <AppContextMenu
            x={x}
            y={y}
            items={items}
            onClose={onClose}
            ariaLabel={tf("param_blocks_tool")}
        />
    );
}
