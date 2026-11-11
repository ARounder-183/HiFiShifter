/**
 * 参数块右键菜单。
 *
 * 条目由 `paramBlockMenuModel` 唯一描述；本组件只负责：
 * 1. 按 `visible` 过滤瞬时不可用命令；
 * 2. 把块 / 选区 / 剪贴板平铺，把变换 / 文件 IO 收进“更多参数操作”子菜单；
 * 3. 从 `shortcutAction` 读取真实快捷键文本，避免在描述表里伪造按键。
 */
import { AppContextMenu, type AppMenuItemSpec } from "../../../../ui";
import { useMenuShortcut } from "../../../../ui/useMenuShortcut";
import type { ActionId } from "../../../../features/keybindings/types";
import type { MessageKey } from "../../../../i18n/messages";
import {
    PARAM_BLOCK_MENU_ITEMS,
    type ParamBlockMenuChannel,
    type ParamBlockMenuContext,
    type ParamBlockMenuItem,
} from "../paramBlockMenuModel";

export interface ParamBlockContextMenuProps {
    readonly x: number;
    readonly y: number;
    readonly context: ParamBlockMenuContext;
    readonly onAction: (channel: ParamBlockMenuChannel, op: string) => void;
    readonly onClose: () => void;
    readonly tf: (key: MessageKey) => string;
}

const FLAT_GROUP_ORDER = ["block", "selection", "clipboard"] as const;

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
     * 记录表只覆盖本菜单真正用到的 ActionId；没有绑定（或绑定为 __none__）时
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
        "paramBlock.selectAll": useMenuShortcut("paramBlock.selectAll"),
        "paramBlock.deselect": useMenuShortcut("paramBlock.deselect"),
        "pianoRoll.copy": useMenuShortcut("pianoRoll.copy"),
        "pianoRoll.cut": useMenuShortcut("pianoRoll.cut"),
        "pianoRoll.paste": useMenuShortcut("pianoRoll.paste"),
        "edit.initialize": useMenuShortcut("edit.initialize"),
        "edit.transposeCents": useMenuShortcut("edit.transposeCents"),
        "edit.transposeDegrees": useMenuShortcut("edit.transposeDegrees"),
        "edit.setPitch": useMenuShortcut("edit.setPitch"),
        "edit.average": useMenuShortcut("edit.average"),
        "edit.smooth": useMenuShortcut("edit.smooth"),
        "edit.addVibrato": useMenuShortcut("edit.addVibrato"),
        "edit.quantize": useMenuShortcut("edit.quantize"),
        "edit.meanQuantize": useMenuShortcut("edit.meanQuantize"),
    };

    const toSpec = (item: ParamBlockMenuItem, firstInGroup = false): AppMenuItemSpec => {
        const shortcut =
            item.shortcutAction === undefined ? "" : (shortcuts[item.shortcutAction] ?? "");
        const labelKey =
            typeof item.labelKey === "function" ? item.labelKey(context) : item.labelKey;
        return {
            key: item.key,
            label: tf(labelKey),
            ...(shortcut.length > 0 ? { shortcut } : {}),
            ...(item.separatorBefore === true && !firstInGroup ? { separatorBefore: true } : {}),
            ...(item.danger === true ? { danger: true } : {}),
            disabled: !item.enabled(context),
            onSelect: () => onAction(item.channel, item.op),
        };
    };

    const visibleItems = PARAM_BLOCK_MENU_ITEMS.filter((item) => item.visible?.(context) ?? true);
    const flatItems = FLAT_GROUP_ORDER.flatMap((group) =>
        visibleItems.filter((item) => item.group === group && item.key !== "delete"),
    ).map((item, index) => toSpec(item, index === 0));
    const moreItems = visibleItems
        .filter((item) => item.group === "transform" || item.group === "io")
        .map((item, index) => toSpec(item, index === 0));
    const destructiveItems = visibleItems
        .filter((item) => item.key === "delete")
        .map((item) => toSpec(item));

    const items: AppMenuItemSpec[] = [
        ...flatItems,
        ...(moreItems.length > 0
            ? [
                  {
                      key: "more",
                      label: tf("param_blocks_menu_more"),
                      items: moreItems,
                  } satisfies AppMenuItemSpec,
              ]
            : []),
        ...destructiveItems,
    ];

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
