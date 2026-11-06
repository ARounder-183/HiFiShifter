/**
 * 参数块右键菜单：合并多选块 / 解合并当前块。
 *
 * 菜单只承载块编辑的语义动作；打开位置、点击外部关闭与键盘导航交给
 * `AppContextMenu`，与参数编辑器已有的编辑菜单保持一致。
 */
import { AppContextMenu, type AppMenuItemSpec } from "../../../../ui";
import type { MessageKey } from "../../../../i18n/messages";

export interface ParamBlockContextMenuProps {
    readonly x: number;
    readonly y: number;
    readonly selectedCount: number;
    readonly canMerge: boolean;
    readonly canUnmerge: boolean;
    readonly mergeMessage: string | null;
    readonly splitAtSec: number | null;
    readonly onSplit: (sec: number) => void;
    readonly onMerge: () => void;
    readonly onUnmerge: () => void;
    readonly onClose: () => void;
    readonly tf: (key: MessageKey) => string;
}

export function ParamBlockContextMenu({
    x,
    y,
    selectedCount,
    canMerge,
    canUnmerge,
    mergeMessage,
    splitAtSec,
    onSplit,
    onMerge,
    onUnmerge,
    onClose,
    tf,
}: ParamBlockContextMenuProps) {
    const items: AppMenuItemSpec[] = [
        {
            key: "blocks-heading",
            label: tf("param_blocks_tool"),
            heading: true,
        },
        {
            key: "blocks-split",
            label: tf("param_blocks_split_here"),
            disabled: splitAtSec === null || selectedCount !== 1,
            description: splitAtSec === null ? tf("param_blocks_split_none") : undefined,
            onSelect: () => {
                if (splitAtSec !== null) onSplit(splitAtSec);
            },
        },
        {
            key: "blocks-merge",
            label: tf("param_blocks_merge"),
            disabled: !canMerge,
            description:
                mergeMessage !== null
                    ? mergeMessage
                    : selectedCount < 2
                      ? tf("param_blocks_merge_need_two")
                      : undefined,
            onSelect: onMerge,
        },
        {
            key: "blocks-unmerge",
            label: tf("param_blocks_unmerge"),
            disabled: !canUnmerge,
            description: !canUnmerge ? tf("param_blocks_unmerge_none") : undefined,
            onSelect: onUnmerge,
        },
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
