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
    readonly canDelete: boolean;
    readonly canRestoreGame?: boolean;
    readonly onRestoreGame?: () => void;
    readonly splitAtSec: number | null;
    readonly selectedDeleted: boolean;
    readonly onSplit: (sec: number) => void;
    readonly onMerge: () => void;
    readonly onDelete: () => void;
    readonly onClose: () => void;
    readonly tf: (key: MessageKey) => string;
}

export function ParamBlockContextMenu({
    x,
    y,
    selectedCount,
    canMerge,
    canDelete,
    canRestoreGame,
    onRestoreGame,
    splitAtSec,
    selectedDeleted,
    onSplit,
    onMerge,
    onDelete,
    onClose,
    tf,
}: ParamBlockContextMenuProps) {
    const items: AppMenuItemSpec[] = [
        {
            key: "blocks-split",
            label: tf("param_blocks_split_here"),
            disabled: splitAtSec === null || selectedCount !== 1 || selectedDeleted,
            onSelect: () => {
                if (splitAtSec !== null) onSplit(splitAtSec);
            },
        },
        {
            key: "blocks-merge",
            label: tf("param_blocks_merge"),
            disabled: !canMerge || selectedDeleted,
            onSelect: onMerge,
        },
        {
            key: "blocks-restore-game",
            label: tf("param_blocks_restore_game"),
            disabled: !canRestoreGame,
            onSelect: onRestoreGame ?? (() => {}),
        },
        {
            key: "blocks-delete",
            label: tf("param_blocks_delete"),
            disabled: !canDelete,
            onSelect: onDelete,
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
