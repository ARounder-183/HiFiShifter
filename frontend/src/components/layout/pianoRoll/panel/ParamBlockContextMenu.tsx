/**
 * 参数块右键菜单。
 *
 * 菜单只承载块编辑的语义动作；打开位置、点击外部关闭与键盘导航交给
 * `AppContextMenu`，与参数编辑器已有的编辑菜单保持一致。
 *
 * 【为什么这里没有任何"为什么点不了"的解释】块动作的禁用条件是**瞬时状态**
 * （当前选了几个块、块是不是编组块、有没有可还原的区间）。它会被用户的
 * 下一次点击改变，而菜单弹出期间不可能再变一次 —— 写进第二行只会把菜单
 * 变成解释墙。`description` 留给"宿主能力"那一类用户无论如何都改不了的
 * 限制（见 `ui/Menu.tsx` 的字段注释）。
 */
import { AppContextMenu, type AppMenuItemSpec } from "../../../../ui";
import { useMenuShortcut } from "../../../../ui/useMenuShortcut";
import type { MessageKey } from "../../../../i18n/messages";

export interface ParamBlockContextMenuProps {
    readonly x: number;
    readonly y: number;
    readonly selectedCount: number;
    readonly canGroup: boolean;
    readonly canUngroup: boolean;
    readonly canDelete: boolean;
    readonly canRestoreGame?: boolean;
    readonly canRestoreDeleted?: boolean;
    readonly canNormalizePitch?: boolean;
    readonly splitAtSec: number | null;
    readonly onSplit: (sec: number) => void;
    readonly onGroup: () => void;
    readonly onUngroup: () => void;
    readonly onRestoreGame?: () => void;
    readonly onRestoreDeleted?: () => void;
    readonly onNormalizePitch?: () => void;
    readonly onDelete: () => void;
    readonly onClose: () => void;
    readonly tf: (key: MessageKey) => string;
}

export function ParamBlockContextMenu({
    x,
    y,
    selectedCount,
    canGroup,
    canUngroup,
    canDelete,
    canRestoreGame,
    canRestoreDeleted,
    canNormalizePitch,
    splitAtSec,
    onSplit,
    onGroup,
    onUngroup,
    onRestoreGame,
    onRestoreDeleted,
    onNormalizePitch,
    onDelete,
    onClose,
    tf,
}: ParamBlockContextMenuProps) {
    const splitShortcut = useMenuShortcut("paramBlock.split");
    const groupShortcut = useMenuShortcut("paramBlock.group");
    const ungroupShortcut = useMenuShortcut("paramBlock.ungroup");
    const restoreShortcut = useMenuShortcut("paramBlock.restoreGame");
    const restoreDeletedShortcut = useMenuShortcut("paramBlock.restoreDeleted");
    const normalizeShortcut = useMenuShortcut("paramBlock.normalizePitch");
    const deleteShortcut = useMenuShortcut("paramBlock.delete");

    /*
     * 三段：编辑 / 状态恢复 / 破坏性。删除永远单独成段并带 `danger`，
     * 避免和"恢复删除的块"这类可逆动作挨在一起被误点。
     */
    const items: AppMenuItemSpec[] = [
        {
            key: "blocks-split",
            label: tf("param_blocks_split_here"),
            shortcut: splitShortcut,
            disabled: splitAtSec === null || selectedCount !== 1,
            onSelect: () => {
                if (splitAtSec !== null) onSplit(splitAtSec);
            },
        },
        {
            key: "blocks-group",
            label: tf("param_blocks_group"),
            shortcut: groupShortcut,
            separatorBefore: true,
            disabled: !canGroup,
            onSelect: onGroup,
        },
        {
            key: "blocks-ungroup",
            label: tf("param_blocks_ungroup"),
            shortcut: ungroupShortcut,
            disabled: !canUngroup,
            onSelect: onUngroup,
        },
        {
            key: "blocks-restore-game",
            label: tf("param_blocks_restore_game"),
            shortcut: restoreShortcut,
            disabled: !canRestoreGame,
            onSelect: onRestoreGame ?? (() => {}),
        },
        {
            key: "blocks-restore-deleted",
            label: tf("param_blocks_restore_deleted"),
            shortcut: restoreDeletedShortcut,
            disabled: !canRestoreDeleted,
            onSelect: onRestoreDeleted ?? (() => {}),
        },
        {
            key: "blocks-normalize-pitch",
            label: tf("param_blocks_normalize_pitch"),
            shortcut: normalizeShortcut,
            disabled: !canNormalizePitch,
            onSelect: onNormalizePitch ?? (() => {}),
        },
        {
            key: "blocks-delete",
            label: tf("param_blocks_delete"),
            shortcut: deleteShortcut,
            danger: true,
            separatorBefore: true,
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
