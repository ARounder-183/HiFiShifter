/**
 * 参数块右键菜单。
 *
 * 菜单只承载块编辑的语义动作；打开位置、点击外部关闭与键盘导航交给
 * \`AppContextMenu\`，与参数编辑器已有的编辑菜单保持一致。
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
    readonly splitDisabledReason?: string;
    readonly groupDisabledReason?: string;
    readonly ungroupDisabledReason?: string;
    readonly restoreDisabledReason?: string;
    readonly restoreDeletedDisabledReason?: string;
    readonly normalizeDisabledReason?: string;
    readonly deleteDisabledReason?: string;
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
    splitDisabledReason,
    groupDisabledReason,
    ungroupDisabledReason,
    restoreDisabledReason,
    restoreDeletedDisabledReason,
    normalizeDisabledReason,
    deleteDisabledReason,
}: ParamBlockContextMenuProps) {
    const splitShortcut = useMenuShortcut("paramBlock.split");
    const groupShortcut = useMenuShortcut("paramBlock.group");
    const ungroupShortcut = useMenuShortcut("paramBlock.ungroup");
    const restoreShortcut = useMenuShortcut("paramBlock.restoreGame");
    const restoreDeletedShortcut = useMenuShortcut("paramBlock.restoreDeleted");
    const normalizeShortcut = useMenuShortcut("paramBlock.normalizePitch");
    const deleteShortcut = useMenuShortcut("paramBlock.delete");

    const items: AppMenuItemSpec[] = [
        {
            key: "blocks-split",
            label: tf("param_blocks_split_here"),
            shortcut: splitShortcut,
            disabled: splitAtSec === null || selectedCount !== 1,
            description: splitDisabledReason,
            onSelect: () => {
                if (splitAtSec !== null) onSplit(splitAtSec);
            },
        },
        {
            key: "blocks-group",
            label: tf("param_blocks_group"),
            shortcut: groupShortcut,
            disabled: !canGroup,
            description: groupDisabledReason,
            onSelect: onGroup,
        },
        {
            key: "blocks-ungroup",
            label: tf("param_blocks_ungroup"),
            shortcut: ungroupShortcut,
            disabled: !canUngroup,
            description: ungroupDisabledReason,
            onSelect: onUngroup,
        },
        {
            key: "blocks-restore-game",
            label: tf("param_blocks_restore_game"),
            shortcut: restoreShortcut,
            disabled: !canRestoreGame,
            description: restoreDisabledReason,
            onSelect: onRestoreGame ?? (() => {}),
        },
        {
            key: "blocks-restore-deleted",
            label: tf("param_blocks_restore_deleted"),
            shortcut: restoreDeletedShortcut,
            disabled: !canRestoreDeleted,
            description: restoreDeletedDisabledReason,
            onSelect: onRestoreDeleted ?? (() => {}),
        },
        {
            key: "blocks-normalize-pitch",
            label: tf("param_blocks_normalize_pitch"),
            shortcut: normalizeShortcut,
            disabled: !canNormalizePitch,
            description: normalizeDisabledReason,
            onSelect: onNormalizePitch ?? (() => {}),
        },
        {
            key: "blocks-delete",
            label: tf("param_blocks_delete"),
            shortcut: deleteShortcut,
            danger: true,
            disabled: !canDelete,
            description: deleteDisabledReason,
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
