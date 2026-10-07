// 轨道空白区右键菜单：插件保留菜单显示，未接宿主写口的动作不执行App私有编辑。
import React from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../../../i18n/I18nProvider";
import { AppContextMenu, useMenuShortcut } from "../../../ui";
import { canClipboardHostClips, canSplitHostClips, dawControlledReason, isPluginMode } from "../../../services/hostCapabilities";

export const TrackAreaContextMenu: React.FC<{
    x: number;
    y: number;
    canPaste: boolean;
    canSplit: boolean;
    /** 点击位置之后该轨道上是否还有 Clip（关闭间隙的启用条件）。 */
    canCloseGaps: boolean;
    onPaste: () => void;
    onSplit: () => void;
    onCloseGaps: () => void;
    onClose: () => void;
}> = ({ x, y, canPaste, canSplit, canCloseGaps, onPaste, onSplit, onCloseGaps, onClose }) => {
    const { t } = useI18n();
    const plugin = isPluginMode();
    // 显示菜单不等于放开全部几何操作：插件目前只接通这张菜单里的分割。
    const pasteAllowed = canPaste && (!plugin || canClipboardHostClips());
    const splitAllowed = canSplit && (!plugin || canSplitHostClips());
    const closeGapsAllowed = canCloseGaps && !plugin;
    // 快捷键提示：从快捷键注册表读取当前生效的绑定（随用户自定义实时变化）。
    // 时间轴的右键菜单此前只有 label/onSelect，把 `shortcut` 整条信息丢了 ——
    // 于是同一张时间轴上，剪辑菜单有快捷键、轨道区域菜单没有（见 ui/useMenuShortcut）。
    // 「关闭间隙」没有对应动作，因此不显示（空值时原语不渲染那一列）。
    const pasteShortcut = useMenuShortcut("clip.paste");
    const splitShortcut = useMenuShortcut("clip.split");

    return createPortal(
        <AppContextMenu
            x={x}
            y={y}
            onClose={onClose}
            // 时间轴浮动菜单契约：提示气泡抑制 / 内联编辑器失焦 / 角标编辑守卫
            // 都靠这个标记识别（见 src/ui/Menu.tsx 的 `floating` 说明）。
            floating
            items={[
                {
                    key: "paste",
                    label: t("menu_paste"),
                    shortcut: pasteShortcut,
                    disabled: !pasteAllowed,
                    tooltip: plugin && !canClipboardHostClips() ? dawControlledReason() : undefined,
                    onSelect: () => { if (pasteAllowed) onPaste(); },
                },
                {
                    key: "split",
                    label: t("ctx_split_at_playhead"),
                    shortcut: splitShortcut,
                    disabled: !splitAllowed,
                    tooltip: plugin && !canSplitHostClips() ? dawControlledReason() : undefined,
                    onSelect: () => { if (splitAllowed) onSplit(); },
                },
                {
                    key: "closeGaps",
                    label: t("ctx_close_gaps"),
                    disabled: !closeGapsAllowed,
                    tooltip: plugin ? dawControlledReason() : undefined,
                    onSelect: () => { if (closeGapsAllowed) onCloseGaps(); },
                },
            ]}
        />,
        document.body,
    );
};
