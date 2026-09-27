import React from "react";
import { createPortal } from "react-dom";
import { useI18n } from "../../../i18n/I18nProvider";
import { AppContextMenu } from "../../../ui/Menu";

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
                    disabled: !canPaste,
                    onSelect: onPaste,
                },
                {
                    key: "split",
                    label: t("ctx_split_at_playhead"),
                    disabled: !canSplit,
                    onSelect: onSplit,
                },
                {
                    key: "closeGaps",
                    label: t("ctx_close_gaps"),
                    disabled: !canCloseGaps,
                    onSelect: onCloseGaps,
                },
            ]}
        />,
        document.body,
    );
};
