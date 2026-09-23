/*
 * 拖拽落点覆盖层。
 *
 * 只订阅 `dockDragStore`（一个极小的外部 store），因此指针每移动一次只有这
 * 一层重渲染 —— 不会把 33Hz 的播放轮询订阅者、菜单栏、时间轴一起拖下水。
 *
 * 预览矩形由 `dropPreviewRect` 算出，与真正提交时的落点判定同源（同一个
 * `pickDropTarget` + `resolveDropZone`），所以"看到的"与"松手得到的"必然一致。
 */

import { useEffect, useSyncExternalStore } from "react";

import { useAppSelector } from "../../app/hooks";
import {
    getDockDragState,
    subscribeDockDrag,
    type DockDragState,
} from "../../features/dock/dockDragStore";
import { dropPreviewRect } from "../../features/dock/dockDropTarget";
import { DOCK_SPLITTER_PX } from "../../features/dock/dockTypes";
import { getPanel } from "../../features/dock/panelRegistry";
import { useI18n } from "../../i18n/I18nProvider";

export function DockDropOverlay() {
    const drag = useSyncExternalStore(subscribeDockDrag, getDockDragState, getDockDragState);
    const started = drag?.started === true;

    // 拖拽期间给 body 打标：禁掉全局文本选择、并让内嵌 iframe 不吃指针事件
    // （否则指针划过 iframe 时 pointermove 会中断，拖拽"卡住"）。
    useEffect(() => {
        if (!started) return;
        document.body.dataset.dockDragging = "true";
        return () => {
            delete document.body.dataset.dockDragging;
        };
    }, [started]);

    if (!drag?.started) return null;
    return <DockDropOverlayContent drag={drag} />;
}

function DockDropOverlayContent({ drag }: { drag: DockDragState }) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const showPreview = useAppSelector((s) => s.dock.settings.showDropPreview);
    const definition = getPanel(drag.panelId);
    const title = definition ? tAny(definition.titleKey) : drag.panelId;

    const preview =
        showPreview && drag.dockIntent && drag.target
            ? dropPreviewRect(drag.target.rect, drag.target.zone, DOCK_SPLITTER_PX)
            : null;

    return (
        <div className="hs-dock-overlay">
            {preview ? (
                <div
                    className="hs-dock-drop-preview"
                    style={{
                        left: preview.x,
                        top: preview.y,
                        width: preview.w,
                        height: preview.h,
                    }}
                />
            ) : null}
            <div
                className="hs-dock-ghost"
                data-intent={drag.dockIntent ? "dock" : "float"}
                style={{ left: drag.pointerX + 14, top: drag.pointerY + 14 }}
            >
                {title}
                <span style={{ opacity: 0.7 }}>
                    {drag.dockIntent
                        ? drag.target
                            ? tAny("dock_hint_dock")
                            : tAny("dock_hint_snapback")
                        : tAny("dock_hint_float")}
                </span>
            </div>
        </div>
    );
}
