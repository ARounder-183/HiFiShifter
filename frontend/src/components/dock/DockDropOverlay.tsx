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
import { DOCK_SPLITTER_PX, type DockDropZone } from "../../features/dock/dockTypes";
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

/** 落点方向的可读名称（让提示从"停靠到此处"变成"停靠到左侧"）。 */
function describeZone(zone: DockDropZone, tAny: (key: string) => string): string {
    switch (zone) {
        case "left":
            return tAny("dock_side_left");
        case "right":
            return tAny("dock_side_right");
        case "top":
            return tAny("dock_side_top");
        case "bottom":
            return tAny("dock_side_bottom");
        default:
            return tAny("dock_side_center");
    }
}

function DockDropOverlayContent({ drag }: { drag: DockDragState }) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const showPreview = useAppSelector((s) => s.dock.settings.showDropPreview);
    const definition = getPanel(drag.panelId);
    const title = definition ? tAny(definition.titleKey) : drag.panelId;

    // 停靠预览：按住修饰键且命中某个 Zone 时，画出"新组会占哪半边"。
    const dockPreview =
        showPreview && drag.dockIntent && drag.target
            ? dropPreviewRect(drag.target.rect, drag.target.zone, DOCK_SPLITTER_PX)
            : null;

    // 浮动预览：没有停靠落点时，画出"浮窗会落在哪、多大"的虚线轮廓。
    // 【为什么必须有】不按修饰键拖拽的语义就是浮动，而浮动同样是一个用户需要
    // 预判的结果 —— 只给一个小标签跟着鼠标，用户无从知道松手后窗体会多大、
    // 会不会盖住他要看的东西。轮廓用**记住的浮窗尺寸**，与松手后的结果一致。
    const floatPreview = showPreview && !dockPreview ? drag.floatRect : null;

    return (
        <div className="hs-dock-overlay">
            {floatPreview ? (
                <div
                    className="hs-dock-float-preview"
                    style={{
                        left: floatPreview.x,
                        top: floatPreview.y,
                        width: floatPreview.w,
                        height: floatPreview.h,
                    }}
                >
                    <span className="hs-dock-float-preview-title">{title}</span>
                </div>
            ) : null}
            {dockPreview ? (
                <div
                    className="hs-dock-drop-preview"
                    style={{
                        left: dockPreview.x,
                        top: dockPreview.y,
                        width: dockPreview.w,
                        height: dockPreview.h,
                    }}
                />
            ) : null}
            {/*
              拖拽提示块复用自定义 tooltip 的外观（`app-tooltip` 类）—— 它是拖拽
              期间用户唯一能看到的落点说明，样式必须与其它悬停提示同源，否则会像
              另一套 UI。`hs-dock-ghost` 只补拖拽特有的部分（意图着色、跟随指针）。
            */}
            <div
                className="app-tooltip hs-dock-ghost"
                data-intent={drag.dockIntent ? "dock" : "float"}
                style={{ left: drag.pointerX + 14, top: drag.pointerY + 14 }}
            >
                {title}
                <span className="hs-dock-ghost-hint">
                    {drag.dockIntent
                        ? drag.target
                            ? `${tAny("dock_hint_dock")} · ${describeZone(drag.target.zone, tAny)}`
                            : tAny("dock_hint_snapback")
                        : tAny("dock_hint_float")}
                </span>
            </div>
        </div>
    );
}
