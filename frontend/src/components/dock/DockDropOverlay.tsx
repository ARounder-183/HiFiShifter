/*
 * 拖拽落点覆盖层。
 *
 * 只订阅 `dockDragStore`（一个极小的外部 store），因此只有这一层跟着拖拽重渲染
 * —— 不会把 33Hz 的播放轮询订阅者、菜单栏、时间轴一起拖下水。控制器（见
 * `dockDragController`）已把指针事件合并到每帧一次，这里每次重渲染都对应一帧。
 *
 * 预览矩形由 `dropPreviewRect` 算出，与真正提交时的落点判定同源（同一个
 * `pickDropTarget` + `resolveDropZone`），所以"看到的"与"松手得到的"必然一致。
 */

import { useEffect, useLayoutEffect, useRef, useState, useSyncExternalStore } from "react";

import { clampAxisPosition, EDGE_GAP } from "../appTooltipPosition";
import { useAppSelector } from "../../app/hooks";
import {
    getDockDragState,
    subscribeDockDrag,
    type DockDragState,
} from "../../features/dock/dockDragStore";
import { dropPreviewRect } from "../../features/dock/dockDropTarget";
import { DOCK_SPLITTER_PX, type DockDropZone } from "../../features/dock/dockTypes";
import { getPanel } from "../../features/dock/panelRegistry";
import { dockModifierHint } from "./dockTooltips";
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

/** 幽灵相对指针的偏移（右下角跟随）：比 tooltip（14/18）更贴近指针。 */
const GHOST_OFFSET_PX = 14;

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
    const dockModifier = useAppSelector((s) => s.dock.settings.dockModifier);
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

    // 拖拽幽灵的右缘钳制需要实测宽度：提示块 `max-width: 320px` 且不折行，宽度
    // 由内容决定，预留固定值会在窄内容时把幽灵整段甩离指针（与 tooltip 的修复
    // 同一套结论，见 `clampAxisPosition`）。每帧量一次 —— 内容不变时 setState
    // 值相同，React 直接跳过，不会造成额外渲染循环。
    const ghostRef = useRef<HTMLDivElement | null>(null);
    const [ghostWidth, setGhostWidth] = useState(0);
    // 依赖随拖拽内容（标题/面板名）变化而变化，故意不列依赖：每次渲染后都量一次，
    // 值不变时 setState 直接跳过（见上方注释）。
    // eslint-disable-next-line react-hooks/exhaustive-deps
    useLayoutEffect(() => {
        const element = ghostRef.current;
        if (!element) return;
        const width = element.getBoundingClientRect().width;
        setGhostWidth((previous) => (previous === width ? previous : width));
    });

    // 幽灵跟随用 transform 而不是 left/top：前者只走合成，不触发重新布局 ——
    // 拖拽期间每帧都要挪动它，这是白拿的帧预算。横坐标经 `clampAxisPosition`
    // 钳制（指针贴窗口右缘时收回必要距离，指针坐标为负时也不画出左缘外）；
    // 垂直方向与修复前一致（+14），不需要钳制。
    const ghostX = clampAxisPosition(
        drag.pointerX,
        ghostWidth,
        window.innerWidth,
        GHOST_OFFSET_PX,
        EDGE_GAP,
    );

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
                ref={ghostRef}
                className="app-tooltip hs-dock-ghost"
                data-intent={drag.dockIntent ? "dock" : "float"}
                style={{
                    left: 0,
                    top: 0,
                    transform: `translate(${ghostX}px, ${drag.pointerY + GHOST_OFFSET_PX}px)`,
                }}
            >
                <div className="hs-dock-ghost-line">
                    {title}
                    <span className="hs-dock-ghost-hint">
                        {drag.dockIntent
                            ? drag.target
                                ? `${tAny("dock_hint_dock")} · ${describeZone(drag.target.zone, tAny)}`
                                : tAny("dock_hint_snapback")
                            : tAny("dock_hint_float")}
                    </span>
                </div>
                {/*
                  第二行**永远**给出"按住 {修饰键} 可停靠"：拖拽期间用户看不到抓手上的
                  悬停提示，这里是唯一能告诉他"怎么才能停靠"的地方（用户明确要求）。
                  与抓手提示共用 `dockModifierHint`，两处不会分叉。
                */}
                <div className="hs-dock-ghost-modifier">{dockModifierHint(dockModifier, tAny)}</div>
            </div>
        </div>
    );
}
