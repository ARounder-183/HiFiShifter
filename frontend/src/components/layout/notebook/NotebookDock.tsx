/*
 * 记事本在右侧栏里的停靠壳：负责宽度拖拽与持久化。
 *
 * 宽度归"布局"管，所以放在这里而不是面板内部 —— 面板只管内容，不关心自己
 * 占了多宽。默认 360px：富文本编辑 + 图片在 320px 下太挤，而 360px 仍然
 * 给时间轴留足空间。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { patchNotebookSettings } from "../../../features/notebook/notebookSlice";
import { settingsApi } from "../../../services/api/settings";
import { NotebookPanel } from "./NotebookPanel";
import {
    NOTEBOOK_PANEL_MAX_WIDTH,
    NOTEBOOK_PANEL_MIN_WIDTH,
} from "./notebookSettings";

export function NotebookDock() {
    const dispatch = useAppDispatch();
    const width = useAppSelector((state) => state.notebook.settings.panelWidth);
    const [dragging, setDragging] = useState(false);
    const widthRef = useRef(width);
    const containerRef = useRef<HTMLDivElement | null>(null);

    useEffect(() => {
        widthRef.current = width;
    }, [width]);

    const onPointerDown = useCallback((event: React.PointerEvent<HTMLDivElement>) => {
        event.preventDefault();
        setDragging(true);
        const startX = event.clientX;
        const startWidth = widthRef.current;

        const onMove = (moveEvent: PointerEvent) => {
            // 面板贴右边缘，向左拖变宽。
            const next = Math.min(
                NOTEBOOK_PANEL_MAX_WIDTH,
                Math.max(NOTEBOOK_PANEL_MIN_WIDTH, startWidth + (startX - moveEvent.clientX)),
            );
            widthRef.current = next;
            // 拖动过程只改 DOM，松手才进 React/落盘（与时间轴分隔条同一策略）。
            const element = containerRef.current;
            if (element) element.style.width = `${next}px`;
        };
        const onUp = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onUp);
            window.removeEventListener("pointercancel", onUp);
            setDragging(false);
            dispatch(patchNotebookSettings({ panelWidth: widthRef.current }));
            void settingsApi
                .saveUiSettings({ notebook: { panelWidth: widthRef.current } })
                .catch(() => {});
        };
        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onUp);
        window.addEventListener("pointercancel", onUp);
    }, [dispatch]);

    return (
        <div className="flex min-h-0 shrink-0">
            <div
                className="hs-notebook-resizer"
                data-dragging={dragging ? "true" : "false"}
                onPointerDown={onPointerDown}
                title=""
            />
            <div
                ref={containerRef}
                className="flex min-h-0 flex-col bg-qt-window"
                style={{ width }}
            >
                <NotebookPanel />
            </div>
        </div>
    );
}
