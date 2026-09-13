import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Cross2Icon, EnterIcon } from "@radix-ui/react-icons";
import { shallowEqual } from "react-redux";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import {
    persistUiSettings,
    setHistoryPositionRemote,
    setProjectSaveUndoHistoryRemote,
    setSaveUndoHistoryByDefault,
} from "../../features/session/sessionSlice";

/**
 * 「操作记录」窗口（REAPER Undo History 风格）。
 *
 * 【非模态】不使用 Dialog：没有遮罩、不使背景变暗、不拦截事件 —— 窗口打开
 * 期间用户照常编辑时间轴，条目随 `history_state` 事件实时刷新。面板本体只
 * 占据自身矩形，其余区域的指针事件照常落到轨道上。
 *
 * 【跳转】双击条目（或点击条目右侧的跳转按钮）跳到该状态；窗口保持打开，
 * 便于连续前后对照。当前位置之前/之后（可重做部分）在同一条列表里。
 *
 * 【反馈】跳转到越界位置或原地不动时后端回 `ok = false`，前端静默跳过：
 * 界面零变化，不弹提示。
 */
export const UndoHistoryPanel: React.FC<{
    /** 打开时的锚点矩形（撤销/重做按钮），用于把窗口放到按钮下方。 */
    anchorRect?: DOMRect | null;
    onClose: () => void;
}> = ({ anchorRect, onClose }) => {
    const { t, locale } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();
    const s = useAppSelector(selectHistoryPanelState, shallowEqual);
    const listRef = useRef<HTMLDivElement | null>(null);
    const dragRef = useRef<{ offsetX: number; offsetY: number } | null>(null);
    const [position, setPosition] = useState(() => initialPanelPosition(anchorRect));

    // ── 窗口位置：锚点下方（视口内夹紧）──────────────────────────
    const moveTo = useCallback((clientX: number, clientY: number) => {
        const el = listRef.current?.closest<HTMLElement>("[data-undo-history-panel='1']");
        const width = el?.offsetWidth ?? 380;
        const height = el?.offsetHeight ?? 320;
        setPosition({
            x: Math.min(Math.max(0, clientX), Math.max(0, window.innerWidth - width)),
            y: Math.min(Math.max(0, clientY), Math.max(0, window.innerHeight - height)),
        });
    }, []);

    const onHeaderPointerDown = useCallback(
        (event: React.PointerEvent<HTMLDivElement>) => {
            if (event.button !== 0) return;
            // 关闭按钮走自己的 onClick：这里只处理标题栏拖拽。
            if ((event.target as HTMLElement).closest("button")) return;
            event.preventDefault();
            dragRef.current = {
                offsetX: event.clientX - position.x,
                offsetY: event.clientY - position.y,
            };
            const onMove = (ev: PointerEvent) => {
                const drag = dragRef.current;
                if (!drag) return;
                moveTo(ev.clientX - drag.offsetX, ev.clientY - drag.offsetY);
            };
            const onUp = () => {
                dragRef.current = null;
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                window.removeEventListener("pointercancel", onUp);
            };
            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
            window.addEventListener("pointercancel", onUp);
        },
        [moveTo, position.x, position.y],
    );

    // ── Escape 关闭（模态弹窗打开时不抢 Escape）──────────────────
    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            if (event.key !== "Escape") return;
            if (document.querySelector(".rt-BaseDialogOverlay")) return;
            event.preventDefault();
            onClose();
        };
        window.addEventListener("keydown", onKeyDown);
        return () => window.removeEventListener("keydown", onKeyDown);
    }, [onClose]);

    // ── 当前位置滚动到可见（撤销 / 重做 / 跳转后窗口不关闭也要跟上）──
    useEffect(() => {
        const current = listRef.current?.querySelector<HTMLElement>(
            "[data-undo-history-current='1']",
        );
        current?.scrollIntoView({ block: "nearest" });
    }, [s.position, s.records]);

    const timeFormatter = useMemo(
        () =>
            new Intl.DateTimeFormat(locale, {
                year: "numeric",
                month: "numeric",
                day: "numeric",
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit",
                hour12: false,
            }),
        [locale],
    );

    const labelOf = useCallback(
        (label: string | null): string => {
            const key = label ? `history_op_${label}` : "history_op_initial";
            const text = tAny(key);
            return typeof text === "string" && text.length > 0 ? text : (label ?? "—");
        },
        [tAny],
    );

    const jumpTo = useCallback(
        (index: number) => {
            if (index === s.position) return;
            void dispatch(setHistoryPositionRemote(index));
        },
        [dispatch, s.position],
    );

    const rows = useMemo(
        () => s.records.map((record, index) => ({ ...record, index })).reverse(),
        [s.records],
    );
    const countText = useMemo(
        () => tAny("undo_history_count").replace("{count}", String(s.records.length)),
        [s.records.length, tAny],
    );

    return createPortal(
        <div
            data-undo-history-panel="1"
            data-hs-floating-menu="1"
            className="fixed z-[500] flex w-[380px] max-w-[calc(100vw-16px)] flex-col rounded border border-qt-border bg-qt-window text-qt-text shadow-xl select-none"
            style={{ left: position.x, top: position.y, maxHeight: "min(60vh, 520px)" }}
            onPointerDown={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
        >
            {/* 标题栏：拖拽手柄 + 关闭 */}
            <div
                className="flex shrink-0 cursor-move items-center gap-2 border-b border-qt-border px-3 py-1.5"
                onPointerDown={onHeaderPointerDown}
            >
                <span className="text-[12px] font-medium">{tAny("undo_history_title")}</span>
                <span className="ml-auto text-[10px] text-qt-text-muted"></span>
                <button
                    type="button"
                    title={tAny("undo_history_close")}
                    className="rounded p-0.5 text-qt-text-muted transition-colors hover:bg-qt-button-hover hover:text-qt-text"
                    onClick={onClose}
                >
                    <Cross2Icon width="12" height="12" />
                </button>
            </div>

            {/* 条目列表：最新在最上（与 REAPER 一致），当前状态高亮 */}
            <div ref={listRef} className="min-h-0 flex-1 overflow-y-auto py-0.5 custom-scrollbar">
                {rows.map((row) => {
                    const isCurrent = row.index === s.position;
                    const isFuture = row.index > s.position;
                    return (
                        <div
                            key={row.index}
                            data-undo-history-current={isCurrent ? "1" : undefined}
                            className={`group flex items-center gap-2 px-2 py-[3px] text-[12px] ${
                                isCurrent
                                    ? "bg-qt-highlight/20 font-medium"
                                    : "hover:bg-qt-button-hover"
                            } ${isFuture ? "opacity-60" : ""}`}
                            title={isCurrent ? tAny("undo_history_current") : undefined}
                            onDoubleClick={() => jumpTo(row.index)}
                        >
                            <span
                                className="min-w-0 flex-1 truncate"
                                aria-current={isCurrent ? "true" : undefined}
                            >
                                {labelOf(row.label)}
                            </span>
                            <span className="shrink-0 text-[11px] tabular-nums text-qt-text-muted">
                                {timeFormatter.format(new Date(row.atMs))}
                            </span>
                            <button
                                type="button"
                                title={tAny("undo_history_jump")}
                                className={`shrink-0 rounded p-0.5 text-qt-text-muted transition-colors hover:bg-qt-button-hover hover:text-qt-text ${
                                    isCurrent ? "invisible" : "opacity-0 group-hover:opacity-100"
                                }`}
                                onClick={() => jumpTo(row.index)}
                            >
                                <EnterIcon width="12" height="12" />
                            </button>
                        </div>
                    );
                })}
            </div>

            <div className="shrink-0 space-y-1 border-t border-qt-border px-3 py-1.5">
                {/* 工程级：保存本工程时是否写出 UNDO 数据（随工程文件持久化）。
                    无论勾选与否，打开工程时都会尝试读取伴生文件。 */}
                <label className="flex cursor-pointer items-center gap-2 select-none">
                    <input
                        type="checkbox"
                        checked={s.saveUndoHistory}
                        onChange={(event) => {
                            void dispatch(
                                setProjectSaveUndoHistoryRemote(event.target.checked),
                            );
                        }}
                    />
                    <span className="text-[11px]">{tAny("undo_history_save_with_project")}</span>
                </label>
                {/* 全局：新工程的默认值（默认开启）。 */}
                <label className="flex cursor-pointer items-center gap-2 select-none">
                    <input
                        type="checkbox"
                        checked={s.saveUndoHistoryByDefault}
                        onChange={(event) => {
                            dispatch(setSaveUndoHistoryByDefault(event.target.checked));
                            void dispatch(persistUiSettings());
                        }}
                    />
                    <span className="text-[11px]">{tAny("undo_history_save_by_default")}</span>
                </label>
                <div className="text-[10px] text-qt-text-muted">{countText}</div>
            </div>
        </div>,
        document.body,
    );
};

/** 面板只消费「操作记录」与当前位置：其余 session 变化（播放轮询等）不重渲染。 */
const selectHistoryPanelState = (state: RootState) => ({
    records: state.session.historyRecords,
    position: state.session.historyUndoDepth,
    /** 工程级开关：保存本工程时是否写出 UNDO 数据。 */
    saveUndoHistory: state.session.project.saveUndoHistory,
    /** 全局默认：新工程是否默认保存 UNDO 数据。 */
    saveUndoHistoryByDefault: state.session.saveUndoHistoryByDefault,
});

/** 初始位置：锚点（撤销/重做按钮）下方；无锚点时贴近左上角。 */
function initialPanelPosition(anchorRect?: DOMRect | null): { x: number; y: number } {
    const width = 380;
    const top = anchorRect ? Math.round(anchorRect.bottom + 6) : 48;
    const left = anchorRect ? Math.round(anchorRect.left - 8) : 12;
    return {
        x: Math.max(0, Math.min(left, Math.max(0, window.innerWidth - width))),
        y: Math.max(0, top),
    };
}
