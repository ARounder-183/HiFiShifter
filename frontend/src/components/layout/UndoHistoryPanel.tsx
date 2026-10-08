import React, { useCallback, useEffect, useMemo, useRef } from "react";
import { EnterIcon } from "@radix-ui/react-icons";
import { shallowEqual } from "react-redux";

import { useAppDispatch, useAppSelector } from "../../app/hooks";
import type { RootState } from "../../app/store";
import { useI18n } from "../../i18n/I18nProvider";
import { AppForm, AppSwitchRow } from "../../ui/Field";
import { isPluginMode, dawControlledReason } from "../../services/hostCapabilities";
import {
    persistUiSettings,
    setHistoryPositionRemote,
    setProjectSaveUndoHistoryRemote,
    setSaveUndoHistoryByDefault,
} from "../../features/session/sessionSlice";

/**
 * 「操作记录」面板（REAPER Undo History 风格）。
 *
 * 【它是一个可停靠面板】窗口位置、尺寸、浮动/停靠、关闭都由停靠系统统一
 * 管理（见 `components/dock`）—— 面板自己不再维护 `position` state，也不
 * 自己画标题栏与拖拽手柄。此前那份手搓实现（portal + fixed + 头部拖拽）已
 * 被停靠系统的通用能力取代，同样的能力现在对每个面板都成立。
 *
 * 【跳转】双击条目（或点击条目右侧的跳转按钮）跳到该状态；面板保持打开，
 * 便于连续前后对照。当前位置之前/之后（可重做部分）在同一条列表里。
 *
 * 【反馈】跳转到越界位置或原地不动时后端回 `ok = false`，前端静默跳过：
 * 界面零变化，不弹提示。
 */
export const UndoHistoryPanel: React.FC = () => {
    const { tf, locale } = useI18n();
    const dispatch = useAppDispatch();
    const s = useAppSelector(selectHistoryPanelState, shallowEqual);
    const listRef = useRef<HTMLDivElement | null>(null);

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
            const text = tf(key);
            return typeof text === "string" && text.length > 0 ? text : (label ?? "—");
        },
        [tf],
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
        () => tf("undo_history_count").replace("{count}", String(s.records.length)),
        [s.records.length, tf],
    );

    return (
        <div
            data-undo-history-panel="1"
            className="flex h-full w-full min-h-0 flex-col bg-qt-window text-qt-text select-none"
            onPointerDown={(event) => event.stopPropagation()}
            onContextMenu={(event) => event.preventDefault()}
        >
            {/* 条目列表：最新在最上（与 REAPER 一致），当前状态高亮 */}
            <div
                ref={listRef}
                className="hs-scroll-gutter min-h-0 flex-1 overflow-y-auto py-0.5 custom-scrollbar"
            >
                {rows.map((row) => {
                    const isCurrent = row.index === s.position;
                    const isFuture = row.index > s.position;
                    return (
                        <div
                            key={row.index}
                            data-undo-history-current={isCurrent ? "1" : undefined}
                            className={`group flex items-center gap-2 px-2 py-[3px] text-qt-sm ${
                                isCurrent
                                    ? "bg-qt-highlight/20 font-medium"
                                    : "hover:bg-qt-button-hover"
                            } ${isFuture ? "opacity-60" : ""}`}
                            data-tooltip={isCurrent ? tf("undo_history_current") : undefined}
                            onDoubleClick={() => jumpTo(row.index)}
                        >
                            <span
                                className="min-w-0 flex-1 truncate"
                                aria-current={isCurrent ? "true" : undefined}
                            >
                                {labelOf(row.label)}
                            </span>
                            <span className="shrink-0 text-qt-xs tabular-nums text-qt-text-muted">
                                {timeFormatter.format(new Date(row.atMs))}
                            </span>
                            <button
                                type="button"
                                data-tooltip={tf("undo_history_jump")}
                                /* 图标按钮没有可见文字，`data-tooltip` 不是可访问名称。 */
                                aria-label={tf("undo_history_jump")}
                                /*
                                 * 悬停之外还要在**聚焦**时显形：这个按钮默认
                                 * `opacity-0`，Tab 到它时若仍透明，焦点环就画在
                                 * 一个看不见的元素上 —— 键盘用户以为焦点丢了。
                                 */
                                className={`shrink-0 rounded p-0.5 text-qt-text-muted transition-colors hover:bg-qt-button-hover hover:text-qt-text focus-visible:opacity-100 group-focus-within:opacity-100 ${
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
                {/* 【为什么在插件里禁用】`set_project_save_undo_history` 只存在于独立
                    App，而插件里根本没有"工程文件"可写 —— 勾选它只会静默失败。 */}
                <AppForm booleanRow="leading">
                    <AppSwitchRow
                        control="checkbox"
                        label={tf("undo_history_save_with_project")}
                        checked={s.saveUndoHistory}
                        disabled={isPluginMode()}
                        hint={isPluginMode() ? dawControlledReason() : undefined}
                        onCheckedChange={(checked) => {
                            void dispatch(setProjectSaveUndoHistoryRemote(checked));
                        }}
                    />
                </AppForm>
                {/* 全局：新工程的默认值（默认开启）。 */}
                <AppForm booleanRow="leading">
                    <AppSwitchRow
                        control="checkbox"
                        label={tf("undo_history_save_by_default")}
                        checked={s.saveUndoHistoryByDefault}
                        onCheckedChange={(checked) => {
                            dispatch(setSaveUndoHistoryByDefault(checked));
                            void dispatch(persistUiSettings());
                        }}
                    />
                </AppForm>
                <div className="text-qt-micro text-qt-text-muted">{countText}</div>
            </div>
        </div>
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
