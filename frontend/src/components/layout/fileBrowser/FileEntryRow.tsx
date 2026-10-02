/**
 * 文件浏览器的一行。
 *
 * 【为什么从面板里搬出来】行是纯展示：给它条目、几个布尔量与已格式化好的文本，
 * 它不认识 Redux、Tauri、i18n 或匹配类型。搬出来后面板只留编排，行也能单独推敲
 * 三个状态通道（选中 / 光标 / 播放）的表达。
 */

import React from "react";
import { PlayIcon, StopIcon } from "@radix-ui/react-icons";

import { AppListRow, type AppListRowDensity } from "../../../ui";
import type { FileEntry } from "../../../services/api/fileBrowser";
import { isAudioFile, isMidiFile, isProjectFile } from "../../../features/fileBrowser/fileKinds";
import { FileKindIcon } from "./fileIcons";

export interface FileEntryRowProps {
    entry: FileEntry;
    /** 在 displayEntries 中的下标，用于 roving tabindex 的焦点登记。 */
    index: number;
    /** roving tabindex：活动行为 0，其余为 -1。 */
    tabIndex: number;
    /** 该行是否是键盘光标所在行（走描边通道，与 `isSelected` 的背景通道正交）。 */
    active: boolean;
    onFocus: (index: number) => void;
    registerRowRef: (index: number, el: HTMLDivElement | null) => void;
    isPlaying: boolean;
    isSelected?: boolean;
    onDoubleClickDir: (dirPath: string) => void;
    /**
     * 单击一行。
     *
     * 【为什么所有行都要有】此前只有音频行接了 `onClick`，于是左键点一个 `.txt`
     * 既不选中也不做任何事 —— 随后的 Ctrl+C / Delete / F2 就没有作用对象，而右键
     * 菜单却能对它操作。行的左键语义应当是"选中它"，由调用方再按类型决定是否
     * 额外触发试听。
     */
    onRowClick: (entry: FileEntry, ev: React.MouseEvent) => void;
    onPointerDownForDrag: (e: React.PointerEvent<HTMLDivElement>, entry: FileEntry) => void;
    onContextMenu: (e: React.MouseEvent, entry: FileEntry) => void;
    isDragging: boolean;
    density?: AppListRowDensity;
    /** 第二行的所在目录提示（搜索模式下有意义）。 */
    pathHint?: string;
    /**
     * 命中原因文案（「匹配拼音 zhuge」）。由调用方格式化好再传进来，
     * 行组件保持纯展示，不认识 i18n 键与匹配类型。
     */
    matchReason?: string;
    /** 右侧详情列文本（大小 / 修改日期）。同样由调用方格式化。 */
    detailText?: string;
}

export const FileEntryRow: React.FC<FileEntryRowProps> = React.memo(
    ({
        entry,
        index,
        tabIndex,
        active,
        onFocus,
        registerRowRef,
        isPlaying,
        isSelected,
        onDoubleClickDir,
        onRowClick,
        onPointerDownForDrag,
        onContextMenu,
        isDragging,
        density = "compact",
        pathHint,
        matchReason,
        detailText,
    }) => {
        const isAudio = isAudioFile(entry);
        // 可拖拽 = 音频/视频 + MIDI + 工程文件（与时间轴、参数编辑器的拖放准入一致）。
        const isDraggable = isAudio || isMidiFile(entry) || isProjectFile(entry);

        return (
            <AppListRow
                ref={(el) => registerRowRef(index, el)}
                role="option"
                selected={isSelected}
                active={active}
                density={density}
                tabIndex={tabIndex}
                onFocus={() => onFocus(index)}
                className={[
                    // 试听高亮：改动前 20%，选中态（22%）优先。
                    isPlaying && !isSelected
                        ? "bg-[color-mix(in_oklab,var(--qt-highlight)_20%,transparent)]"
                        : "",
                    isDragging ? "opacity-50" : "",
                ]
                    .filter(Boolean)
                    .join(" ")}
                onPointerDown={isDraggable ? (e) => onPointerDownForDrag(e, entry) : undefined}
                onContextMenu={(e) => onContextMenu(e, entry)}
                onDoubleClick={entry.isDir ? () => onDoubleClickDir(entry.path) : undefined}
                onClick={(ev) => onRowClick(entry, ev)}
            >
                {/* 图标。播放中换成停止图标（这是行的状态，不属于类型）。 */}
                <span className="shrink-0 w-[14px] flex items-center justify-center">
                    {isAudio && isPlaying ? (
                        <StopIcon width="12" height="12" className="text-qt-highlight" />
                    ) : (
                        <FileKindIcon entry={entry} />
                    )}
                </span>

                {/* 文件名 + 路径提示 */}
                <div className="flex flex-col min-w-0 flex-1">
                    <span
                        className={`hs-type-label ${
                            isProjectFile(entry) ? "truncate text-amber-300" : "truncate"
                        }`}
                        data-tooltip={entry.name}
                    >
                        {entry.name}
                        {entry.isDir ? "/" : ""}
                    </span>
                    {(pathHint || matchReason) && (
                        <span
                            className="hs-type-caption truncate leading-none"
                            style={{ fontSize: "var(--qt-fs-micro)" }}
                        >
                            {pathHint}
                            {pathHint && matchReason ? " · " : ""}
                            {matchReason}
                        </span>
                    )}
                </div>

                {/* 右侧信息 */}
                {detailText && (
                    <span
                        className="hs-type-caption shrink-0"
                        style={{ fontSize: "var(--qt-fs-micro)" }}
                    >
                        {detailText}
                    </span>
                )}

                {/* 音频播放指示 */}
                {isPlaying && (
                    <PlayIcon
                        width="10"
                        height="10"
                        className="shrink-0 text-qt-highlight animate-pulse"
                    />
                )}
            </AppListRow>
        );
    },
);

FileEntryRow.displayName = "FileEntryRow";
