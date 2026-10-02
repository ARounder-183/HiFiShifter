/**
 * 文件浏览器的一行。
 *
 * 【为什么从面板里搬出来】行是纯展示：给它条目、几个布尔量与已格式化好的文本，
 * 它不认识 Redux、Tauri、i18n 或匹配类型。搬出来后面板只留编排，行也能单独推敲
 * 三个状态通道（选中 / 光标 / 播放）的表达。
 */

import React from "react";
import { FileIcon, PlayIcon, StopIcon } from "@radix-ui/react-icons";

import { AppListRow, type AppListRowDensity } from "../../../ui";
import type { FileEntry } from "../../../services/api/fileBrowser";
import {
    isAudioFile,
    isMidiFile,
    isProjectFile,
    isVideoFile,
} from "../../../features/fileBrowser/fileKinds";
import { AudioIcon, FolderIcon, MidiIcon, ProjectIcon, VideoIcon } from "./fileIcons";

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
    onClickAudio: (entry: FileEntry, ev?: React.MouseEvent) => void;
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
        onClickAudio,
        onPointerDownForDrag,
        onContextMenu,
        isDragging,
        density = "compact",
        pathHint,
        matchReason,
        detailText,
    }) => {
        const isAudio = isAudioFile(entry);
        const isMidi = isMidiFile(entry);
        const isProject = isProjectFile(entry);
        const isDraggable = isAudio || isMidi || isProject;
        // 既不能打开、也不能拖拽的行（例如 .txt）此前被标成 disabled（50% 透明 +
        // aria-disabled）。那在语义上是错的：右键菜单仍可对它复制路径 / 重命名 /
        // 删除 / 查看属性 —— "什么都不能做"是假话。现在只把图标调暗表示"不是可导入
        // 的媒体"，行本身保持正常对比度与可交互性。
        const isInert = !entry.isDir && !isDraggable;

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
                onClick={isAudio ? (ev) => onClickAudio(entry, ev) : undefined}
            >
                {/* 图标 */}
                <span className="shrink-0 w-[14px] flex items-center justify-center">
                    {entry.isDir ? (
                        <FolderIcon className="text-yellow-500" />
                    ) : isAudio ? (
                        isPlaying ? (
                            <StopIcon width="12" height="12" className="text-qt-highlight" />
                        ) : isVideoFile(entry) ? (
                            <VideoIcon className="text-purple-400" />
                        ) : (
                            <AudioIcon className="text-blue-400" />
                        )
                    ) : isMidi ? (
                        <MidiIcon className="text-qt-highlight" />
                    ) : isProject ? (
                        // 工程文件高亮：橙色星标文档图标（备份文件如 .hshp-bak 不在此列）。
                        <ProjectIcon className="text-amber-400" />
                    ) : (
                        <FileIcon
                            width="12"
                            height="12"
                            className={
                                isInert ? "text-qt-text-muted opacity-60" : "text-qt-text-muted"
                            }
                        />
                    )}
                </span>

                {/* 文件名 + 路径提示 */}
                <div className="flex flex-col min-w-0 flex-1">
                    <span
                        className={`hs-type-label ${isProject ? "truncate text-amber-300" : "truncate"}`}
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
