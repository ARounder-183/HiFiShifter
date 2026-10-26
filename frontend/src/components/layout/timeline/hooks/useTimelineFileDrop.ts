// 时间轴 DOM 拖入预览与落点导入。拆自 TimelinePanel。
import { isPluginMode, canImportHostAudio } from "../../../../services/hostCapabilities";
import { isTauriRuntime } from "../../../../services/hostDetect";
import {
    importAudioAtPosition,
    importAudioFileAtPosition,
    importMultipleAudioFilesAtPosition,
} from "../../../../features/session/sessionSlice";
import { emitExternalFileAction } from "../../../../features/session/projectOpenEvents";
import {
    detectExternalPathAction,
    extractLocalFilePath,
    hasFileDrag,
    isAcceptedDropFile,
    isAcceptedDropPath,
} from "../../timeline";
import type { AppDispatch } from "../../../../app/store";
import type React from "react";

type DropPreview = {
    path: string;
    fileName: string;
    trackId: string | null;
    startSec: number;
    durationSec: number;
};

type Deps = {
    tauriDraggedPathRef: React.MutableRefObject<string | null>;
    tauriLastDropPathRef: React.MutableRefObject<string | null>;
    tauriDropHandledAtRef: React.MutableRefObject<number>;
    beatFromClientX: (clientX: number, rect: DOMRect, scrollLeft: number) => number;
    dragScrollLeftOf: (el: HTMLElement) => number;
    resolveTrackIdAtClientY: (clientY: number) => string | null;
    dropPreview: DropPreview | null;
    setDropPreview: React.Dispatch<React.SetStateAction<DropPreview | null>>;
    ensureDropPreviewDuration: (path: string) => void;
    dispatch: AppDispatch;
    onMidiClipPathChange: (path: string | null) => void;
    onMidiClipStartSecChange: (sec: number) => void;
    onMidiClipTrackIdChange: (trackId: string | null) => void;
    onMidiClipDialogOpenChange: (open: boolean) => void;
};

export function useTimelineFileDrop({
    tauriDraggedPathRef,
    tauriLastDropPathRef,
    tauriDropHandledAtRef,
    beatFromClientX,
    dragScrollLeftOf,
    resolveTrackIdAtClientY,
    dropPreview,
    setDropPreview,
    ensureDropPreviewDuration,
    dispatch,
    onMidiClipPathChange,
    onMidiClipStartSecChange,
    onMidiClipTrackIdChange,
    onMidiClipDialogOpenChange,
}: Deps) {
    const handleTimelineDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        const dt = e.dataTransfer;
        const tauriPath = tauriDraggedPathRef.current;
        const hasDomFile = Boolean(dt?.files && dt.files.length > 0);
        const isTauri = isTauriRuntime();
        if (!isTauri && !hasFileDrag(dt) && !hasDomFile && !tauriPath) return;
        e.preventDefault();
        const info = extractLocalFilePath(dt);
        const el = e.currentTarget as HTMLDivElement;
        const bounds = el.getBoundingClientRect();
        const beat = beatFromClientX(e.clientX, bounds, dragScrollLeftOf(el));
        const trackId = resolveTrackIdAtClientY(e.clientY);
        const path = info?.path || tauriPath || "";
        const fileName =
            info?.name ||
            (tauriPath
                ? String(tauriPath.split(/[\\/]/).pop() ?? tauriPath)
                : hasDomFile
                  ? String(dt?.files?.[0]?.name ?? "Audio")
                  : "Audio");
        const dragAction = detectExternalPathAction(path);
        // 准入判据：未知扩展名**不再**显示落点预览。此前这里只挡"已知的非音频种类"，
        // 未知类型（`dragAction === null`）会一路穿透到音频导入分支并建出 Clip。
        if (path && !isAcceptedDropPath(path)) {
            if (dropPreview !== null) setDropPreview(null);
            return;
        }
        if (path && dragAction !== "importAudio" && dragAction !== "importMidi") {
            if (dropPreview !== null) setDropPreview(null);
            return;
        }
        // 插件模式也保留 MIDI 落点预览：松手会打开导入对话框，而对话框里的
        // "导入到音高曲线"在插件里可用（见 `MidiTrackSelectDialog`）。此前这里
        // 整条挡掉，理由是"插件没有 MIDI 导入链路" —— 那条理由已经不成立。
        // dragover 在文件悬停期间高频连发；落点各字段都没变时不 setState（新对象
        // 无法让 React 跳过重渲，逐事件重渲整个面板纯属浪费）。
        const nextDurationSec = dragAction === "importMidi" ? 2 : 0;
        if (
            dropPreview !== null &&
            dropPreview.path === path &&
            dropPreview.fileName === fileName &&
            dropPreview.trackId === trackId &&
            dropPreview.startSec === beat &&
            dropPreview.durationSec === nextDurationSec
        ) {
            return;
        }
        if (dragAction === "importMidi") {
            // MIDI 文件使用默认时长显示 drop preview
            setDropPreview({
                path,
                fileName,
                trackId,
                startSec: beat,
                durationSec: 2,
            });
        } else {
            if (path) {
                ensureDropPreviewDuration(path);
            }
            setDropPreview({
                path,
                fileName,
                trackId,
                startSec: beat,
                durationSec: 0,
            });
        }
    };

    const handleTimelineDrop = (e: React.DragEvent<HTMLDivElement>) => {
        if (isPluginMode() && !canImportHostAudio()) {
            e.preventDefault();
            return;
        }
        const dt = e.dataTransfer;
        const tauriPath = tauriDraggedPathRef.current;
        const lastTauriDropPath = tauriLastDropPathRef.current;
        const hasDomFile = Boolean(dt?.files && dt.files.length > 0);
        const isTauri = isTauriRuntime();
        if (!isTauri && !hasFileDrag(dt) && !hasDomFile && !tauriPath) return;
        e.preventDefault();

        if (isTauri && Date.now() - (tauriDropHandledAtRef.current || 0) < 500) {
            setDropPreview(null);
            return;
        }

        const info = extractLocalFilePath(dt);
        const el = e.currentTarget as HTMLDivElement;
        const bounds = el.getBoundingClientRect();
        const beat = beatFromClientX(e.clientX, bounds, dragScrollLeftOf(el));
        const trackId = resolveTrackIdAtClientY(e.clientY);
        setDropPreview(null);
        const resolvedPath = info?.path || lastTauriDropPath || tauriPath;
        if (resolvedPath) {
            tauriDraggedPathRef.current = null;
            tauriLastDropPathRef.current = null;
            const actionKind = detectExternalPathAction(resolvedPath);
            // 准入判据（缺陷修复点）：未知扩展名不再落入下面的音频导入默认分支。
            if (actionKind === null) return;
            if (actionKind === "importMidi") {
                onMidiClipPathChange(resolvedPath);
                onMidiClipStartSecChange(beat);
                onMidiClipTrackIdChange(trackId);
                onMidiClipDialogOpenChange(true);
                return;
            }
            if (actionKind !== "importAudio") {
                // 目录不经这条通道：`ExternalFileActionKind` 表达不了"导入目录"
                // （它只带一个路径，没有落点与选项），而且 HTML5 的 dataTransfer
                // 拿不到目录信息 —— 目录由 Tauri 原生拖放通道处理。
                if (actionKind !== "importFolder") emitExternalFileAction(actionKind, resolvedPath);
                return;
            }
            void dispatch(
                importAudioAtPosition({
                    audioPath: resolvedPath,
                    trackId,
                    startSec: beat,
                }),
            );
            return;
        }

        if (isTauri) {
            window.setTimeout(() => {
                const p = tauriLastDropPathRef.current || tauriDraggedPathRef.current;
                if (!p) return;
                tauriDraggedPathRef.current = null;
                tauriLastDropPathRef.current = null;
                const actionKind = detectExternalPathAction(p);
                // 同上的准入判据：延迟分支也必须拒绝未知类型。
                if (actionKind === null) return;
                if (actionKind === "importMidi") {
                    onMidiClipPathChange(p);
                    onMidiClipStartSecChange(beat);
                    onMidiClipTrackIdChange(trackId);
                    onMidiClipDialogOpenChange(true);
                    return;
                }
                if (actionKind !== "importAudio") {
                    // 同上：目录不走这条通道。
                    if (actionKind !== "importFolder") emitExternalFileAction(actionKind, p);
                    return;
                }
                void dispatch(
                    importAudioAtPosition({
                        audioPath: p,
                        trackId,
                        startSec: beat,
                    }),
                );
            }, 0);
        }

        const fallbackFile = dt.files?.[0] ?? null;
        if (isPluginMode() && dt.files?.length > 1) {
            const files = Array.from(dt.files).filter(isAcceptedDropFile);
            if (files.length)
                void dispatch(
                    importMultipleAudioFilesAtPosition({
                        files,
                        mode: "across-time",
                        trackId,
                        startSec: beat,
                    }),
                );
            return;
        }
        // 无本地路径的兜底分支同样必须过准入判据：此前它**完全不做校验**就把任何
        // `File` 按音频导入（后端内容嗅探会放行，于是无关文件也变成 Clip）。
        if (fallbackFile && isAcceptedDropFile(fallbackFile)) {
            void dispatch(
                importAudioFileAtPosition({
                    file: fallbackFile,
                    trackId,
                    startSec: beat,
                }),
            );
        }
    };

    return { handleTimelineDragOver, handleTimelineDrop };
}
