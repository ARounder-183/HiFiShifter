import React, { useCallback, useEffect, useRef, useState } from "react";
import { Dialog, Flex, Text, Button, ScrollArea, RadioGroup } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { paramsApi } from "../../services/api/params";

/** MIDI 轨道信息（与后端返回结构对齐） */
interface MidiTrackInfo {
    index: number;
    name: string;
    note_count: number;
    min_note: number;
    max_note: number;
}

interface MidiTrackSelectDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
    /** MIDI 文件路径（由文件对话框选定） */
    midiPath: string | null;
    /** 选区起始帧（与 paste_reaper_clipboard 一致） */
    selectionStartFrame?: number;
    /** 选区最大帧数（与 paste_reaper_clipboard 一致） */
    selectionMaxFrames?: number;
    /** 导入完成后的回调 */
    onImported?: (result: { notes_imported: number; frames_touched: number }) => void;
    /** 导入模式：pitchEdit（默认，写入 pitch_edit）或 clip（创建 MIDI clip）或 replaceMidi（替换已有 MIDI clip 数据） */
    mode?: "pitchEdit" | "clip" | "replaceMidi";
    /** clip 模式下的确认回调 */
    onImportAsClip?: (result: {
        trackIndices: number[];
        notesCount: number;
        midiPath: string;
        fillGaps: boolean;
        multiTrackMerge?: boolean;
        noteBpmMode?: string;
        specifiedBpm?: number;
        importBpmAsProject?: boolean;
        clipboardGuid?: string;
        closeLeadingGap?: boolean;
        importAsTempoMap?: boolean;
        importTempo?: boolean;
        importTimeSignature?: boolean;
        importKeySignature?: boolean;
    }) => void;
    /** 默认导入目标（弹窗首次打开时的选中项） */
    defaultImportTarget?: "pitchRef" | "pitchParam";
    /** 持久化的导入目标值（优先于 defaultImportTarget） */
    importTarget?: string;
    /** 导入目标变更回调（用于持久化） */
    onImportTargetChange?: (v: string) => void;
    /** 根轨是否已开启 Compose（用于 pitchEdit 模式的前置校验） */
    rootTrackComposeEnabled?: boolean;
    /** 请求开启 Compose 的回调（在 pitchEdit 模式下合成未开启时触发） */
    onRequestEnableCompose?: () => void;
    /** 剪贴板 GUID（从剪贴板读取的 MIDI 数据，非文件路径） */
    clipboardGuid?: string | null;
    /** 多轨合并选项（仅 clip / pitchRef 模式下生效） */
    multiTrackMerge?: boolean;
    /** 多轨合并选项变更回调 */
    onMultiTrackMergeChange?: (v: boolean) => void;
    /** 导入位置模式：projectStart / playhead / selection */
    importPosition?: string;
    /** 导入位置变更回调（用于持久化） */
    onImportPositionChange?: (position: string) => void;
    /** selection 模式是否可用（有选区且当前为选择工具） */
    selectionAvailable?: boolean;
    /** 是否填补音符之间的空隙 */
    fillGaps?: boolean;
    /** 填补空隙选项变更回调（用于持久化） */
    onFillGapsChange?: (fillGaps: boolean) => void;
    /** 当前工程 BPM */
    projectBpm?: number;
    /** 是否将 MIDI BPM 导入为工程 BPM */
    importBpmAsProject?: boolean;
    /** 导入为工程 BPM 选项变更回调（用于持久化） */
    onImportBpmAsProjectChange?: (v: boolean) => void;
    /** 音符 BPM 模式："midi" | "project" | "specified" */
    noteBpmMode?: string;
    /** 音符 BPM 模式变更回调（用于持久化） */
    onNoteBpmModeChange?: (v: string) => void;
    /** 指定 BPM 数值 */
    specifiedBpm?: number;
    /** 指定 BPM 数值变更回调（用于持久化） */
    onSpecifiedBpmChange?: (v: number) => void;
    /** 是否关闭开头空隙（将第一个音符对齐到导入位置） */
    closeLeadingGap?: boolean;
    /** 关闭开头空隙变更回调（用于持久化） */
    onCloseLeadingGapChange?: (v: boolean) => void;
    /** ── 导入为 Tempo Map（仅音高参考块目标显示） ── */
    importTempoMapEnabled?: boolean;
    onImportTempoMapEnabledChange?: (v: boolean) => void;
    /** 导入 Tempo（默认启用） */
    importTempoMapTempo?: boolean;
    onImportTempoMapTempoChange?: (v: boolean) => void;
    /** 导入拍号（默认启用） */
    importTempoMapTimeSignature?: boolean;
    onImportTempoMapTimeSignatureChange?: (v: boolean) => void;
    /** 导入音阶（默认关闭） */
    importTempoMapKeySignature?: boolean;
    onImportTempoMapKeySignatureChange?: (v: boolean) => void;
}

/** MIDI note number → 音名 */
function noteToName(note: number): string {
    const names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    const octave = Math.floor(note / 12) - 1;
    return `${names[note % 12]}${octave}`;
}

/**
 * MIDI 轨道选择弹窗
 *
 * 当 MIDI 文件包含多个有音符的轨道时，弹出此对话框让用户选择要导入的轨道。
 */
export const MidiTrackSelectDialog: React.FC<MidiTrackSelectDialogProps> = ({
    open,
    onOpenChange,
    midiPath,
    selectionStartFrame,
    selectionMaxFrames,
    onImported,
    mode = "pitchEdit",
    onImportAsClip,
    defaultImportTarget,
    importTarget,
    onImportTargetChange,
    rootTrackComposeEnabled,
    onRequestEnableCompose,
    clipboardGuid = null,
    importPosition = "selection",
    onImportPositionChange,
    selectionAvailable = false,
    fillGaps = false,
    onFillGapsChange,
    multiTrackMerge,
    onMultiTrackMergeChange,
    projectBpm,
    importBpmAsProject = false,
    onImportBpmAsProjectChange,
    noteBpmMode = "midi",
    onNoteBpmModeChange,
    specifiedBpm = 120,
    onSpecifiedBpmChange,
    closeLeadingGap = true,
    onCloseLeadingGapChange,
    importTempoMapEnabled = false,
    onImportTempoMapEnabledChange,
    importTempoMapTempo = true,
    onImportTempoMapTempoChange,
    importTempoMapTimeSignature = true,
    onImportTempoMapTimeSignatureChange,
    importTempoMapKeySignature = false,
    onImportTempoMapKeySignatureChange,
}) => {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;

    // 导入目标（统一弹窗用）：pitchRef = 创建音高参考块，pitchParam = 导入到音高参数
    const isReplaceMode = mode === "replaceMidi";
    const resolveImportTarget = () =>
        (importTarget as "pitchRef" | "pitchParam") ?? defaultImportTarget ?? "pitchParam";
    const [currentTarget, setCurrentTarget] = useState<"pitchRef" | "pitchParam">(
        resolveImportTarget(),
    );
    // 弹窗重新打开时重置目标
    useEffect(() => {
        if (open && !isReplaceMode) {
            setCurrentTarget(resolveImportTarget());
        }
    }, [open, defaultImportTarget, isReplaceMode, importTarget]);
    // 当 currentTarget 为 paramEditor 时，行为即 pitchEdit
    const effectiveMode = isReplaceMode
        ? "replaceMidi"
        : currentTarget === "pitchParam"
          ? "pitchEdit"
          : "clip";

    const [tracks, setTracks] = useState<MidiTrackInfo[]>([]);
    const [loading, setLoading] = useState(false);
    const [importing, setImporting] = useState(false);
    const [error, setError] = useState<string | null>(null);
    // 多选轨道：存储被选中的轨道 index 数组
    const [selectedTracks, setSelectedTracks] = useState<number[]>([]);

    // 内部状态：用户通过 Browse / Clipboard 选择的路径
    const [localMidiPath, setLocalMidiPath] = useState<string | null>(null);
    const [localClipboardGuid, setLocalClipboardGuid] = useState<string | null>(null);
    const [initialBpm, setInitialBpm] = useState<number | null>(null);
    const [midiHasBpm, setMidiHasBpm] = useState<boolean>(true);
    const [midiHasTimeSignature, setMidiHasTimeSignature] = useState<boolean>(false);
    const [midiHasKeySignature, setMidiHasKeySignature] = useState<boolean>(false);
    const [midiTempoPointCount, setMidiTempoPointCount] = useState<number>(0);
    const [midiTimeSigCount, setMidiTimeSigCount] = useState<number>(0);
    const [midiKeySigCount, setMidiKeySigCount] = useState<number>(0);
    const [composeConfirmOpen, setComposeConfirmOpen] = useState(false);
    const [readingClipboard, setReadingClipboard] = useState(false);
    const autoReadTriedRef = useRef(false);
    const composePendingRef = useRef(false);

    // 当前有效的 MIDI 路径（内部选择优先）
    const effectivePath = localMidiPath ?? midiPath;
    // 当前有效的剪贴板 GUID
    const effectiveClipboardGuid = localClipboardGuid ?? clipboardGuid;

    // 加载轨道列表的函数
    const loadTracks = useCallback(
        (path: string) => {
            console.info("[midi_import_ui] load_tracks:start", { midiPath: path });

            setLoading(true);
            setError(null);

            paramsApi
                .getMidiTracks(path)
                .then((res) => {
                    console.info("[midi_import_ui] load_tracks:response", res);
                    if (res.ok && res.tracks) {
                        setTracks(res.tracks);
                        setInitialBpm(res.initial_bpm ?? null);
                        setMidiHasBpm(res.has_bpm ?? true);
                        setMidiHasTimeSignature(res.has_time_signature ?? false);
                        setMidiHasKeySignature(res.has_key_signature ?? false);
                        setMidiTempoPointCount(res.tempo_point_count ?? 0);
                        setMidiTimeSigCount(res.time_signature_count ?? 0);
                        setMidiKeySigCount(res.key_signature_count ?? 0);
                        // 默认全选
                        setSelectedTracks(res.tracks.map((t) => t.index));
                    } else {
                        setError(res.error ?? tAny("midi_import_failed"));
                        setTracks([]);
                        setInitialBpm(null);
                        setMidiHasBpm(true);
                    }
                })
                .catch((err) => {
                    console.error("[midi_import_ui] load_tracks:error", err);
                    setError(tAny("midi_import_failed"));
                    setTracks([]);
                    setInitialBpm(null);
                    setMidiHasBpm(true);
                })
                .finally(() => setLoading(false));
        },
        [tAny],
    );

    // 从剪贴板 GUID 加载轨道（通过后端缓存查询，不重复读取剪贴板）
    const loadTracksFromClipboard = useCallback(
        (guid: string) => {
            console.info("[midi_import_ui] loadTracksFromClipboard:start", { guid });
            setLoading(true);
            setError(null);
            paramsApi
                .getMidiTracks("", guid)
                .then((res) => {
                    console.info("[midi_import_ui] loadTracksFromClipboard:response", res);
                    if (res.ok && res.tracks) {
                        setTracks(res.tracks);
                        setInitialBpm(res.initial_bpm ?? null);
                        setMidiHasBpm(res.has_bpm ?? true);
                        setMidiHasTimeSignature(res.has_time_signature ?? false);
                        setMidiHasKeySignature(res.has_key_signature ?? false);
                        setMidiTempoPointCount(res.tempo_point_count ?? 0);
                        setMidiTimeSigCount(res.time_signature_count ?? 0);
                        setMidiKeySigCount(res.key_signature_count ?? 0);
                        setSelectedTracks(res.tracks.map((t) => t.index));
                    } else {
                        setError(res.error ?? tAny("midi_clipboard_read_failed"));
                        setTracks([]);
                        setInitialBpm(null);
                        setMidiHasBpm(true);
                    }
                })
                .catch((err) => {
                    console.error("[midi_import_ui] loadTracksFromClipboard:error", err);
                    setError(tAny("midi_clipboard_read_failed"));
                    setTracks([]);
                    setInitialBpm(null);
                    setMidiHasBpm(true);
                })
                .finally(() => setLoading(false));
        },
        [tAny],
    );

    // 当弹窗打开且有 effectivePath 或 effectiveClipboardGuid，加载轨道列表
    useEffect(() => {
        if (!open) {
            setTracks([]);
            setError(null);
            setSelectedTracks([]);
            setInitialBpm(null);
            return;
        }
        // 剪贴板来源：若 tracks 已从 readMidiClipboardToMemory 直接加载，则跳过 getMidiTracks
        if (effectiveClipboardGuid) {
            if (tracks.length === 0) {
                loadTracksFromClipboard(effectiveClipboardGuid);
            }
            return;
        }
        if (!effectivePath) {
            setTracks([]);
            setError(null);
            setSelectedTracks([]);
            setInitialBpm(null);
            return;
        }

        loadTracks(effectivePath);
    }, [open, effectivePath, effectiveClipboardGuid, loadTracks]);

    // 弹窗关闭时重置内部状态
    useEffect(() => {
        if (!open) {
            setLocalMidiPath(null);
            setLocalClipboardGuid(null);
            setInitialBpm(null);
            setMidiHasBpm(true);
            setReadingClipboard(false);
            autoReadTriedRef.current = false;
            composePendingRef.current = false;
        }
    }, [open]);

    // MIDI 不含 BPM 时，若当前选中 "MIDI 自身 BPM"，显示为 "当前工程 BPM"（不持久化）
    const displayNoteBpmMode = !midiHasBpm && noteBpmMode === "midi" ? "project" : noteBpmMode;

    // 弹窗打开时，若无预设来源，尝试自动读取剪贴板中的 Standard MIDI File 数据
    useEffect(() => {
        if (!open || isReplaceMode || effectivePath || effectiveClipboardGuid) return;
        if (autoReadTriedRef.current) return;
        autoReadTriedRef.current = true;
        paramsApi
            .readMidiClipboardToMemory()
            .then((res) => {
                if (res.ok && res.guid) {
                    setLocalClipboardGuid(res.guid);
                    if (res.tracks && res.tracks.length > 0) {
                        setTracks(res.tracks);
                        setInitialBpm(res.initial_bpm ?? null);
                        setMidiHasBpm(res.has_bpm ?? true);
                        setMidiHasTimeSignature(res.has_time_signature ?? false);
                        setMidiHasKeySignature(res.has_key_signature ?? false);
                        setMidiTempoPointCount(res.tempo_point_count ?? 0);
                        setMidiTimeSigCount(res.time_signature_count ?? 0);
                        setMidiKeySigCount(res.key_signature_count ?? 0);
                        setSelectedTracks(res.tracks.map((t) => t.index));
                    }
                }
            })
            .catch(() => {
                // 剪贴板无可用的 MIDI 数据，静默忽略
            });
    }, [open, isReplaceMode, effectivePath, effectiveClipboardGuid]);

    // Browse 按钮：打开原生文件对话框
    const handleBrowse = useCallback(async () => {
        try {
            const coreApi = (await import("../../services/api/core")).coreApi;
            const picked = await coreApi.openMidiDialog();
            if (!(picked as { ok?: boolean }).ok) return;
            if ((picked as { canceled?: boolean }).canceled || !(picked as { path?: string }).path)
                return;
            setLocalMidiPath((picked as { path: string }).path);
            setLocalClipboardGuid(null);
            setError(null);
        } catch {
            // 静默忽略
        }
    }, []);

    // Read from Clipboard 按钮
    const handleReadClipboard = useCallback(async () => {
        setReadingClipboard(true);
        setError(null);
        try {
            const res = await paramsApi.readMidiClipboardToMemory();
            if (res.ok && res.guid) {
                setLocalMidiPath(null);
                setLocalClipboardGuid(res.guid);
                // 直接从返回结果设置轨道，避免再次请求
                if (res.tracks && res.tracks.length > 0) {
                    setTracks(res.tracks);
                    setInitialBpm(res.initial_bpm ?? null);
                    setMidiHasBpm(res.has_bpm ?? true);
                    setMidiHasTimeSignature(res.has_time_signature ?? false);
                    setMidiHasKeySignature(res.has_key_signature ?? false);
                    setMidiTempoPointCount(res.tempo_point_count ?? 0);
                    setMidiTimeSigCount(res.time_signature_count ?? 0);
                    setMidiKeySigCount(res.key_signature_count ?? 0);
                    setSelectedTracks(res.tracks.map((t) => t.index));
                }
            } else {
                const errorKey = res.error ?? "midi_clipboard_read_failed";
                setError(tAny(errorKey));
            }
        } catch {
            setError(tAny("midi_clipboard_read_failed"));
        } finally {
            setReadingClipboard(false);
        }
    }, [tAny]);

    const handleImport = useCallback(async () => {
        if ((!effectivePath && !effectiveClipboardGuid) || selectedTracks.length === 0) return;

        // pitchEdit 模式下，若根轨未开启 Compose，先弹出确认对话框
        if (effectiveMode === "pitchEdit" && rootTrackComposeEnabled === false) {
            composePendingRef.current = true;
            setComposeConfirmOpen(true);
            return;
        }

        setImporting(true);
        try {
            const trackIndices = selectedTracks;
            const midiSrc = effectivePath ?? "";

            if (effectiveMode === "clip" || effectiveMode === "replaceMidi") {
                const notesCount = tracks
                    .filter((t) => selectedTracks.includes(t.index))
                    .reduce((sum, t) => sum + t.note_count, 0);
                const tempoMapImportEnabled =
                    importTempoMapEnabled && currentTarget === "pitchRef" && !isReplaceMode;
                onImportAsClip?.({
                    trackIndices,
                    notesCount,
                    midiPath: midiSrc,
                    fillGaps,
                    multiTrackMerge,
                    // 导入为 Tempo Map 时音符时间沿用 MIDI 自身速度（不做全局缩放）。
                    noteBpmMode: tempoMapImportEnabled ? "midi" : noteBpmMode,
                    specifiedBpm:
                        !tempoMapImportEnabled && noteBpmMode === "specified"
                            ? specifiedBpm
                            : undefined,
                    importBpmAsProject:
                        !tempoMapImportEnabled && importBpmAsProject ? true : undefined,
                    clipboardGuid: effectiveClipboardGuid ?? undefined,
                    closeLeadingGap,
                    importAsTempoMap: tempoMapImportEnabled || undefined,
                    importTempo: importTempoMapTempo || undefined,
                    importTimeSignature: importTempoMapTimeSignature || undefined,
                    importKeySignature: importTempoMapKeySignature || undefined,
                });
                onOpenChange(false);
                return;
            }

            // 根据导入位置模式计算帧偏移
            let effectivePosition = importPosition;
            if (effectivePosition === "selection") {
                if (selectionStartFrame == null || !selectionAvailable) {
                    effectivePosition = "playhead"; // 回退
                }
            }
            const startFrame =
                effectivePosition === "projectStart"
                    ? 0
                    : effectivePosition === "selection"
                      ? selectionStartFrame
                      : undefined;
            const maxFrames = effectivePosition === "selection" ? selectionMaxFrames : undefined;

            console.info("[midi_import_ui] import:start", {
                midiPath: midiSrc,
                clipboardGuid: effectiveClipboardGuid,
                trackIndices,
                importPosition,
                effectivePosition,
                startFrame,
                maxFrames,
                noteBpmMode,
                specifiedBpm,
                importBpmAsProject,
            });
            const res = await paramsApi.importMidiToPitch(
                midiSrc,
                trackIndices,
                startFrame,
                maxFrames,
                fillGaps || undefined,
                noteBpmMode,
                noteBpmMode === "specified" ? specifiedBpm : undefined,
                importBpmAsProject || undefined,
                effectiveClipboardGuid ?? undefined,
                closeLeadingGap,
            );
            console.info("[midi_import_ui] import:response", res);
            if (res.ok) {
                onImported?.({
                    notes_imported: res.notes_imported ?? 0,
                    frames_touched: res.frames_touched ?? 0,
                });
                onOpenChange(false);
            } else {
                const errKey = res.error ?? "midi_import_failed";
                // 尝试翻译已知的错误键
                const knownErrors: Record<string, string> = {
                    file_not_found: tAny("midi_file_not_found"),
                    no_notes_in_track: tAny("midi_no_notes"),
                    no_frames_touched: tAny("midi_no_frames_touched"),
                    no_pitch_line_selected: tAny("vs_paste_no_pitch_line"),
                    pitch_requires_compose: tAny("pitch_requires_compose"),
                    pitch_requires_algo: tAny("pitch_requires_algo"),
                };
                setError(knownErrors[errKey] ?? errKey);
            }
        } catch (err) {
            console.error("[midi_import_ui] import:error", err);
            setError(tAny("midi_import_failed"));
        } finally {
            setImporting(false);
        }
    }, [
        effectivePath,
        selectionStartFrame,
        selectionMaxFrames,
        selectedTracks,
        onImported,
        onImportAsClip,
        onOpenChange,
        tAny,
        effectiveMode,
        tracks,
        importPosition,
        selectionAvailable,
        fillGaps,
        multiTrackMerge,
        noteBpmMode,
        specifiedBpm,
        importBpmAsProject,
        importTempoMapEnabled,
        importTempoMapTempo,
        importTempoMapTimeSignature,
        importTempoMapKeySignature,
        currentTarget,
        isReplaceMode,
        effectiveClipboardGuid,
        closeLeadingGap,
        rootTrackComposeEnabled,
    ]);

    // Compose 确认回调：开启合成后继续导入
    const handleComposeConfirm = useCallback(() => {
        setComposeConfirmOpen(false);
        composePendingRef.current = false;
        onRequestEnableCompose?.();
        // 重新触发导入（此时 rootTrackComposeEnabled 可能还未更新，但后端不会再报错）
        setImporting(true);
        const midiSrc = effectivePath ?? "";
        const trackIndices = selectedTracks;
        let effectivePosition = importPosition;
        if (effectivePosition === "selection") {
            if (selectionStartFrame == null || !selectionAvailable) {
                effectivePosition = "playhead";
            }
        }
        const startFrame =
            effectivePosition === "projectStart"
                ? 0
                : effectivePosition === "selection"
                  ? selectionStartFrame
                  : undefined;
        const maxFrames = effectivePosition === "selection" ? selectionMaxFrames : undefined;
        paramsApi
            .importMidiToPitch(
                midiSrc,
                trackIndices,
                startFrame,
                maxFrames,
                fillGaps || undefined,
                noteBpmMode,
                noteBpmMode === "specified" ? specifiedBpm : undefined,
                importBpmAsProject || undefined,
                effectiveClipboardGuid ?? undefined,
                closeLeadingGap,
            )
            .then((res) => {
                if (res.ok) {
                    onImported?.({
                        notes_imported: res.notes_imported ?? 0,
                        frames_touched: res.frames_touched ?? 0,
                    });
                    onOpenChange(false);
                } else {
                    const knownErrors: Record<string, string> = {
                        file_not_found: tAny("midi_file_not_found"),
                        no_notes_in_track: tAny("midi_no_notes"),
                        no_frames_touched: tAny("midi_no_frames_touched"),
                        no_pitch_line_selected: tAny("vs_paste_no_pitch_line"),
                        pitch_requires_compose: tAny("pitch_requires_compose"),
                        pitch_requires_algo: tAny("pitch_requires_algo"),
                    };
                    setError(
                        knownErrors[res.error ?? ""] ?? res.error ?? tAny("midi_import_failed"),
                    );
                }
            })
            .catch(() => {
                setError(tAny("midi_import_failed"));
            })
            .finally(() => {
                setImporting(false);
            });
    }, [
        effectivePath,
        effectiveClipboardGuid,
        selectedTracks,
        importPosition,
        selectionStartFrame,
        selectionMaxFrames,
        selectionAvailable,
        fillGaps,
        noteBpmMode,
        specifiedBpm,
        importBpmAsProject,
        closeLeadingGap,
        onRequestEnableCompose,
        onImported,
        onOpenChange,
        tAny,
    ]);

    const handleComposeDecline = useCallback(() => {
        setComposeConfirmOpen(false);
        composePendingRef.current = false;
    }, []);

    return (
        <Dialog.Root open={open} onOpenChange={onOpenChange}>
            <Dialog.Content maxWidth="520px">
                <Dialog.Title>
                    {effectiveMode === "replaceMidi"
                        ? tAny("midi_replace_title")
                        : currentTarget === "pitchParam"
                          ? tAny("midi_import_title")
                          : tAny("midi_import_clip_title")}
                </Dialog.Title>
                <Dialog.Description size="2" color="gray">
                    {effectiveMode === "replaceMidi"
                        ? tAny("midi_replace_desc")
                        : currentTarget === "pitchParam"
                          ? tAny("midi_import_desc")
                          : tAny("midi_import_clip_desc")}
                </Dialog.Description>

                {/* ── 导入目标选择（replace 模式不显示） ── */}
                {!isReplaceMode && (
                    <Flex direction="column" gap="1" mt="3">
                        <Text size="1" weight="medium">
                            {tAny("midi_import_target")}
                        </Text>
                        <RadioGroup.Root
                            value={currentTarget}
                            onValueChange={(v) => {
                                const target = v as "pitchRef" | "pitchParam";
                                setCurrentTarget(target);
                                onImportTargetChange?.(v);
                            }}
                        >
                            <Flex gap="3">
                                <label className="flex items-center gap-1 cursor-pointer">
                                    <RadioGroup.Item value="pitchParam" />
                                    <Text size="1">{tAny("midi_import_target_pitch_param")}</Text>
                                </label>
                                <label className="flex items-center gap-1 cursor-pointer">
                                    <RadioGroup.Item value="pitchRef" />
                                    <Text size="1">{tAny("midi_import_target_pitch_block")}</Text>
                                </label>
                            </Flex>
                        </RadioGroup.Root>
                    </Flex>
                )}

                {/* ── 文件选择区域（始终显示） ── */}
                <Flex direction="column" gap="1" mt="3">
                    <Text size="1" weight="medium">
                        {tAny("midi_file_path")}
                    </Text>
                    <Flex gap="2" align="center">
                        <input
                            type="text"
                            className="flex-1 px-2 py-1 text-xs rounded border border-qt-border bg-qt-base text-qt-text"
                            readOnly
                            value={
                                effectiveClipboardGuid
                                    ? tAny("midi_clipboard_midi_prefix") +
                                      effectiveClipboardGuid +
                                      ".mid"
                                    : effectivePath
                                      ? effectivePath
                                      : tAny("midi_no_file_selected")
                            }
                            style={{
                                color: effectivePath || effectiveClipboardGuid ? undefined : "#888",
                                cursor: "default",
                                minWidth: 0,
                            }}
                        />
                        <Button variant="soft" size="1" onClick={handleBrowse} disabled={importing}>
                            {tAny("midi_browse")}
                        </Button>
                        <Button
                            variant="soft"
                            size="1"
                            onClick={handleReadClipboard}
                            disabled={importing || readingClipboard}
                        >
                            {readingClipboard
                                ? tAny("midi_importing")
                                : tAny("midi_read_clipboard")}
                        </Button>
                    </Flex>
                </Flex>

                {loading && (
                    <Flex justify="center" py="4">
                        <Text size="2" color="gray">
                            {tAny("loading")}
                        </Text>
                    </Flex>
                )}

                {error && (
                    <Flex py="2">
                        <Text size="2" color="red">
                            {error}
                        </Text>
                    </Flex>
                )}

                {effectivePath && !loading && !error && tracks.length === 0 && (
                    <Flex py="4" justify="center">
                        <Text size="2" color="gray">
                            {tAny("midi_no_tracks")}
                        </Text>
                    </Flex>
                )}

                {!loading && tracks.length > 0 && (
                    <>
                        {/* 全选 / 全不选 快捷按钮 */}
                        <Flex gap="2" mt="3">
                            <Button
                                variant="soft"
                                color="gray"
                                size="1"
                                onClick={() => setSelectedTracks(tracks.map((t) => t.index))}
                            >
                                {tAny("midi_select_all")}
                            </Button>
                            <Button
                                variant="soft"
                                color="gray"
                                size="1"
                                onClick={() => setSelectedTracks([])}
                            >
                                {tAny("midi_deselect_all")}
                            </Button>
                            {initialBpm != null && (
                                <Text
                                    size="1"
                                    color={midiHasBpm ? "gray" : "red"}
                                    className="ml-auto self-center"
                                >
                                    {midiHasBpm
                                        ? tAny("midi_midi_bpm_label").replace(
                                              "{bpm}",
                                              initialBpm.toFixed(2),
                                          )
                                        : `${tAny("midi_no_bpm")}`}
                                </Text>
                            )}
                        </Flex>

                        <ScrollArea
                            style={{ maxHeight: 200 }}
                            className="mt-2 rounded border border-qt-border"
                        >
                            <Flex direction="column" gap="0">
                                {/* 各个轨道选项（多选） */}
                                {tracks.map((track) => (
                                    <label
                                        key={track.index}
                                        className="flex items-center gap-2 px-3 py-2 hover:bg-qt-highlight cursor-pointer border-b border-qt-border last:border-b-0"
                                    >
                                        <input
                                            type="checkbox"
                                            className="w-4 h-4"
                                            checked={selectedTracks.includes(track.index)}
                                            onChange={(e) => {
                                                if (e.target.checked) {
                                                    setSelectedTracks([
                                                        ...selectedTracks,
                                                        track.index,
                                                    ]);
                                                } else {
                                                    setSelectedTracks(
                                                        selectedTracks.filter(
                                                            (i) => i !== track.index,
                                                        ),
                                                    );
                                                }
                                            }}
                                        />
                                        <Flex direction="column" gap="0" className="flex-1 min-w-0">
                                            <Text size="2" weight="medium" className="truncate">
                                                {track.name || `Track ${track.index + 1}`}
                                            </Text>
                                            <Flex gap="2">
                                                <Text size="1" color="gray">
                                                    {tAny("midi_track_notes").replace(
                                                        "{count}",
                                                        String(track.note_count),
                                                    )}
                                                </Text>
                                                <Text size="1" color="gray">
                                                    {tAny("midi_track_range")
                                                        .replace(
                                                            "{min}",
                                                            noteToName(track.min_note),
                                                        )
                                                        .replace(
                                                            "{max}",
                                                            noteToName(track.max_note),
                                                        )}
                                                </Text>
                                            </Flex>
                                        </Flex>
                                    </label>
                                ))}
                            </Flex>
                        </ScrollArea>

                        {/* ── BPM 选项（在多轨合并和填补空隙上方） ── */}
                        {/* 将 MIDI BPM 导入为工程 BPM */}
                        <label
                            className={`flex items-center gap-2 mt-3 ${
                                midiHasBpm && !importTempoMapEnabled
                                    ? "cursor-pointer"
                                    : "opacity-50"
                            }`}
                        >
                            <input
                                type="checkbox"
                                checked={importBpmAsProject && !importTempoMapEnabled}
                                onChange={(e) => onImportBpmAsProjectChange?.(e.target.checked)}
                                disabled={!midiHasBpm || importTempoMapEnabled}
                                className="w-4 h-4"
                            />
                            <Text size="1" color={midiHasBpm && !importTempoMapEnabled ? undefined : "gray"}>
                                {tAny("midi_import_bpm_as_project")}
                            </Text>
                        </label>

                        {/* ── 导入为 Tempo Map（仅“音高参考块”目标显示） ── */}
                        {currentTarget === "pitchRef" && !isReplaceMode && (
                            <Flex direction="column" gap="1" mt="3" className="rounded border border-qt-border p-2">
                                <label className="flex items-center gap-2 cursor-pointer">
                                    <input
                                        type="checkbox"
                                        checked={importTempoMapEnabled}
                                        onChange={(e) => onImportTempoMapEnabledChange?.(e.target.checked)}
                                        className="w-4 h-4"
                                    />
                                    <Text size="1" weight="medium">
                                        {tAny("midi_import_as_tempo_map")}
                                    </Text>
                                </label>
                                <Flex direction="column" gap="1" className="ml-6 mt-1">
                                    <label
                                        className={`flex items-center gap-2 ${
                                            importTempoMapEnabled && midiHasBpm
                                                ? "cursor-pointer"
                                                : "opacity-50"
                                        }`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={importTempoMapTempo}
                                            onChange={(e) =>
                                                onImportTempoMapTempoChange?.(e.target.checked)
                                            }
                                            disabled={!importTempoMapEnabled || !midiHasBpm}
                                            className="w-4 h-4"
                                        />
                                        <Text
                                            size="1"
                                            color={
                                                importTempoMapEnabled && midiHasBpm ? undefined : "gray"
                                            }
                                        >
                                            {tAny("midi_import_tempo_map_tempo")}
                                            {midiTempoPointCount > 1
                                                ? ` (${midiTempoPointCount})`
                                                : ""}
                                        </Text>
                                    </label>
                                    <label
                                        className={`flex items-center gap-2 ${
                                            importTempoMapEnabled && midiHasTimeSignature
                                                ? "cursor-pointer"
                                                : "opacity-50"
                                        }`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={importTempoMapTimeSignature}
                                            onChange={(e) =>
                                                onImportTempoMapTimeSignatureChange?.(e.target.checked)
                                            }
                                            disabled={!importTempoMapEnabled || !midiHasTimeSignature}
                                            className="w-4 h-4"
                                        />
                                        <Text
                                            size="1"
                                            color={
                                                importTempoMapEnabled && midiHasTimeSignature
                                                    ? undefined
                                                    : "gray"
                                            }
                                        >
                                            {tAny("midi_import_tempo_map_time_signature")}
                                            {midiTimeSigCount > 0 ? ` (${midiTimeSigCount})` : ""}
                                        </Text>
                                    </label>
                                    <label
                                        className={`flex items-center gap-2 ${
                                            importTempoMapEnabled && midiHasKeySignature
                                                ? "cursor-pointer"
                                                : "opacity-50"
                                        }`}
                                    >
                                        <input
                                            type="checkbox"
                                            checked={importTempoMapKeySignature}
                                            onChange={(e) =>
                                                onImportTempoMapKeySignatureChange?.(e.target.checked)
                                            }
                                            disabled={!importTempoMapEnabled || !midiHasKeySignature}
                                            className="w-4 h-4"
                                        />
                                        <Text
                                            size="1"
                                            color={
                                                importTempoMapEnabled && midiHasKeySignature
                                                    ? undefined
                                                    : "gray"
                                            }
                                        >
                                            {tAny("midi_import_tempo_map_key_signature")}
                                            {midiKeySigCount > 0 ? ` (${midiKeySigCount})` : ""}
                                        </Text>
                                    </label>
                                </Flex>
                                <Text size="1" color="gray" className="ml-6 mt-1">
                                    {tAny("midi_import_as_tempo_map_hint")}
                                </Text>
                            </Flex>
                        )}

                        {/* 音符 BPM 设置 */}
                        <Flex direction="column" gap="1" mt="2">
                            <Text size="2" weight="medium">
                                {tAny("midi_note_bpm")}
                            </Text>
                            <RadioGroup.Root
                                value={importTempoMapEnabled ? "midi" : displayNoteBpmMode}
                                onValueChange={(v) => {
                                    if (!importTempoMapEnabled) onNoteBpmModeChange?.(v);
                                }}
                            >
                                <Flex direction="column" gap="1">
                                    <label
                                        className={`flex items-center gap-1 ${
                                            midiHasBpm && !importTempoMapEnabled
                                                ? "cursor-pointer"
                                                : "opacity-50"
                                        }`}
                                    >
                                        <RadioGroup.Item
                                            value="midi"
                                            disabled={!midiHasBpm || importTempoMapEnabled}
                                        />
                                        <Text
                                            size="1"
                                            color={
                                                midiHasBpm && !importTempoMapEnabled ? undefined : "gray"
                                            }
                                        >
                                            {tAny("midi_note_bpm_midi")}
                                        </Text>
                                    </label>
                                    <label
                                        className={`flex items-center gap-1 ${
                                            importTempoMapEnabled ? "opacity-50" : "cursor-pointer"
                                        }`}
                                    >
                                        <RadioGroup.Item value="project" disabled={importTempoMapEnabled} />
                                        <Text size="1" color={importTempoMapEnabled ? "gray" : undefined}>
                                            {tAny("midi_note_bpm_project")}
                                            {projectBpm != null
                                                ? ` (${projectBpm.toFixed(2)} BPM)`
                                                : ""}
                                        </Text>
                                    </label>
                                    <label
                                        className={`flex items-center gap-1 ${
                                            importTempoMapEnabled ? "opacity-50" : "cursor-pointer"
                                        }`}
                                    >
                                        <RadioGroup.Item value="specified" disabled={importTempoMapEnabled} />
                                        <Text size="1" color={importTempoMapEnabled ? "gray" : undefined}>
                                            {tAny("midi_note_bpm_specified")}
                                        </Text>
                                    </label>
                                    {noteBpmMode === "specified" && !importTempoMapEnabled && (
                                        <Flex gap="2" align="center" className="ml-5 mt-1">
                                            <input
                                                type="number"
                                                className="w-20 px-2 py-1 text-xs rounded border border-qt-border bg-qt-base text-qt-text"
                                                value={specifiedBpm}
                                                min={1}
                                                max={999}
                                                step={1}
                                                onKeyDown={(e) => {
                                                    if (e.key === "ArrowUp") {
                                                        e.preventDefault();
                                                        const next = specifiedBpm + 1;
                                                        if (next <= 999)
                                                            onSpecifiedBpmChange?.(next);
                                                    } else if (e.key === "ArrowDown") {
                                                        e.preventDefault();
                                                        const next = specifiedBpm - 1;
                                                        if (next >= 1) onSpecifiedBpmChange?.(next);
                                                    }
                                                }}
                                                onWheel={(e) => {
                                                    e.preventDefault();
                                                    const dir = e.deltaY < 0 ? 1 : -1;
                                                    const next = specifiedBpm + dir;
                                                    if (next >= 1 && next <= 999)
                                                        onSpecifiedBpmChange?.(next);
                                                }}
                                                onChange={(e) => {
                                                    const v = parseFloat(e.target.value);
                                                    if (!isNaN(v) && v > 0) {
                                                        onSpecifiedBpmChange?.(v);
                                                    }
                                                }}
                                            />
                                            <Text size="1" color="gray">
                                                {tAny("midi_specified_bpm_placeholder")}
                                            </Text>
                                        </Flex>
                                    )}
                                </Flex>
                            </RadioGroup.Root>
                        </Flex>
                    </>
                )}

                {(effectivePath || effectiveClipboardGuid) && !loading && tracks.length > 0 && (
                    <>
                        {/* 导入位置选项（仅在 paramEditor 目标下显示） */}
                        {currentTarget === "pitchParam" && !isReplaceMode && (
                            <Flex direction="column" gap="1" mt="3">
                                <Text size="2" weight="medium">
                                    {tAny("midi_import_position")}
                                </Text>
                                <RadioGroup.Root
                                    value={
                                        importPosition === "selection" && !selectionAvailable
                                            ? "playhead"
                                            : importPosition
                                    }
                                    onValueChange={(v) => onImportPositionChange?.(v)}
                                >
                                    <Flex gap="3">
                                        <label className="flex items-center gap-1 cursor-pointer">
                                            <RadioGroup.Item value="projectStart" />
                                            <Text size="1">
                                                {tAny("midi_import_position_start")}
                                            </Text>
                                        </label>
                                        <label className="flex items-center gap-1 cursor-pointer">
                                            <RadioGroup.Item value="playhead" />
                                            <Text size="1">
                                                {tAny("midi_import_position_playhead")}
                                            </Text>
                                        </label>
                                        <label className="flex items-center gap-1 cursor-pointer">
                                            <RadioGroup.Item
                                                value="selection"
                                                disabled={!selectionAvailable}
                                            />
                                            <Text
                                                size="1"
                                                color={selectionAvailable ? undefined : "gray"}
                                            >
                                                {tAny("midi_import_position_selection")}
                                            </Text>
                                        </label>
                                    </Flex>
                                </RadioGroup.Root>
                            </Flex>
                        )}

                        {/* 多轨合并选项 */}
                        {!isReplaceMode && (
                            <label
                                className={`flex items-center gap-2 mt-3 ${
                                    currentTarget === "pitchParam" ? "opacity-60" : "cursor-pointer"
                                }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={
                                        currentTarget === "pitchParam"
                                            ? true
                                            : (multiTrackMerge ?? true)
                                    }
                                    onChange={(e) =>
                                        currentTarget !== "pitchParam" &&
                                        onMultiTrackMergeChange?.(e.target.checked)
                                    }
                                    disabled={currentTarget === "pitchParam"}
                                    className="w-4 h-4"
                                />
                                <Text size="1">{tAny("midi_multi_track_merge")}</Text>
                            </label>
                        )}

                        {/* 关闭开头空隙选项 */}
                        <label className="flex items-center gap-2 mt-3 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={closeLeadingGap ?? true}
                                onChange={(e) => onCloseLeadingGapChange?.(e.target.checked)}
                                className="w-4 h-4"
                            />
                            <Text size="1">{tAny("midi_close_leading_gap")}</Text>
                        </label>

                        {/* 填补空隙选项 */}
                        <label className="flex items-center gap-2 mt-3 cursor-pointer">
                            <input
                                type="checkbox"
                                checked={fillGaps}
                                onChange={(e) => onFillGapsChange?.(e.target.checked)}
                                className="w-4 h-4"
                            />
                            <Text size="1">{tAny("midi_fill_gaps")}</Text>
                        </label>

                        <Flex justify="end" gap="2" mt="4">
                            <Button
                                variant="soft"
                                color="gray"
                                onClick={() => onOpenChange(false)}
                                disabled={importing}
                            >
                                {tAny("kb_close")}
                            </Button>
                            <Button
                                onClick={handleImport}
                                disabled={
                                    importing ||
                                    loading ||
                                    tracks.length === 0 ||
                                    selectedTracks.length === 0 ||
                                    !!error
                                }
                            >
                                {importing
                                    ? tAny("midi_importing")
                                    : effectiveMode === "replaceMidi"
                                      ? tAny("midi_replace_button")
                                      : currentTarget === "pitchParam"
                                        ? tAny("midi_import")
                                        : tAny("midi_create_clip")}
                            </Button>
                        </Flex>
                    </>
                )}
            </Dialog.Content>

            {/* Compose 未开启确认对话框 */}
            <Dialog.Root open={composeConfirmOpen} onOpenChange={setComposeConfirmOpen}>
                <Dialog.Content maxWidth="400px">
                    <Dialog.Title>{tAny("midi_compose_required_title")}</Dialog.Title>
                    <Dialog.Description size="2" mt="2">
                        {tAny("midi_compose_required_message")}
                    </Dialog.Description>
                    <Flex justify="end" gap="2" mt="4">
                        <Button variant="soft" color="gray" onClick={handleComposeDecline}>
                            {tAny("cancel")}
                        </Button>
                        <Button onClick={handleComposeConfirm}>{tAny("ok")}</Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
        </Dialog.Root>
    );
};
