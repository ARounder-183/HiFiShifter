/*
 * 导出音频配置对话框。
 * 负责收集导出模式、时间范围、输出路径与分轨命名/目标选择，并调用后端统一导出命令。
 */

import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import {
    Button,
    Dialog,
    Flex,
    SegmentedControl,
    Select,
    Slider,
    Text,
    TextField,
} from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { exportAudioAdvanced } from "../../features/session/sessionSlice";
import { fileBrowserApi } from "../../services/api/fileBrowser";
import {
    coreApi,
    type AdvancedExportRequest,
    type ChannelMode,
    type DitherMode,
    type ExportEncoderSpec,
    type ExportFormat,
    type FlacBitDepth,
    type Mp3Tags,
    type WavBitDepth,
} from "../../services/api/core";
import {
    applyExtensionToFileName,
    FLAC_COMPRESSION_RANGE,
    MP3_BITRATES,
    MP3_VBR_AVG_KBPS,
    nearestAllowedSampleRate,
    sampleRateOptions,
} from "../../utils/exportFormat";
import { ProgressBar } from "../ProgressBar";
import type { TrackInfo } from "../../features/session/sessionTypes";
import { applySelectWheelChange } from "../../utils/selectWheel";

interface ExportAudioDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

type ExportMode = "project" | "separated";
type ExportRangeKind = "all" | "custom";
type Mp3ModeKind = "cbr" | "vbr";

/** FLAC 压缩级别滑条的滚轮步进选项（"0" ~ "8"，供 applySelectWheelChange 使用）。 */
const FLAC_LEVEL_OPTIONS = Array.from(
    { length: FLAC_COMPRESSION_RANGE.max - FLAC_COMPRESSION_RANGE.min + 1 },
    (_, index) => String(FLAC_COMPRESSION_RANGE.min + index),
);

type TargetKind = "root" | "sub";

interface TargetOption {
    id: string;
    kind: TargetKind;
    trackId: string;
    trackName: string;
    trackIndex: number;
    excludedByRule: boolean;
    label: string;
}

interface TargetGroup {
    id: string;
    title: string;
    isGroup: boolean;
    options: TargetOption[];
}

function normalizePathKey(input: string) {
    return input.trim().replace(/\\/g, "/").replace(/\/+/g, "/").toLowerCase();
}

function buildTargetGroups(
    tracks: TrackInfo[],
    clips: { trackId: string; muted: boolean }[],
    mainSuffix: string,
    subSuffix: string,
): TargetGroup[] {
    if (tracks.length === 0) return [];

    const indexWidth = String(tracks.length).length;
    const indexMap = new Map<string, number>();
    const trackMap = new Map<string, TrackInfo>();
    const parentMap = new Map<string, string | null>();

    tracks.forEach((track, index) => {
        const trackIndex = index + 1;
        indexMap.set(track.id, trackIndex);
        trackMap.set(track.id, track);
        parentMap.set(track.id, track.parentId ?? null);
    });

    const rootTrackIds = tracks
        .filter((track) => (track.parentId ?? null) === null)
        .map((track) => track.id);

    // 预计算每个轨道是否包含片段以及是否包含未静音片段，以避免在 isTrackExcludedByRule 中反复遍历 clips。
    const trackClipStats = new Map<string, { hasAnyClip: boolean; hasAnyUnmutedClip: boolean }>();
    clips.forEach((clip) => {
        const existing = trackClipStats.get(clip.trackId) ?? {
            hasAnyClip: false,
            hasAnyUnmutedClip: false,
        };
        const updated = {
            hasAnyClip: true,
            hasAnyUnmutedClip: existing.hasAnyUnmutedClip || !clip.muted,
        };
        trackClipStats.set(clip.trackId, updated);
    });

    function isTrackExcludedByRule(trackId: string): boolean {
        const track = trackMap.get(trackId);
        if (!track) return true;
        if (track.muted) return true;

        const stats = trackClipStats.get(trackId);
        if (!stats) return true;
        if (!stats.hasAnyClip) return true;
        if (!stats.hasAnyUnmutedClip) return true;

        return false;
    }

    function resolveRootTrackId(trackId: string): string {
        let current = trackId;
        let safety = 0;
        while (safety < tracks.length + 2) {
            const parentId = parentMap.get(current) ?? null;
            if (!parentId) return current;
            current = parentId;
            safety += 1;
        }
        return trackId;
    }

    function formatTrackLabel(trackIndex: number, trackName: string, suffix: string): string {
        const indexText = String(trackIndex).padStart(indexWidth, "0");
        return `[${indexText}] ${trackName} ${suffix}`;
    }

    return rootTrackIds
        .map((rootTrackId) => {
            const rootTrack = trackMap.get(rootTrackId);
            if (!rootTrack) return null;
            const rootIndex = indexMap.get(rootTrackId) ?? 1;

            const childTracks = tracks.filter((track) => {
                if (track.id === rootTrackId) return false;
                return resolveRootTrackId(track.id) === rootTrackId;
            });
            const isGroup = childTracks.length > 0;
            const branchTracks = [rootTrack, ...childTracks];
            const rootSelfExcluded = isTrackExcludedByRule(rootTrack.id);
            const rootBranchExcluded = branchTracks.every((track) =>
                isTrackExcludedByRule(track.id),
            );

            const options: TargetOption[] = [];
            options.push({
                id: `root:${rootTrackId}`,
                kind: "root",
                trackId: rootTrackId,
                trackName: rootTrack.name,
                trackIndex: rootIndex,
                excludedByRule: isGroup ? rootBranchExcluded : rootSelfExcluded,
                label: isGroup
                    ? formatTrackLabel(rootIndex, rootTrack.name, mainSuffix)
                    : `[${String(rootIndex).padStart(indexWidth, "0")}] ${rootTrack.name}`,
            });

            if (isGroup) {
                options.push({
                    id: `sub:${rootTrackId}`,
                    kind: "sub",
                    trackId: rootTrackId,
                    trackName: rootTrack.name,
                    trackIndex: rootIndex,
                    excludedByRule: rootSelfExcluded,
                    label: formatTrackLabel(rootIndex, rootTrack.name, subSuffix),
                });
            }

            for (const track of childTracks) {
                const currentIndex = indexMap.get(track.id) ?? 1;
                options.push({
                    id: `sub:${track.id}`,
                    kind: "sub",
                    trackId: track.id,
                    trackName: track.name,
                    trackIndex: currentIndex,
                    excludedByRule: isTrackExcludedByRule(track.id),
                    label: formatTrackLabel(currentIndex, track.name, subSuffix),
                });
            }

            const rootTitleIndex = String(rootIndex).padStart(indexWidth, "0");
            return {
                id: rootTrackId,
                title: `[${rootTitleIndex}] ${rootTrack.name}`,
                isGroup,
                options,
            };
        })
        .filter((item): item is TargetGroup => Boolean(item));
}

export function ExportAudioDialog({ open, onOpenChange }: ExportAudioDialogProps) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const dispatch = useAppDispatch();
    const session = useAppSelector((state) => state.session);

    const [mode, setMode] = useState<ExportMode>("project");
    const [rangeKind, setRangeKind] = useState<ExportRangeKind>("all");
    const [customStartSec, setCustomStartSec] = useState("0");
    const [customEndSec, setCustomEndSec] = useState("0");
    const [projectOutputDir, setProjectOutputDir] = useState("");
    const [projectFileName, setProjectFileName] = useState("<ProjectName>.wav");
    const [separatedOutputDir, setSeparatedOutputDir] = useState("");
    const [separatedNamePattern, setSeparatedNamePattern] = useState(
        "<ExportIndex>_<TrackName>.wav",
    );
    const [sampleRate, setSampleRate] = useState("48000");
    const [sampleRateNotice, setSampleRateNotice] = useState("");
    // ── 编码格式与参数（持久化经 get_export_audio_defaults / 导出成功回写）──
    const [format, setFormat] = useState<ExportFormat>("wav");
    const [channelMode, setChannelMode] = useState<ChannelMode>("stereo");
    const [dither, setDither] = useState<DitherMode>("none");
    const [wavBitDepth, setWavBitDepth] = useState<WavBitDepth>("f32");
    const [flacBitDepth, setFlacBitDepth] = useState<FlacBitDepth>("i24");
    const [flacLevel, setFlacLevel] = useState<number>(FLAC_COMPRESSION_RANGE.default);
    const [mp3Mode, setMp3Mode] = useState<Mp3ModeKind>("vbr");
    const [mp3Bitrate, setMp3Bitrate] = useState(320);
    const [mp3Quality, setMp3Quality] = useState(2);
    const [mp3Tags, setMp3Tags] = useState<Mp3Tags>({});
    const [encoderOpen, setEncoderOpen] = useState(false);
    const [selectedTargetIds, setSelectedTargetIds] = useState<string[]>([]);
    const [errorText, setErrorText] = useState("");
    const [submitting, setSubmitting] = useState(false);
    const [exportProgress, setExportProgress] = useState<{
        active: boolean;
        mode: ExportMode | null;
        progress: number | null;
        current: number | null;
        total: number | null;
    }>({ active: false, mode: null, progress: null, current: null, total: null });
    const [displayProgress, setDisplayProgress] = useState(0);
    const [keepProgressVisible, setKeepProgressVisible] = useState(false);
    const [awaitingConflictDecision, setAwaitingConflictDecision] = useState(false);
    const [activeInputKey, setActiveInputKey] = useState<
        | "projectOutputDir"
        | "projectFileName"
        | "separatedOutputDir"
        | "separatedNamePattern"
        | null
    >(null);
    const activeInputRef = useRef<HTMLInputElement | null>(null);
    const [conflictDialog, setConflictDialog] = useState<{
        open: boolean;
        path: string;
        applyAll: boolean;
        kind: "exists" | "source-path";
    }>({ open: false, path: "", applyAll: false, kind: "exists" });

    const sourceClipPathKeys = useMemo(() => {
        const keys = new Set<string>();
        for (const clip of session.clips) {
            if (!clip.sourcePath) continue;
            const key = normalizePathKey(clip.sourcePath);
            if (key) keys.add(key);
        }
        return keys;
    }, [session.clips]);
    const conflictResolverRef = useRef<
        ((value: { choice: "overwrite" | "skip" | "cancel"; applyAll: boolean }) => void) | null
    >(null);

    const targetGroups = useMemo(
        () =>
            buildTargetGroups(
                session.tracks,
                session.clips.map((clip) => ({
                    trackId: clip.trackId,
                    muted: Boolean(clip.muted),
                })),
                tAny("export_track_label_root_suffix"),
                tAny("export_track_label_sub_suffix"),
            ),
        [session.tracks, session.clips, tAny],
    );

    const allTargets = useMemo(
        () => targetGroups.flatMap((group) => group.options),
        [targetGroups],
    );

    // 初始化只应在 open 变为 true 时执行一次。此前依赖数组里包含
    // session.projectSec 与 targetGroups（随 tracks/clips 引用变化），
    // 对话框打开期间任何后台更新（导入完成、录音入库、工程加载）都会
    // 重跑该 effect，把用户已填写的输出目录/文件名/时间范围/目标选择
    // 静默重置为硬编码回退值。projectSec 经 ref 读取打开瞬间的值。
    const projectSecAtOpenRef = useRef(session.projectSec);
    useEffect(() => {
        if (open) {
            projectSecAtOpenRef.current = session.projectSec;
        }
    }, [open, session.projectSec]);

    useEffect(() => {
        if (!open) return;

        setMode("project");
        setRangeKind("all");
        setCustomStartSec("0");
        setCustomEndSec(String(Math.max(0, Math.ceil(projectSecAtOpenRef.current))));
        setProjectOutputDir("");
        setProjectFileName("<ProjectName>.wav");
        setSeparatedOutputDir("");
        setSeparatedNamePattern("<ExportIndex>_<TrackName>.wav");
        setSampleRate("48000");
        setSampleRateNotice("");
        setFormat("wav");
        setChannelMode("stereo");
        setDither("none");
        setWavBitDepth("f32");
        setFlacBitDepth("i24");
        setFlacLevel(FLAC_COMPRESSION_RANGE.default);
        setMp3Mode("vbr");
        setMp3Bitrate(320);
        setMp3Quality(2);
        setMp3Tags({});
        setEncoderOpen(false);
        setSubmitting(false);
        setExportProgress({
            active: false,
            mode: null,
            progress: null,
            current: null,
            total: null,
        });
        setDisplayProgress(0);
        setKeepProgressVisible(false);
        setAwaitingConflictDecision(false);

        const defaultSelected = targetGroups.flatMap((group) => {
            return group.options
                .filter((option) => option.kind === "root" && !option.excludedByRule)
                .map((option) => option.id);
        });
        setSelectedTargetIds(defaultSelected);
        setErrorText("");
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在打开时重置一次表单
    }, [open]);

    useEffect(() => {
        if (!open) return;
        let disposed = false;

        async function loadDefaults() {
            try {
                const defaults = await coreApi.getExportAudioDefaults();
                if (disposed || !defaults?.ok) return;
                setProjectOutputDir(defaults.projectOutputDir ?? "");
                setSeparatedOutputDir(defaults.separatedOutputDir ?? "");
                const nextFormat: ExportFormat =
                    defaults.format === "mp3" || defaults.format === "flac"
                        ? defaults.format
                        : "wav";
                setFormat(nextFormat);
                setChannelMode(defaults.encoder?.channelMode === "mono" ? "mono" : "stereo");
                setDither(defaults.encoder?.dither === "tpdf" ? "tpdf" : "none");
                setWavBitDepth(
                    defaults.encoder?.wav?.bitDepth === "i16" ||
                        defaults.encoder?.wav?.bitDepth === "i24"
                        ? defaults.encoder.wav.bitDepth
                        : "f32",
                );
                setFlacBitDepth(defaults.encoder?.flac?.bitDepth === "i16" ? "i16" : "i24");
                setFlacLevel(
                    typeof defaults.encoder?.flac?.compressionLevel === "number"
                        ? defaults.encoder.flac.compressionLevel
                        : FLAC_COMPRESSION_RANGE.default,
                );
                const persistedMode = defaults.encoder?.mp3?.mode;
                if (persistedMode?.mode === "cbr") {
                    setMp3Mode("cbr");
                    setMp3Bitrate(persistedMode.bitrateKbps);
                } else if (persistedMode?.mode === "vbr") {
                    setMp3Mode("vbr");
                    setMp3Quality(persistedMode.qualityIndex);
                }
                setMp3Tags({
                    title: defaults.encoder?.mp3?.tags?.title ?? "",
                    artist: defaults.encoder?.mp3?.tags?.artist ?? "",
                    album: defaults.encoder?.mp3?.tags?.album ?? "",
                    comment: defaults.encoder?.mp3?.tags?.comment ?? "",
                });
                // 文件名 / 命名模板的旧后缀按持久化格式归一。
                setProjectFileName(
                    applyExtensionToFileName(
                        defaults.projectFileName ?? "<ProjectName>.wav",
                        nextFormat,
                    ),
                );
                setSeparatedNamePattern(
                    applyExtensionToFileName(
                        defaults.separatedFileName ?? "<ExportIndex>_<TrackName>.wav",
                        nextFormat,
                    ),
                );
                const defaultRate = Number(defaults.sampleRate ?? 48000);
                const corrected = nearestAllowedSampleRate(nextFormat, defaultRate);
                if (corrected != null) {
                    setSampleRate(String(corrected));
                    setSampleRateNotice(
                        tAny("export_dialog_sample_rate_autocorrected").replace(
                            "{rate}",
                            String(corrected),
                        ),
                    );
                } else {
                    setSampleRate(String(defaultRate));
                    setSampleRateNotice("");
                }
            } catch {
                // 保持回退默认值。
            }
        }

        void loadDefaults();
        return () => {
            disposed = true;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 与上方打开时重置同理，仅在打开时加载一次持久化设置
    }, [open]);

    useEffect(() => {
        if (!open) return;
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen(
                    "export_audio_progress",
                    (event: {
                        payload?: {
                            active?: boolean;
                            mode?: "project" | "separated";
                            progress?: number | null;
                            current?: number | null;
                            total?: number | null;
                        };
                    }) => {
                        if (disposed) return;
                        const payload = (event?.payload ?? {}) as {
                            active?: boolean;
                            mode?: "project" | "separated";
                            progress?: number | null;
                            current?: number | null;
                            total?: number | null;
                        };

                        const progressValue =
                            typeof payload.progress === "number" &&
                            Number.isFinite(payload.progress)
                                ? Math.max(0, Math.min(1, payload.progress))
                                : null;

                        setExportProgress({
                            active: Boolean(payload.active),
                            mode:
                                payload.mode === "project" || payload.mode === "separated"
                                    ? payload.mode
                                    : null,
                            progress: progressValue,
                            current:
                                typeof payload.current === "number" &&
                                Number.isFinite(payload.current)
                                    ? Math.max(0, Math.floor(payload.current))
                                    : null,
                            total:
                                typeof payload.total === "number" && Number.isFinite(payload.total)
                                    ? Math.max(0, Math.floor(payload.total))
                                    : null,
                        });
                    },
                );
            } catch {
                // 非 Tauri 环境下忽略。
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [open]);

    useEffect(() => {
        if (!open) return;
        if (!submitting && !exportProgress.active) {
            if (!keepProgressVisible) {
                setDisplayProgress(0);
            }
            return;
        }

        const timer = window.setInterval(() => {
            setDisplayProgress((prev) => {
                if (awaitingConflictDecision && !exportProgress.active) {
                    return prev;
                }
                if (
                    typeof exportProgress.progress === "number" &&
                    Number.isFinite(exportProgress.progress)
                ) {
                    const target = Math.round(
                        Math.max(0, Math.min(1, exportProgress.progress)) * 100,
                    );
                    if (target > prev) return Math.min(target, prev + 6);
                    if (target < prev) return prev;
                    if (target >= 100) return 100;
                }
                return Math.min(95, prev + 2);
            });
        }, 180);

        return () => {
            window.clearInterval(timer);
        };
    }, [
        open,
        submitting,
        exportProgress.active,
        exportProgress.progress,
        keepProgressVisible,
        awaitingConflictDecision,
    ]);

    /** 格式切换：文件名/命名模板扩展名原地替换 + 采样率合法性自动纠正。 */
    function handleFormatChange(next: ExportFormat) {
        if (next === format) return;
        setFormat(next);
        setProjectFileName((name) => applyExtensionToFileName(name, next));
        setSeparatedNamePattern((pattern) => applyExtensionToFileName(pattern, next));

        const corrected = nearestAllowedSampleRate(next, Number(sampleRate));
        if (corrected != null) {
            setSampleRate(String(corrected));
            setSampleRateNotice(
                tAny("export_dialog_sample_rate_autocorrected").replace(
                    "{rate}",
                    String(corrected),
                ),
            );
        } else {
            setSampleRateNotice("");
        }
    }

    /** 组装完整编码参数包（与后端 crate::encode::OutputSpec 对应）。 */
    function buildEncoderSpec(): ExportEncoderSpec {
        return {
            format,
            channelMode,
            dither,
            wav: { bitDepth: wavBitDepth },
            mp3: {
                mode:
                    mp3Mode === "cbr"
                        ? { mode: "cbr", bitrateKbps: mp3Bitrate }
                        : { mode: "vbr", qualityIndex: mp3Quality },
                tags: {
                    title: mp3Tags.title?.trim() ? mp3Tags.title : null,
                    artist: mp3Tags.artist?.trim() ? mp3Tags.artist : null,
                    album: mp3Tags.album?.trim() ? mp3Tags.album : null,
                    comment: mp3Tags.comment?.trim() ? mp3Tags.comment : null,
                },
            },
            flac: { bitDepth: flacBitDepth, compressionLevel: flacLevel },
        };
    }

    function toggleTarget(targetId: string) {
        setSelectedTargetIds((prev) => {
            if (prev.includes(targetId)) {
                return prev.filter((id) => id !== targetId);
            }
            return [...prev, targetId];
        });
    }

    function selectAllTargets() {
        setSelectedTargetIds(allTargets.map((target) => target.id));
    }

    function selectExcludeMutedTargets() {
        setSelectedTargetIds(
            allTargets.filter((target) => !target.excludedByRule).map((target) => target.id),
        );
    }

    function clearSelectedTargets() {
        setSelectedTargetIds([]);
    }

    function selectAllSubTargets() {
        setSelectedTargetIds((prev) => {
            const next = new Set(prev);
            for (const target of allTargets) {
                if (target.kind !== "sub") continue;
                if (next.has(target.id)) continue;
                if (target.excludedByRule) continue;
                next.add(target.id);
            }
            return Array.from(next);
        });
    }

    async function browseProjectOutputDir() {
        const picked = await fileBrowserApi.pickDirectory();
        if (picked.ok && !picked.canceled && picked.path) {
            setProjectOutputDir(picked.path.replace(/%/g, "%%"));
        }
    }

    async function browseSeparatedOutputDir() {
        const picked = await fileBrowserApi.pickDirectory();
        if (picked.ok && !picked.canceled && picked.path) {
            setSeparatedOutputDir(picked.path.replace(/%/g, "%%"));
        }
    }

    function applyTokenToActiveInput(token: string) {
        if (!activeInputKey || !activeInputRef.current) return;
        const input = activeInputRef.current;
        const start = input.selectionStart ?? input.value.length;
        const end = input.selectionEnd ?? input.value.length;
        const nextValue = `${input.value.slice(0, start)}${token}${input.value.slice(end)}`;

        const focusAndRestore = () => {
            input.focus();
            const pos = start + token.length;
            input.setSelectionRange(pos, pos);
        };

        switch (activeInputKey) {
            case "projectOutputDir":
                setProjectOutputDir(nextValue);
                break;
            case "projectFileName":
                setProjectFileName(nextValue);
                break;
            case "separatedOutputDir":
                setSeparatedOutputDir(nextValue);
                break;
            case "separatedNamePattern":
                setSeparatedNamePattern(nextValue);
                break;
        }
        window.requestAnimationFrame(focusAndRestore);
    }

    async function askConflict(path: string, kind: "exists" | "source-path") {
        return new Promise<{ choice: "overwrite" | "skip" | "cancel"; applyAll: boolean }>(
            (resolve) => {
                setAwaitingConflictDecision(true);
                conflictResolverRef.current = (value) => {
                    setAwaitingConflictDecision(false);
                    resolve(value);
                };
                setConflictDialog({ open: true, path, applyAll: false, kind });
            },
        );
    }

    async function resolveExportConflicts(request: AdvancedExportRequest) {
        const plan = await coreApi.previewExportAudioPlan(request);
        if (!plan?.ok || !Array.isArray(plan.targets)) {
            return {
                overwriteExistingPaths: [] as string[],
                skipExistingPaths: [] as string[],
                canceled: false,
            };
        }

        const existingKeys = new Set(
            (Array.isArray(plan.existingPaths) ? plan.existingPaths : []).map((path) =>
                normalizePathKey(path),
            ),
        );

        const targetPaths = Array.from(
            new Set((plan.targets ?? []).map((target) => target.path).filter(Boolean)),
        );

        const overwriteExistingPaths: string[] = [];
        const skipExistingPaths: string[] = [];
        let applyAllChoice: "overwrite" | "skip" | null = null;

        for (const path of targetPaths) {
            const pathKey = normalizePathKey(path);
            const isExisting = existingKeys.has(pathKey);
            const isSourceClipPath = sourceClipPathKeys.has(pathKey);
            if (!isExisting && !isSourceClipPath) continue;

            if (applyAllChoice === "overwrite") {
                overwriteExistingPaths.push(path);
                continue;
            }
            if (applyAllChoice === "skip") {
                skipExistingPaths.push(path);
                continue;
            }

            const result = await askConflict(path, isSourceClipPath ? "source-path" : "exists");
            if (result.choice === "cancel") {
                return { overwriteExistingPaths: [], skipExistingPaths: [], canceled: true };
            }
            if (result.choice === "overwrite") {
                overwriteExistingPaths.push(path);
                if (result.applyAll) applyAllChoice = "overwrite";
            } else {
                skipExistingPaths.push(path);
                if (result.applyAll) applyAllChoice = "skip";
            }
        }

        return { overwriteExistingPaths, skipExistingPaths, canceled: false };
    }

    function parseCustomRange(): { startSec: number; endSec: number } | null {
        const startSec = Number(customStartSec);
        const endSec = Number(customEndSec);
        if (!Number.isFinite(startSec) || !Number.isFinite(endSec)) {
            setErrorText(tAny("export_dialog_error_invalid_range"));
            return null;
        }
        if (endSec <= startSec) {
            setErrorText(tAny("export_dialog_error_invalid_range"));
            return null;
        }
        return {
            startSec: Math.max(0, startSec),
            endSec: Math.max(0, endSec),
        };
    }

    function parseSampleRate(): number | null {
        const value = Number(sampleRate);
        if (!Number.isFinite(value) || value <= 0) {
            setErrorText(tAny("export_dialog_error_invalid_sample_rate"));
            return null;
        }
        return Math.round(value);
    }

    function mapExportError(error: unknown): string {
        const code = String(error ?? "").trim();
        if (code === "export_cancelled") {
            return "";
        }
        if (code === "export_invalid_time_format") {
            return tAny("export_dialog_error_invalid_time_format");
        }
        if (code === "mp3_unsupported_sample_rate") {
            return tAny("export_dialog_error_mp3_unsupported_sample_rate");
        }
        return code || tAny("status_export_failed");
    }

    async function handleCancel() {
        if (submitting || exportProgress.active) {
            try {
                await coreApi.cancelExportAudio();
            } catch {
                // ignore cancellation command failures
            }
        }
        onOpenChange(false);
    }

    async function submitExport() {
        setErrorText("");
        setSubmitting(true);
        setKeepProgressVisible(false);
        // 如果上一次导出已达到 100%，需要将显示进度重置为较低值
        // 否则保持当前进度（不降级），并至少从 2% 开始缓升。
        setDisplayProgress((prev) => (prev >= 100 ? 2 : Math.max(prev, 2)));

        const range =
            rangeKind === "all"
                ? { kind: "all" as const }
                : (() => {
                      const custom = parseCustomRange();
                      if (!custom) return null;
                      return {
                          kind: "custom" as const,
                          startSec: custom.startSec,
                          endSec: custom.endSec,
                      };
                  })();

        if (!range) {
            setSubmitting(false);
            return;
        }

        const resolvedSampleRate = parseSampleRate();
        if (!resolvedSampleRate) {
            setSubmitting(false);
            return;
        }

        if (mode === "project") {
            const outputDir = projectOutputDir.trim();
            const fileName = projectFileName.trim();
            if (!outputDir) {
                setErrorText(tAny("export_dialog_error_missing_project_output_dir"));
                setSubmitting(false);
                return;
            }
            if (!fileName) {
                setErrorText(tAny("export_dialog_error_missing_project_file_name"));
                setSubmitting(false);
                return;
            }

            try {
                const conflicts = await resolveExportConflicts({
                    mode: "project",
                    range,
                    projectOutputDir: outputDir,
                    projectFileName: fileName,
                    sampleRate: resolvedSampleRate,
                    format,
                    encoder: buildEncoderSpec(),
                });
                if (conflicts.canceled) {
                    setSubmitting(false);
                    return;
                }
                const result = await dispatch(
                    exportAudioAdvanced({
                        mode: "project",
                        range,
                        projectOutputDir: outputDir,
                        projectFileName: fileName,
                        sampleRate: resolvedSampleRate,
                        format,
                        encoder: buildEncoderSpec(),
                        overwriteExistingPaths: conflicts.overwriteExistingPaths,
                        skipExistingPaths: conflicts.skipExistingPaths,
                    }),
                ).unwrap();
                if (!result?.ok) {
                    if (result?.cancelled || result?.error === "export_cancelled") {
                        setSubmitting(false);
                        return;
                    }
                    setErrorText(mapExportError(result?.error));
                    setSubmitting(false);
                    return;
                }
                setDisplayProgress(100);
                setKeepProgressVisible(true);
            } finally {
                setSubmitting(false);
            }
            return;
        }

        const outputDir = separatedOutputDir.trim();
        if (!outputDir) {
            setErrorText(tAny("export_dialog_error_missing_output_dir"));
            setSubmitting(false);
            return;
        }

        const selectedTargets = allTargets
            .filter((target) => selectedTargetIds.includes(target.id))
            .map((target) => ({
                kind: target.kind,
                trackId: target.trackId,
            }));

        if (selectedTargets.length === 0) {
            setErrorText(tAny("export_dialog_error_missing_targets"));
            setSubmitting(false);
            return;
        }

        try {
            const conflicts = await resolveExportConflicts({
                mode: "separated",
                range,
                separatedOutputDir: outputDir,
                separatedNamePattern:
                    separatedNamePattern.trim() || "<ExportIndex>_<TrackName>.wav",
                separatedTargets: selectedTargets,
                sampleRate: resolvedSampleRate,
                format,
                encoder: buildEncoderSpec(),
            });
            if (conflicts.canceled) {
                setSubmitting(false);
                return;
            }

            const result = await dispatch(
                exportAudioAdvanced({
                    mode: "separated",
                    range,
                    separatedOutputDir: outputDir,
                    separatedNamePattern:
                        separatedNamePattern.trim() || "<ExportIndex>_<TrackName>.wav",
                    separatedTargets: selectedTargets,
                    sampleRate: resolvedSampleRate,
                    format,
                    encoder: buildEncoderSpec(),
                    overwriteExistingPaths: conflicts.overwriteExistingPaths,
                    skipExistingPaths: conflicts.skipExistingPaths,
                }),
            ).unwrap();

            if (!result?.ok) {
                if (result?.cancelled || result?.error === "export_cancelled") {
                    setSubmitting(false);
                    return;
                }
                setErrorText(mapExportError(result?.error));
                setSubmitting(false);
                return;
            }
            setDisplayProgress(100);
            setKeepProgressVisible(true);
        } finally {
            setSubmitting(false);
        }
    }

    const exportCompleted =
        keepProgressVisible && !submitting && !exportProgress.active && displayProgress >= 100;

    const progressLabel = exportCompleted
        ? mode === "separated"
            ? tAny("status_export_separated_done")
            : tAny("status_export_done")
        : mode === "separated"
          ? (() => {
                const current = exportProgress.current;
                const total = exportProgress.total;
                if (current != null && total != null && total > 0) {
                    return `${tAny("export_dialog_progress")}${" "}${current}/${total}`;
                }
                return tAny("export_dialog_progress");
            })()
          : tAny("export_dialog_progress");

    const shouldShowProgress = submitting || exportProgress.active || keepProgressVisible;

    return (
        <>
            <Dialog.Root open={open} onOpenChange={onOpenChange}>
                <Dialog.Content
                    style={{ maxWidth: 760 }}
                    onKeyDown={(event) => event.stopPropagation()}
                >
                    <Dialog.Title>{tAny("menu_export_audio")}</Dialog.Title>
                    <Dialog.Description>{tAny("export_dialog_desc")}</Dialog.Description>

                    <Flex direction="column" gap="3" mt="3">
                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 132 }}>
                                {tAny("export_dialog_mode")}
                            </Text>
                            <Select.Root
                                value={mode}
                                onValueChange={(value) => setMode(value as ExportMode)}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: mode,
                                            options: ["project", "separated"],
                                            onChange: (next) => setMode(next as ExportMode),
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="project">
                                        {tAny("export_dialog_mode_project")}
                                    </Select.Item>
                                    <Select.Item value="separated">
                                        {tAny("export_dialog_mode_separated")}
                                    </Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </Flex>

                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 132 }}>
                                {tAny("export_dialog_range")}
                            </Text>
                            <Select.Root
                                value={rangeKind}
                                onValueChange={(value) => setRangeKind(value as ExportRangeKind)}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: rangeKind,
                                            options: ["all", "custom"],
                                            onChange: (next) =>
                                                setRangeKind(next as ExportRangeKind),
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="all">
                                        {tAny("export_dialog_range_all")}
                                    </Select.Item>
                                    <Select.Item value="custom">
                                        {tAny("export_dialog_range_custom")}
                                    </Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </Flex>

                        {rangeKind === "custom" && (
                            <Flex gap="2" align="center">
                                <Text size="2" style={{ minWidth: 132 }}>
                                    {tAny("export_dialog_range_custom_label")}
                                </Text>
                                <TextField.Root
                                    size="2"
                                    type="number"
                                    min={0}
                                    step="0.001"
                                    value={customStartSec}
                                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                        setCustomStartSec(event.target.value)
                                    }
                                    style={{ width: 160 }}
                                />
                                <Text size="2" color="gray">
                                    ~
                                </Text>
                                <TextField.Root
                                    size="2"
                                    type="number"
                                    min={0}
                                    step="0.001"
                                    value={customEndSec}
                                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                        setCustomEndSec(event.target.value)
                                    }
                                    style={{ width: 160 }}
                                />
                                <Text size="1" color="gray">
                                    sec
                                </Text>
                            </Flex>
                        )}

                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 132 }}>
                                {tAny("export_dialog_format")}
                            </Text>
                            <SegmentedControl.Root
                                value={format}
                                onValueChange={(value) => handleFormatChange(value as ExportFormat)}
                            >
                                <SegmentedControl.Item value="wav">WAV</SegmentedControl.Item>
                                <SegmentedControl.Item value="mp3">MP3</SegmentedControl.Item>
                                <SegmentedControl.Item value="flac">FLAC</SegmentedControl.Item>
                            </SegmentedControl.Root>
                        </Flex>

                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 132 }}>
                                {tAny("export_dialog_sample_rate")}
                            </Text>
                            <Select.Root
                                value={sampleRate}
                                onValueChange={(value) => {
                                    setSampleRate(value);
                                    setSampleRateNotice("");
                                }}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: sampleRate,
                                            options: sampleRateOptions(format).map(String),
                                            onChange: (next) => {
                                                setSampleRate(next);
                                                setSampleRateNotice("");
                                            },
                                        });
                                    }}
                                />
                                <Select.Content>
                                    {sampleRateOptions(format).map((rate) => (
                                        <Select.Item key={rate} value={String(rate)}>
                                            {rate} Hz
                                        </Select.Item>
                                    ))}
                                </Select.Content>
                            </Select.Root>
                        </Flex>

                        {sampleRateNotice ? (
                            <Text size="1" color="amber">
                                {sampleRateNotice}
                            </Text>
                        ) : null}

                        {format !== "mp3" ? (
                            <Flex align="center" gap="2">
                                <Text size="2" style={{ minWidth: 132 }}>
                                    {tAny("export_dialog_bit_depth")}
                                </Text>
                                <Select.Root
                                    value={format === "wav" ? wavBitDepth : flacBitDepth}
                                    onValueChange={(value) => {
                                        if (format === "wav") {
                                            if (
                                                value === "i16" ||
                                                value === "i24" ||
                                                value === "f32"
                                            ) {
                                                setWavBitDepth(value);
                                            }
                                        } else if (value === "i16" || value === "i24") {
                                            setFlacBitDepth(value);
                                        }
                                    }}
                                >
                                    <Select.Trigger
                                        style={{ flex: 1 }}
                                        onWheel={(event) => {
                                            const isWav = format === "wav";
                                            applySelectWheelChange({
                                                event,
                                                currentValue: isWav ? wavBitDepth : flacBitDepth,
                                                options: isWav ? ["i16", "i24", "f32"] : ["i16", "i24"],
                                                onChange: (next) => {
                                                    if (isWav) {
                                                        if (
                                                            next === "i16" ||
                                                            next === "i24" ||
                                                            next === "f32"
                                                        ) {
                                                            setWavBitDepth(next);
                                                        }
                                                    } else if (next === "i16" || next === "i24") {
                                                        setFlacBitDepth(next);
                                                    }
                                                },
                                            });
                                        }}
                                    />
                                    <Select.Content>
                                        <Select.Item value="i16">16-bit</Select.Item>
                                        <Select.Item value="i24">24-bit</Select.Item>
                                        {format === "wav" && (
                                            <Select.Item value="f32">32-bit float</Select.Item>
                                        )}
                                    </Select.Content>
                                </Select.Root>
                            </Flex>
                        ) : (
                            <Text size="1" color="gray">
                                {tAny("export_dialog_mp3_bit_depth_note")}
                            </Text>
                        )}

                        <Flex align="center" gap="2">
                            <Button
                                variant="ghost"
                                color="gray"
                                size="1"
                                onClick={() => setEncoderOpen((prev) => !prev)}
                            >
                                {encoderOpen ? "▾" : "▸"} {tAny("export_dialog_encoder_params")}
                            </Button>
                        </Flex>

                        {encoderOpen && (
                            <Flex direction="column" gap="3" pl="1">
                                    {format === "mp3" && (
                                        <>
                                            <Flex align="center" gap="2">
                                                <Text size="2" style={{ minWidth: 132 }}>
                                                    {tAny("export_dialog_mp3_mode")}
                                                </Text>
                                                <Select.Root
                                                    value={mp3Mode}
                                                    onValueChange={(value) => {
                                                        if (value === "cbr" || value === "vbr") {
                                                            setMp3Mode(value);
                                                        }
                                                    }}
                                                >
                                                    <Select.Trigger
                                                        style={{ flex: 1 }}
                                                        onWheel={(event) => {
                                                            applySelectWheelChange({
                                                                event,
                                                                currentValue: mp3Mode,
                                                                options: ["vbr", "cbr"],
                                                                onChange: (next) => {
                                                                    if (next === "cbr" || next === "vbr") {
                                                                        setMp3Mode(next);
                                                                    }
                                                                },
                                                            });
                                                        }}
                                                    />
                                                    <Select.Content>
                                                        <Select.Item value="vbr">
                                                            {tAny("export_dialog_mp3_mode_vbr")}
                                                        </Select.Item>
                                                        <Select.Item value="cbr">
                                                            {tAny("export_dialog_mp3_mode_cbr")}
                                                        </Select.Item>
                                                    </Select.Content>
                                                </Select.Root>
                                            </Flex>

                                            {mp3Mode === "cbr" ? (
                                                <Flex align="center" gap="2">
                                                    <Text size="2" style={{ minWidth: 132 }}>
                                                        {tAny("export_dialog_mp3_bitrate")}
                                                    </Text>
                                                    <Select.Root
                                                        value={String(mp3Bitrate)}
                                                        onValueChange={(value) =>
                                                            setMp3Bitrate(Number(value))
                                                        }
                                                    >
                                                        <Select.Trigger
                                                            style={{ flex: 1 }}
                                                            onWheel={(event) => {
                                                                applySelectWheelChange({
                                                                    event,
                                                                    currentValue: String(mp3Bitrate),
                                                                    options: MP3_BITRATES.map(String),
                                                                    onChange: (next) =>
                                                                        setMp3Bitrate(Number(next)),
                                                                });
                                                            }}
                                                        />
                                                        <Select.Content>
                                                            {MP3_BITRATES.map((rate) => (
                                                                <Select.Item
                                                                    key={rate}
                                                                    value={String(rate)}
                                                                >
                                                                    {rate} kbps
                                                                </Select.Item>
                                                            ))}
                                                        </Select.Content>
                                                    </Select.Root>
                                                </Flex>
                                            ) : (
                                                <Flex align="center" gap="2">
                                                    <Text size="2" style={{ minWidth: 132 }}>
                                                        {tAny("export_dialog_mp3_quality")}
                                                    </Text>
                                                    <Select.Root
                                                        value={String(mp3Quality)}
                                                        onValueChange={(value) =>
                                                            setMp3Quality(Number(value))
                                                        }
                                                    >
                                                        <Select.Trigger
                                                            style={{ flex: 1 }}
                                                            onWheel={(event) => {
                                                                applySelectWheelChange({
                                                                    event,
                                                                    currentValue: String(mp3Quality),
                                                                    options: MP3_VBR_AVG_KBPS.map(
                                                                        (_, index) => String(index),
                                                                    ),
                                                                    onChange: (next) =>
                                                                        setMp3Quality(Number(next)),
                                                                });
                                                            }}
                                                        />
                                                        <Select.Content>
                                                            {MP3_VBR_AVG_KBPS.map((avg, index) => (
                                                                <Select.Item
                                                                    key={index}
                                                                    value={String(index)}
                                                                >
                                                                    q{index} · ~{avg} kbps
                                                                </Select.Item>
                                                            ))}
                                                        </Select.Content>
                                                    </Select.Root>
                                                </Flex>
                                            )}

                                            <Flex direction="column" gap="2">
                                                <Text size="2" color="gray">
                                                    {tAny("export_dialog_mp3_tags")}
                                                </Text>
                                                <div className="grid grid-cols-2 gap-2">
                                                    {(
                                                        [
                                                            ["title", "export_dialog_tag_title"],
                                                            ["artist", "export_dialog_tag_artist"],
                                                            ["album", "export_dialog_tag_album"],
                                                            ["comment", "export_dialog_tag_comment"],
                                                        ] as const
                                                    ).map(([key, i18nKey]) => (
                                                        <label
                                                            key={key}
                                                            className="flex flex-col gap-1 text-xs text-qt-text"
                                                        >
                                                            <Text size="1" color="gray">
                                                                {tAny(i18nKey)}
                                                            </Text>
                                                            <TextField.Root
                                                                size="1"
                                                                value={mp3Tags[key] ?? ""}
                                                                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                                                    setMp3Tags((prev) => ({
                                                                        ...prev,
                                                                        [key]: event.target.value,
                                                                    }))
                                                                }
                                                            />
                                                        </label>
                                                    ))}
                                                </div>
                                            </Flex>
                                        </>
                                    )}

                                    {format === "flac" && (
                                        <>
                                            <Flex align="center" gap="2">
                                                <Text size="2" style={{ minWidth: 132 }}>
                                                    {tAny("export_dialog_flac_level")}
                                                </Text>
                                                <Slider
                                                    min={FLAC_COMPRESSION_RANGE.min}
                                                    max={FLAC_COMPRESSION_RANGE.max}
                                                    step={1}
                                                    value={[flacLevel]}
                                                    onValueChange={(value) =>
                                                        setFlacLevel(
                                                            Array.isArray(value) ? value[0] : value,
                                                        )
                                                    }
                                                    onWheel={(event) => {
                                                        applySelectWheelChange({
                                                            event,
                                                            currentValue: String(flacLevel),
                                                            options: FLAC_LEVEL_OPTIONS,
                                                            onChange: (next) =>
                                                                setFlacLevel(Number(next)),
                                                        });
                                                    }}
                                                    style={{ flex: 1 }}
                                                />
                                                <Text
                                                    size="1"
                                                    color="gray"
                                                    style={{ minWidth: 24, textAlign: "right" }}
                                                >
                                                    {flacLevel}
                                                </Text>
                                            </Flex>
                                            <Text size="1" color="gray">
                                                {tAny("export_dialog_flac_level_hint")}
                                            </Text>
                                        </>
                                    )}

                                    {((format === "wav" &&
                                        (wavBitDepth === "i16" || wavBitDepth === "i24")) ||
                                        format === "flac") && (
                                        <Flex align="center" gap="2">
                                            <Text size="2" style={{ minWidth: 132 }}>
                                                {tAny("export_dialog_dither")}
                                            </Text>
                                            <Select.Root
                                                value={dither}
                                                onValueChange={(value) => {
                                                    if (value === "none" || value === "tpdf") {
                                                        setDither(value);
                                                    }
                                                }}
                                            >
                                                <Select.Trigger
                                                    style={{ flex: 1 }}
                                                    onWheel={(event) => {
                                                        applySelectWheelChange({
                                                            event,
                                                            currentValue: dither,
                                                            options: ["none", "tpdf"],
                                                            onChange: (next) => {
                                                                if (next === "none" || next === "tpdf") {
                                                                    setDither(next);
                                                                }
                                                            },
                                                        });
                                                    }}
                                                />
                                                <Select.Content>
                                                    <Select.Item value="none">
                                                        {tAny("export_dialog_dither_none")}
                                                    </Select.Item>
                                                    <Select.Item value="tpdf">
                                                        {tAny("export_dialog_dither_tpdf")}
                                                    </Select.Item>
                                                </Select.Content>
                                            </Select.Root>
                                        </Flex>
                                    )}

                                    <Flex align="center" gap="2">
                                        <Text size="2" style={{ minWidth: 132 }}>
                                            {tAny("export_dialog_channel_mode")}
                                        </Text>
                                        <Select.Root
                                            value={channelMode}
                                            onValueChange={(value) => {
                                                if (value === "stereo" || value === "mono") {
                                                    setChannelMode(value);
                                                }
                                            }}
                                        >
                                            <Select.Trigger
                                                style={{ flex: 1 }}
                                                onWheel={(event) => {
                                                    applySelectWheelChange({
                                                        event,
                                                        currentValue: channelMode,
                                                        options: ["stereo", "mono"],
                                                        onChange: (next) => {
                                                            if (next === "stereo" || next === "mono") {
                                                                setChannelMode(next);
                                                            }
                                                        },
                                                    });
                                                }}
                                            />
                                            <Select.Content>
                                                <Select.Item value="stereo">
                                                    {tAny("export_dialog_channel_stereo")}
                                                </Select.Item>
                                                <Select.Item value="mono">
                                                    {tAny("export_dialog_channel_mono")}
                                                </Select.Item>
                                            </Select.Content>
                                        </Select.Root>
                                    </Flex>
                                </Flex>
                            )}

                        {mode === "project" ? (
                            <>
                                <Flex align="center" gap="2">
                                    <Text size="2" style={{ minWidth: 132 }}>
                                        {tAny("export_dialog_output_dir")}
                                    </Text>
                                    <TextField.Root
                                        size="2"
                                        value={projectOutputDir}
                                        onFocus={(event) => {
                                            setActiveInputKey("projectOutputDir");
                                            activeInputRef.current =
                                                event.target as HTMLInputElement;
                                        }}
                                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                            setProjectOutputDir(event.target.value)
                                        }
                                        style={{ flex: 1 }}
                                    />
                                    <Button
                                        variant="soft"
                                        color="gray"
                                        onClick={() => void browseProjectOutputDir()}
                                    >
                                        {tAny("export_dialog_browse")}
                                    </Button>
                                </Flex>

                                <Flex align="center" gap="2">
                                    <Text size="2" style={{ minWidth: 132 }}>
                                        {tAny("export_dialog_project_file_name")}
                                    </Text>
                                    <TextField.Root
                                        size="2"
                                        value={projectFileName}
                                        onFocus={(event) => {
                                            setActiveInputKey("projectFileName");
                                            activeInputRef.current =
                                                event.target as HTMLInputElement;
                                        }}
                                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                            setProjectFileName(event.target.value)
                                        }
                                        style={{ flex: 1 }}
                                    />
                                </Flex>

                                <Flex gap="2" wrap="wrap" align="center">
                                    <Text size="1" color="gray">
                                        {tAny("export_pattern_placeholders")}
                                    </Text>
                                    {(["<ProjectName>", "<ProjectFolder>"] as const).map(
                                        (token) => (
                                            <Button
                                                key={token}
                                                size="1"
                                                variant="ghost"
                                                color="gray"
                                                onClick={() => applyTokenToActiveInput(token)}
                                            >
                                                {token}
                                            </Button>
                                        ),
                                    )}
                                </Flex>
                            </>
                        ) : (
                            <>
                                <Flex align="center" gap="2">
                                    <Text size="2" style={{ minWidth: 132 }}>
                                        {tAny("export_dialog_output_dir")}
                                    </Text>
                                    <TextField.Root
                                        size="2"
                                        value={separatedOutputDir}
                                        onFocus={(event) => {
                                            setActiveInputKey("separatedOutputDir");
                                            activeInputRef.current =
                                                event.target as HTMLInputElement;
                                        }}
                                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                            setSeparatedOutputDir(event.target.value)
                                        }
                                        style={{ flex: 1 }}
                                    />
                                    <Button
                                        variant="soft"
                                        color="gray"
                                        onClick={() => void browseSeparatedOutputDir()}
                                    >
                                        {tAny("export_dialog_browse")}
                                    </Button>
                                </Flex>

                                <Flex align="center" gap="2">
                                    <Text size="2" style={{ minWidth: 132 }}>
                                        {tAny("export_dialog_name_pattern")}
                                    </Text>
                                    <TextField.Root
                                        size="2"
                                        value={separatedNamePattern}
                                        onFocus={(event) => {
                                            setActiveInputKey("separatedNamePattern");
                                            activeInputRef.current =
                                                event.target as HTMLInputElement;
                                        }}
                                        onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                            setSeparatedNamePattern(event.target.value)
                                        }
                                        style={{ flex: 1 }}
                                    />
                                </Flex>

                                <Flex gap="2" wrap="wrap" align="center">
                                    <Text size="1" color="gray">
                                        {tAny("export_pattern_placeholders")}
                                    </Text>
                                    {[
                                        "<ExportIndex>",
                                        "<TrackIndex>",
                                        "<TrackName>",
                                        "<TrackType>",
                                        "<TrackId>",
                                        "<ProjectName>",
                                        "<ProjectFolder>",
                                    ].map((token) => (
                                        <Button
                                            key={token}
                                            size="1"
                                            variant="ghost"
                                            color="gray"
                                            onClick={() => applyTokenToActiveInput(token)}
                                        >
                                            {token}
                                        </Button>
                                    ))}
                                </Flex>

                                <div className="rounded border border-qt-border bg-qt-base p-2 max-h-[240px] overflow-y-auto">
                                    <Text size="2" className="font-medium">
                                        {tAny("export_dialog_targets")}
                                    </Text>
                                    <Flex gap="1" mt="2" wrap="wrap">
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            onClick={selectAllTargets}
                                        >
                                            {tAny("export_dialog_select_all")}
                                        </Button>
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            onClick={selectExcludeMutedTargets}
                                        >
                                            {tAny("export_dialog_select_exclude_muted")}
                                        </Button>
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            onClick={clearSelectedTargets}
                                        >
                                            {tAny("export_dialog_select_none")}
                                        </Button>
                                        <Button
                                            size="1"
                                            variant="soft"
                                            color="gray"
                                            onClick={selectAllSubTargets}
                                            disabled={
                                                !allTargets.some((target) => target.kind === "sub")
                                            }
                                        >
                                            {tAny("export_dialog_select_all_subtracks")}
                                        </Button>
                                    </Flex>
                                    <Flex direction="column" gap="2" mt="2">
                                        {targetGroups.map((group) => (
                                            <div
                                                key={group.id}
                                                className="rounded border border-qt-border bg-qt-window px-2 py-2"
                                            >
                                                <Text size="1" color="gray">
                                                    {group.title}
                                                </Text>
                                                <Flex direction="column" gap="1" mt="1">
                                                    {group.options.map((target) => (
                                                        <label
                                                            key={target.id}
                                                            className="flex items-center gap-2 text-xs text-qt-text cursor-pointer"
                                                        >
                                                            <input
                                                                type="checkbox"
                                                                checked={selectedTargetIds.includes(
                                                                    target.id,
                                                                )}
                                                                onChange={() =>
                                                                    toggleTarget(target.id)
                                                                }
                                                            />
                                                            <span>{target.label}</span>
                                                        </label>
                                                    ))}
                                                </Flex>
                                            </div>
                                        ))}
                                    </Flex>
                                </div>
                            </>
                        )}

                        {errorText ? (
                            <Text size="2" color="red">
                                {errorText}
                            </Text>
                        ) : null}

                        {shouldShowProgress ? (
                            <div className="rounded border border-qt-border bg-qt-window p-2">
                                <ProgressBar
                                    percentage={displayProgress}
                                    label={progressLabel}
                                    completed={exportCompleted}
                                />
                            </div>
                        ) : null}
                    </Flex>

                    <Flex justify="end" gap="2" mt="4">
                        <Button variant="soft" color="gray" onClick={() => void handleCancel()}>
                            {tAny("cancel")}
                        </Button>
                        <Button
                            onClick={() => {
                                void submitExport();
                            }}
                            disabled={session.busy || submitting}
                        >
                            {tAny("menu_export_audio")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
            <Dialog.Root
                open={conflictDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        const resolver = conflictResolverRef.current;
                        conflictResolverRef.current = null;
                        setConflictDialog((prev) => ({ ...prev, open: false }));
                        resolver?.({ choice: "cancel", applyAll: false });
                    }
                }}
            >
                <Dialog.Content style={{ maxWidth: 620 }}>
                    <Dialog.Title>
                        {conflictDialog.kind === "source-path"
                            ? tAny("export_conflict_source_title")
                            : tAny("export_conflict_exists_title")}
                    </Dialog.Title>
                    <Dialog.Description
                        style={{
                            userSelect: "text",
                            wordBreak: "break-all",
                            whiteSpace: "pre-wrap",
                        }}
                    >
                        {conflictDialog.kind === "source-path"
                            ? tAny("export_conflict_source_desc")
                            : tAny("export_conflict_exists_desc")}
                        {"\n"}
                        {conflictDialog.path}
                    </Dialog.Description>
                    <label className="flex items-center gap-2 text-sm mt-3">
                        <input
                            type="checkbox"
                            checked={conflictDialog.applyAll}
                            onChange={(event) =>
                                setConflictDialog((prev) => ({
                                    ...prev,
                                    applyAll: event.target.checked,
                                }))
                            }
                        />
                        <span>{tAny("export_conflict_apply_all")}</span>
                    </label>
                    <Flex justify="end" gap="2" mt="4">
                        <Button
                            variant={conflictDialog.kind === "source-path" ? "solid" : "soft"}
                            color={conflictDialog.kind === "source-path" ? "amber" : "gray"}
                            onClick={() => {
                                const resolver = conflictResolverRef.current;
                                conflictResolverRef.current = null;
                                setConflictDialog((prev) => ({ ...prev, open: false }));
                                resolver?.({ choice: "skip", applyAll: conflictDialog.applyAll });
                            }}
                        >
                            {tAny("export_conflict_skip")}
                        </Button>
                        <Button
                            variant="soft"
                            color="red"
                            onClick={() => {
                                const resolver = conflictResolverRef.current;
                                conflictResolverRef.current = null;
                                setConflictDialog((prev) => ({ ...prev, open: false }));
                                resolver?.({ choice: "cancel", applyAll: false });
                            }}
                        >
                            {tAny("export_conflict_cancel")}
                        </Button>
                        <Button
                            variant={conflictDialog.kind === "source-path" ? "soft" : "solid"}
                            color={conflictDialog.kind === "source-path" ? "gray" : "blue"}
                            onClick={() => {
                                const resolver = conflictResolverRef.current;
                                conflictResolverRef.current = null;
                                setConflictDialog((prev) => ({ ...prev, open: false }));
                                resolver?.({
                                    choice: "overwrite",
                                    applyAll: conflictDialog.applyAll,
                                });
                            }}
                        >
                            {tAny("export_conflict_overwrite")}
                        </Button>
                    </Flex>
                </Dialog.Content>
            </Dialog.Root>
        </>
    );
}
