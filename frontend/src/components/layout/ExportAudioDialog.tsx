/*
 * 导出音频配置对话框。
 * 负责收集导出模式、时间范围、输出路径与分轨命名/目标选择，并调用后端统一导出命令。
 */

import { useCallback, useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Flex, TextField } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { shallowEqual } from "react-redux";
import type { RootState } from "../../app/store";
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
import {
    DISPLAY_TICK_MS,
    isSameExportProgress,
    nextDisplayProgress,
    realProgressPercent,
} from "./exportProgressDisplay";
import type { TrackInfo } from "../../features/session/sessionTypes";
import {
    AppButton,
    AppNumberField,
    AppSegmentedControl,
    AppSelect,
    AppSlider,
    AppSliderReadout,
} from "../../ui";
import { AppDialog } from "../../ui/Dialog";
import { AppField, AppForm, AppSwitchRow } from "../../ui/Field";

interface ExportAudioDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

type ExportMode = "project" | "separated";
type ExportRangeKind = "all" | "custom";
type Mp3ModeKind = "cbr" | "vbr";

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

/** 取文件路径的父目录（兼容 Windows `\` 与 POSIX `/`）；无目录分隔符时返回空串。 */
function parentDirOfPath(filePath: string): string {
    const trimmed = filePath.replace(/[\\/]+$/, "");
    const idx = Math.max(trimmed.lastIndexOf("/"), trimmed.lastIndexOf("\\"));
    return idx > 0 ? trimmed.slice(0, idx) : "";
}

/** 从导出结果收集实际写出的产物路径：分轨模式为多个文件，工程模式为单个文件。 */
function collectExportPaths(result: {
    path?: string;
    tracks?: Array<{ path?: string; ok?: boolean }>;
}): string[] {
    const paths: string[] = [];
    if (Array.isArray(result.tracks)) {
        for (const track of result.tracks) {
            // 跳过被跳过 / 写入失败的目标（ok:false），它们没有可用产物。
            if (track?.ok === false) continue;
            if (typeof track?.path === "string" && track.path) paths.push(track.path);
        }
    }
    if (typeof result.path === "string" && result.path && !paths.includes(result.path)) {
        paths.push(result.path);
    }
    return paths;
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

/** ExportAudioDialog 实际消费的 session 字段子集（配合 shallowEqual 阻断播放轮询的重渲染）。
 *  新增消费字段时必须同步补充到这里。 */
const selectExportDialogSession = (state: RootState) => {
    const session = state.session;
    return {
        busy: session.busy,
        clips: session.clips,
        projectSec: session.projectSec,
        tracks: session.tracks,
    };
};

export function ExportAudioDialog({ open, onOpenChange }: ExportAudioDialogProps) {
    const { tf } = useI18n();
    const dispatch = useAppDispatch();
    // 只选取本组件实际消费的字段子集并以 shallowEqual 比较：播放期间 playheadSec
    // 每 ~33ms 变一次、session 对象引用随之失效，直接订阅 state.session 会让整个
    // 导出对话框（常驻挂载、关闭时不卸载）以 ≥30Hz 空转重渲。
    const session = useAppSelector(selectExportDialogSession, shallowEqual);

    const [mode, setMode] = useState<ExportMode>("project");
    const [rangeKind, setRangeKind] = useState<ExportRangeKind>("all");
    const [customStartSec, setCustomStartSec] = useState(0);
    const [customEndSec, setCustomEndSec] = useState(0);
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
    /**
     * 后端最近一次真实进度（0..100）；`null` = 本轮尚未收到任何真实进度。
     *
     * 【为什么用 ref 而不是 state】它是显示推进的**目标值**，由高频事件（约 50ms
     * 一次）写入。放进 state 会让每次事件都重渲染整个对话框（约 2000 行）。显示值
     * 本身仍是 state，由固定节拍的计时器按目标插值推进。
     */
    const targetProgressRef = useRef<number | null>(null);
    /**
     * 现场诊断开关：`localStorage.hifiDebugExportProgress = "1"` 时把每个进度事件
     * 打到控制台。用于一次性区分"后端事件没到达"与"前端显示没推进"。
     */
    const debugExportProgress = useMemo(() => {
        try {
            return window.localStorage.getItem("hifiDebugExportProgress") === "1";
        } catch {
            return false;
        }
    }, []);
    const [keepProgressVisible, setKeepProgressVisible] = useState(false);
    const [lastOutputDir, setLastOutputDir] = useState("");
    // 上次成功导出产生的文件路径（分轨为多个）；用于「打开文件夹」时一并选中。
    const [lastOutputPaths, setLastOutputPaths] = useState<string[]>([]);
    const [examplePath, setExamplePath] = useState("");
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
                tf("export_track_label_root_suffix"),
                tf("export_track_label_sub_suffix"),
            ),
        [session.tracks, session.clips, tf],
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
        setCustomStartSec(0);
        setCustomEndSec(Math.max(0, Math.ceil(projectSecAtOpenRef.current)));
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
        targetProgressRef.current = null;
        setKeepProgressVisible(false);
        setAwaitingConflictDecision(false);
        setLastOutputDir("");
        setLastOutputPaths([]);
        setExamplePath("");

        const defaultSelected = targetGroups.flatMap((group) => {
            return group.options
                .filter((option) => option.kind === "root" && !option.excludedByRule)
                .map((option) => option.id);
        });
        setSelectedTargetIds(defaultSelected);
        setErrorText("");
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅在打开时重置一次表单
    }, [open]);

    // 表单最新值（latest-ref）：loadDefaults 的 IPC 迟到时用来判断用户
    // 是否已动过表单（闭包里的 state 是打开那帧的旧值，不能用于该判断）。
    const formLatestRef = useRef<{
        format: ExportFormat;
        channelMode: "mono" | "stereo";
        dither: "tpdf" | "none";
        wavBitDepth: "i16" | "i24" | "f32";
        flacBitDepth: "i16" | "i24";
        flacLevel: number;
        mp3Mode: "cbr" | "vbr";
        mp3Bitrate: number;
        mp3Quality: number;
        mp3Tags: Mp3Tags;
        projectOutputDir: string;
        projectFileName: string;
        separatedOutputDir: string;
        separatedNamePattern: string;
    }>({
        format: "wav",
        channelMode: "stereo",
        dither: "none",
        wavBitDepth: "f32",
        flacBitDepth: "i24",
        flacLevel: FLAC_COMPRESSION_RANGE.default,
        mp3Mode: "vbr",
        mp3Bitrate: 320,
        mp3Quality: 2,
        mp3Tags: {},
        projectOutputDir: "",
        projectFileName: "<ProjectName>.wav",
        separatedOutputDir: "",
        separatedNamePattern: "<ExportIndex>_<TrackName>.wav",
    });
    useEffect(() => {
        formLatestRef.current = {
            format,
            channelMode,
            dither,
            wavBitDepth,
            flacBitDepth,
            flacLevel,
            mp3Mode,
            mp3Bitrate,
            mp3Quality,
            mp3Tags,
            projectOutputDir,
            projectFileName,
            separatedOutputDir,
            separatedNamePattern,
        };
    });

    useEffect(() => {
        if (!open) return;
        let disposed = false;

        async function loadDefaults() {
            try {
                const defaults = await coreApi.getExportAudioDefaults();
                if (disposed || !defaults?.ok) return;
                // IPC 往返期间用户已动过表单（任何将被覆盖的字段离开了重置
                // 回退值）→ 保留用户输入，不再用持久化设置覆盖。
                const cur = formLatestRef.current;
                const userTouched =
                    cur.format !== "wav" ||
                    cur.channelMode !== "stereo" ||
                    cur.dither !== "none" ||
                    cur.wavBitDepth !== "f32" ||
                    cur.flacBitDepth !== "i24" ||
                    cur.flacLevel !== FLAC_COMPRESSION_RANGE.default ||
                    cur.mp3Mode !== "vbr" ||
                    cur.mp3Bitrate !== 320 ||
                    cur.mp3Quality !== 2 ||
                    Object.keys(cur.mp3Tags ?? {}).length > 0 ||
                    cur.projectOutputDir !== "" ||
                    cur.projectFileName !== "<ProjectName>.wav" ||
                    cur.separatedOutputDir !== "" ||
                    cur.separatedNamePattern !== "<ExportIndex>_<TrackName>.wav";
                if (userTouched) return;
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
                    // 手改配置可能出现表外码率：钳制到合法档位，否则下拉框
                    // 渲染出空触发器（后端会再吸附，这里保证 UI 一致）。
                    setMp3Bitrate(
                        MP3_BITRATES.includes(
                            persistedMode.bitrateKbps as (typeof MP3_BITRATES)[number],
                        )
                            ? persistedMode.bitrateKbps
                            : 320,
                    );
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
                        tf("export_dialog_sample_rate_autocorrected").replace(
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

                        // 目标值写 ref（不触发渲染），并把显示值**立即单调推进**：
                        // 即便推进计时器因任何原因停摆，进度条也始终反映真实进度；
                        // 计时器只负责两次事件之间的插值（见 exportProgressDisplay）。
                        const percent = realProgressPercent(progressValue);
                        if (percent !== null) {
                            targetProgressRef.current = percent;
                            setDisplayProgress((prev) => Math.max(prev, percent));
                        }

                        const nextProgress = {
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
                        };

                        if (debugExportProgress) {
                            console.debug("[export-progress] event", {
                                ...nextProgress,
                                atMs: Math.round(performance.now()),
                            });
                        }

                        // 字段未变时复用旧对象，避免高频事件触发无谓的整框重渲染。
                        setExportProgress((prev) =>
                            isSameExportProgress(prev, nextProgress) ? prev : nextProgress,
                        );
                    },
                );
                /*
                 * `listen` 是异步的：cleanup 可能在它 resolve 之前就跑过（对话框在
                 * 动态 import + 注册期间关闭 / effect 重跑）。那时 `unlisten` 还是
                 * null，cleanup 无从注销，注册会跨开合周期累积泄漏。resolve 后补一次
                 * 检查，已 disposed 就当场注销。
                 */
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // 非 Tauri 环境下忽略。
            }
        }

        void setup();
        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
        // `debugExportProgress` 是挂载时求值一次的诊断开关（useMemo 依赖为空），
        // 稳定不变；列入依赖只为满足 exhaustive-deps，不会造成重复订阅。
    }, [open, debugExportProgress]);

    /**
     * 显示推进计时器。
     *
     * 【为什么依赖里**没有** `exportProgress.progress`】这是"进度条不更新"的根因所在：
     * 后端节流后约 50ms 一个事件，若进度值在依赖里，每个事件都会重建这个 effect ——
     * 计时器被反复销毁，寿命（≈50ms）永远够不到周期（{@link DISPLAY_TICK_MS}），
     * 回调从不执行，显示值因此冻结。目标值改由 `targetProgressRef` 承载（事件回调写入），
     * 计时器只按固定节拍读取它；依赖里剩下的都是每次导出只变个位数的粗粒度标志。
     *
     * 【清零时机】显示值的归零只由**显式时机**负责（打开对话框 / 开始提交），不再用
     * `!submitting && !active` 隐式清零 —— 那样会在导出刚完成的瞬间把已到 100% 的显示
     * 抹掉，再靠插值爬回去（完成态因此闪烁）。
     */
    useEffect(() => {
        if (!open) return;
        if (!submitting && !exportProgress.active && !keepProgressVisible) return;
        // 冲突确认挂起期间导出暂停，显示也随之冻结。
        if (awaitingConflictDecision && !exportProgress.active) return;

        const timer = window.setInterval(() => {
            setDisplayProgress((prev) =>
                nextDisplayProgress({
                    current: prev,
                    target: targetProgressRef.current,
                    tickMs: DISPLAY_TICK_MS,
                }),
            );
        }, DISPLAY_TICK_MS);

        return () => {
            window.clearInterval(timer);
        };
    }, [open, submitting, exportProgress.active, keepProgressVisible, awaitingConflictDecision]);

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
                tf("export_dialog_sample_rate_autocorrected").replace("{rate}", String(corrected)),
            );
        } else {
            setSampleRateNotice("");
        }
    }

    /** 组装完整编码参数包（与后端 crate::encode::OutputSpec 对应）。 */
    const buildEncoderSpec = useCallback((): ExportEncoderSpec => {
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
    }, [
        format,
        channelMode,
        dither,
        wavBitDepth,
        mp3Mode,
        mp3Bitrate,
        mp3Quality,
        mp3Tags,
        flacBitDepth,
        flacLevel,
    ]);

    /** 组装导出请求（同时用于冲突预检与「导出路径示例」预览）。 */
    const buildExportRequest = useCallback((): AdvancedExportRequest | null => {
        const range =
            rangeKind === "all"
                ? { kind: "all" as const }
                : (() => {
                      const startSec = Number(customStartSec);
                      const endSec = Number(customEndSec);
                      if (
                          !Number.isFinite(startSec) ||
                          !Number.isFinite(endSec) ||
                          endSec <= startSec
                      ) {
                          return null;
                      }
                      return {
                          kind: "custom" as const,
                          startSec: Math.max(0, startSec),
                          endSec: Math.max(0, endSec),
                      };
                  })();
        if (!range) return null;

        const resolvedSampleRate = Number(sampleRate);
        if (!Number.isFinite(resolvedSampleRate) || resolvedSampleRate <= 0) return null;

        const encoder = buildEncoderSpec();

        if (mode === "project") {
            const outputDir = projectOutputDir.trim();
            const fileName = projectFileName.trim();
            if (!outputDir || !fileName) return null;
            return {
                mode: "project",
                range,
                projectOutputDir: outputDir,
                projectFileName: fileName,
                sampleRate: Math.round(resolvedSampleRate),
                format,
                encoder,
            };
        }

        const outputDir = separatedOutputDir.trim();
        if (!outputDir) return null;
        const selectedTargets = allTargets
            .filter((target) => selectedTargetIds.includes(target.id))
            .map((target) => ({
                kind: target.kind,
                trackId: target.trackId,
            }));
        if (selectedTargets.length === 0) return null;
        return {
            mode: "separated",
            range,
            separatedOutputDir: outputDir,
            separatedNamePattern: separatedNamePattern.trim() || "<ExportIndex>_<TrackName>.wav",
            separatedTargets: selectedTargets,
            sampleRate: Math.round(resolvedSampleRate),
            format,
            encoder,
        };
    }, [
        rangeKind,
        customStartSec,
        customEndSec,
        sampleRate,
        mode,
        projectOutputDir,
        projectFileName,
        separatedOutputDir,
        separatedNamePattern,
        selectedTargetIds,
        allTargets,
        format,
        buildEncoderSpec,
    ]);

    // 「导出路径示例」：配置变化时（去抖）向后端取一次导出计划，取第一条目标
    // 路径作为示例。分轨模式下即第一条分轨的文件路径，足以让用户看懂输出去向。
    // 后端对日期通配符 `%` 已做安全处理：半截 / 未知的 `%` 当作字面量渲染（不会
    // panic），因此这里不需要因为 `%` 跳过刷新——用户即使只输入单个 `%`，示例路径
    // 里也会如实显示那个 `%`。300ms 去抖已足以避免输入日期（如 %s 秒数）时文本频繁跳动。
    useEffect(() => {
        if (!open) return;
        const dir = mode === "project" ? projectOutputDir.trim() : separatedOutputDir.trim();
        if (!dir) {
            setExamplePath("");
            return;
        }
        let disposed = false;
        const timer = window.setTimeout(async () => {
            const request = buildExportRequest();
            if (disposed || !request) {
                if (!disposed) setExamplePath("");
                return;
            }
            try {
                const plan = await coreApi.previewExportAudioPlan(request);
                if (disposed) return;
                if (plan?.ok && Array.isArray(plan.targets) && plan.targets.length > 0) {
                    setExamplePath(plan.targets[0].path || "");
                } else {
                    setExamplePath("");
                }
            } catch {
                if (!disposed) setExamplePath("");
            }
        }, 300);
        return () => {
            disposed = true;
            window.clearTimeout(timer);
        };
    }, [open, mode, projectOutputDir, separatedOutputDir, buildExportRequest]);

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
        // 只对当前已选中的项做筛选：从中移除被静音规则排除的目标，
        // 未选中的项保持原样（不批量改选所有未静音目标）。
        setSelectedTargetIds((prev) =>
            prev.filter((id) => {
                const target = allTargets.find((item) => item.id === id);
                return target != null && !target.excludedByRule;
            }),
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
            setErrorText(tf("export_dialog_error_invalid_range"));
            return null;
        }
        if (endSec <= startSec) {
            setErrorText(tf("export_dialog_error_invalid_range"));
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
            setErrorText(tf("export_dialog_error_invalid_sample_rate"));
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
            return tf("export_dialog_error_invalid_time_format");
        }
        if (code === "mp3_unsupported_sample_rate") {
            return tf("export_dialog_error_mp3_unsupported_sample_rate");
        }
        return code || tf("status_export_failed");
    }

    /**
     * 只取消本次导出：**不关闭对话框**（进度条上的取消走这里）。
     *
     * 【为什么不能顺手关窗】界面里有两个取消入口，用户的预期不同：进度条旁边的取消
     * 只该结束这次渲染（用户可能只是想改个设置再导一次），而页脚的取消/关闭才是
     * "离开对话框"。早先两者共用一个 handler，于是点进度条的取消会把整个对话框也关掉。
     */
    async function cancelRunningExport() {
        if (!submitting && !exportProgress.active) return;
        try {
            await coreApi.cancelExportAudio();
        } catch {
            // 取消命令失败不阻塞 UI：下面的乐观收敛仍会把界面带回"可再次导出"。
        }
        // 乐观收敛，**不依赖取消事件是否到达**（事件可能丢失，或与 invoke 响应竞态）：
        // 清空进度状态 → 进度区收起；`submitting` 由 `submitExport` 的 cancelled 分支
        // 置回 false → 页脚「导出」重新可用。显示值与目标一并归零，避免下一次导出
        // 被上一轮残留的 100% 拉高。
        setExportProgress({
            active: false,
            mode: null,
            progress: null,
            current: null,
            total: null,
        });
        setKeepProgressVisible(false);
        targetProgressRef.current = null;
        setDisplayProgress(0);
    }

    /**
     * 关闭对话框：正在导出时**先取消**，避免留下一个没有界面的后台导出任务。
     * 页脚取消、Esc、点击遮罩都走这里。
     */
    async function handleCloseDialog() {
        await cancelRunningExport();
        onOpenChange(false);
    }

    async function submitExport() {
        setErrorText("");
        setSubmitting(true);
        setKeepProgressVisible(false);
        // 如果上一次导出已达到 100%，需要将显示进度重置为较低值
        // 否则保持当前进度（不降级），并至少从 2% 开始缓升。
        setDisplayProgress((prev) => (prev >= 100 ? 2 : Math.max(prev, 2)));
        // 目标值必须一并清空：否则上一轮残留的 100% 会在计时器下一次推进时
        // 把刚重置到 2% 的显示值直接拉回 100。
        targetProgressRef.current = null;

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
                setErrorText(tf("export_dialog_error_missing_project_output_dir"));
                setSubmitting(false);
                return;
            }
            if (!fileName) {
                setErrorText(tf("export_dialog_error_missing_project_file_name"));
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
                // 目标文件夹：优先用后端返回的 output_dir；缺失时从产物路径反推父目录
                // （兼容后端旧版本 / 各返回分支），保证「打开文件夹」可用。
                const outPaths = collectExportPaths(result);
                setLastOutputPaths(outPaths);
                const resultDir =
                    result.output_dir || (outPaths[0] ? parentDirOfPath(outPaths[0]) : "");
                if (resultDir) setLastOutputDir(resultDir);
            } catch (err) {
                // invoke / thunk 层失败此前没有任何捕获：错误成为未处理
                // 拒绝，进度条消失且无任何提示。取消保持静默
                // （mapExportError 对 export_cancelled 返回空串）。
                const message = err instanceof Error ? err.message : String(err ?? "");
                setErrorText(mapExportError(message));
            } finally {
                setSubmitting(false);
            }
            return;
        }

        const outputDir = separatedOutputDir.trim();
        if (!outputDir) {
            setErrorText(tf("export_dialog_error_missing_output_dir"));
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
            setErrorText(tf("export_dialog_error_missing_targets"));
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
            const outPaths = collectExportPaths(result);
            setLastOutputPaths(outPaths);
            const resultDir =
                result.output_dir || (outPaths[0] ? parentDirOfPath(outPaths[0]) : "");
            if (resultDir) setLastOutputDir(resultDir);
        } catch (err) {
            // 同上：invoke / thunk 层失败必须有用户可见的报错。
            const message = err instanceof Error ? err.message : String(err ?? "");
            setErrorText(mapExportError(message));
        } finally {
            setSubmitting(false);
        }
    }

    const exportCompleted =
        keepProgressVisible && !submitting && !exportProgress.active && displayProgress >= 100;

    const progressLabel = exportCompleted
        ? mode === "separated"
            ? tf("status_export_separated_done")
            : tf("status_export_done")
        : mode === "separated"
          ? (() => {
                const current = exportProgress.current;
                const total = exportProgress.total;
                if (current != null && total != null && total > 0) {
                    return `${tf("export_dialog_progress")}${" "}${current}/${total}`;
                }
                return tf("export_dialog_progress");
            })()
          : tf("export_dialog_progress");

    const shouldShowProgress = submitting || exportProgress.active || keepProgressVisible;

    return (
        <>
            <AppDialog
                open={open}
                /*
                 * 关闭请求（Esc / 点遮罩）与页脚取消同契约：**先取消再关闭**。
                 * 直接透传 onOpenChange 会让"导出中按 Esc"关掉界面却把导出留在后台跑。
                 */
                onOpenChange={(nextOpen) => {
                    if (nextOpen) {
                        onOpenChange(true);
                        return;
                    }
                    void handleCloseDialog();
                }}
                title={tf("menu_export_audio")}
                description={tf("export_dialog_desc")}
                size="xl"
                actions={[
                    ...(!submitting && (lastOutputPaths.length > 0 || lastOutputDir)
                        ? [
                              {
                                  id: "open-folder",
                                  label: tf("export_dialog_open_folder"),
                                  autoClose: false,
                                  onClick: () => {
                                      // 优先定位并选中所有已渲染文件；没有产物路径时退化为
                                      // 打开目标文件夹（后端据路径类型自动分派）。
                                      const targets =
                                          lastOutputPaths.length > 0
                                              ? lastOutputPaths
                                              : [lastOutputDir].filter(Boolean);
                                      void coreApi
                                          .revealExportPaths(targets)
                                          .catch(() => undefined);
                                  },
                              },
                          ]
                        : []),
                    {
                        id: "cancel",
                        label: tf("cancel"),
                        /*
                         * `autoClose: false`：关闭由 `handleCloseDialog` 自己负责
                         * （它要先取消正在跑的导出）。若让 AppDialog 的默认
                         * autoClose 也关一次，同一次点击会走两条关闭路径 ——
                         * 取消命令因此被发两次（回归测试钉住了这一点）。
                         */
                        autoClose: false,
                        onClick: () => {
                            void handleCloseDialog();
                        },
                    },
                    {
                        id: "export",
                        label: tf("export_dialog_export"),
                        intent: "primary",
                        disabled: session.busy || submitting,
                        autoClose: false,
                        onClick: () => {
                            void submitExport();
                        },
                    },
                ]}
            >
                {/* 让表单填满 body 高度，好把下面的"目标"列表变成唯一的滚动区
                    （见该列表上的注释）。 */}
                <AppForm labelWidth="lg" className="h-full min-h-0">
                    <AppField label={tf("export_dialog_mode")}>
                        <AppSelect
                            value={mode}
                            onValueChange={(value) => setMode(value as ExportMode)}
                            options={[
                                { value: "project", label: tf("export_dialog_mode_project") },
                                {
                                    value: "separated",
                                    label: tf("export_dialog_mode_separated"),
                                },
                            ]}
                        />
                    </AppField>

                    <AppField label={tf("export_dialog_range")}>
                        <AppSelect
                            value={rangeKind}
                            onValueChange={(value) => setRangeKind(value as ExportRangeKind)}
                            options={[
                                { value: "all", label: tf("export_dialog_range_all") },
                                { value: "custom", label: tf("export_dialog_range_custom") },
                            ]}
                        />
                    </AppField>

                    {rangeKind === "custom" && (
                        <Flex gap="2" align="center">
                            <span className="hs-type-label shrink-0" style={{ minWidth: 132 }}>
                                {tf("export_dialog_range_custom_label")}
                            </span>
                            <AppNumberField
                                value={customStartSec}
                                unit="seconds"
                                min={0}
                                width={160}
                                ariaLabel={tf("export_dialog_range_custom_label")}
                                onChange={(next) => setCustomStartSec(next)}
                                onCommit={(next) => setCustomStartSec(next)}
                            />
                            <span className="hs-type-muted">~</span>
                            <AppNumberField
                                value={customEndSec}
                                unit="seconds"
                                min={0}
                                width={160}
                                ariaLabel={tf("export_dialog_range_custom_label")}
                                onChange={(next) => setCustomEndSec(next)}
                                onCommit={(next) => setCustomEndSec(next)}
                            />
                            <span className="hs-type-caption">sec</span>
                        </Flex>
                    )}

                    <AppField label={tf("export_dialog_format")}>
                        <AppSegmentedControl
                            size="sm"
                            value={format}
                            options={[
                                { value: "wav", label: "WAV" },
                                { value: "mp3", label: "MP3" },
                                { value: "flac", label: "FLAC" },
                            ]}
                            onChange={(value) => handleFormatChange(value as ExportFormat)}
                            ariaLabel={tf("export_dialog_format")}
                        />
                    </AppField>

                    <AppField label={tf("export_dialog_sample_rate")}>
                        <AppSelect
                            value={sampleRate}
                            onValueChange={(value) => {
                                setSampleRate(value);
                                setSampleRateNotice("");
                            }}
                            options={sampleRateOptions(format).map((rate) => ({
                                value: String(rate),
                                label: `${rate} Hz`,
                            }))}
                        />
                    </AppField>

                    {sampleRateNotice ? (
                        <span
                            className="hs-type-caption"
                            style={{ color: "var(--qt-warning-text)" }}
                        >
                            {sampleRateNotice}
                        </span>
                    ) : null}

                    {format !== "mp3" ? (
                        <AppField label={tf("export_dialog_bit_depth")}>
                            <AppSelect
                                value={format === "wav" ? wavBitDepth : flacBitDepth}
                                onValueChange={(value) => {
                                    if (format === "wav") {
                                        if (value === "i16" || value === "i24" || value === "f32") {
                                            setWavBitDepth(value);
                                        }
                                    } else if (value === "i16" || value === "i24") {
                                        setFlacBitDepth(value);
                                    }
                                }}
                                options={[
                                    { value: "i16", label: "16-bit" },
                                    { value: "i24", label: "24-bit" },
                                    ...(format === "wav"
                                        ? [{ value: "f32", label: "32-bit float" }]
                                        : []),
                                ]}
                            />
                        </AppField>
                    ) : (
                        <span className="hs-type-caption">
                            {tf("export_dialog_mp3_bit_depth_note")}
                        </span>
                    )}

                    <Flex align="center" gap="2">
                        <AppButton size="sm" onClick={() => setEncoderOpen((prev) => !prev)}>
                            {encoderOpen ? "▾" : "▸"} {tf("export_dialog_encoder_params")}
                        </AppButton>
                    </Flex>

                    {encoderOpen && (
                        <Flex direction="column" gap="3" pl="1">
                            {format === "mp3" && (
                                <>
                                    <AppField label={tf("export_dialog_mp3_mode")}>
                                        <AppSelect
                                            value={mp3Mode}
                                            onValueChange={(value) => {
                                                if (value === "cbr" || value === "vbr") {
                                                    setMp3Mode(value);
                                                }
                                            }}
                                            options={[
                                                {
                                                    value: "vbr",
                                                    label: tf("export_dialog_mp3_mode_vbr"),
                                                },
                                                {
                                                    value: "cbr",
                                                    label: tf("export_dialog_mp3_mode_cbr"),
                                                },
                                            ]}
                                        />
                                    </AppField>

                                    {mp3Mode === "cbr" ? (
                                        <AppField label={tf("export_dialog_mp3_bitrate")}>
                                            <AppSelect
                                                value={String(mp3Bitrate)}
                                                onValueChange={(value) =>
                                                    setMp3Bitrate(Number(value))
                                                }
                                                options={MP3_BITRATES.map((rate) => ({
                                                    value: String(rate),
                                                    label: `${rate} kbps`,
                                                }))}
                                            />
                                        </AppField>
                                    ) : (
                                        <AppField label={tf("export_dialog_mp3_quality")}>
                                            <AppSelect
                                                value={String(mp3Quality)}
                                                onValueChange={(value) =>
                                                    setMp3Quality(Number(value))
                                                }
                                                options={MP3_VBR_AVG_KBPS.map((avg, index) => ({
                                                    value: String(index),
                                                    label: `q${index} · ~${avg} kbps`,
                                                }))}
                                            />
                                        </AppField>
                                    )}

                                    <Flex direction="column" gap="2">
                                        <span className="hs-type-label">
                                            {tf("export_dialog_mp3_tags")}
                                        </span>
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
                                                    className="flex flex-col gap-1 text-qt-xs text-qt-text"
                                                >
                                                    <span className="hs-type-label">
                                                        {tf(i18nKey)}
                                                    </span>
                                                    <TextField.Root
                                                        size="1"
                                                        value={mp3Tags[key] ?? ""}
                                                        onChange={(
                                                            event: ChangeEvent<HTMLInputElement>,
                                                        ) =>
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
                                        <span
                                            className="hs-type-label shrink-0"
                                            style={{ minWidth: 132 }}
                                        >
                                            {tf("export_dialog_flac_level")}
                                        </span>
                                        <AppSlider
                                            value={flacLevel}
                                            unit="integer"
                                            min={FLAC_COMPRESSION_RANGE.min}
                                            max={FLAC_COMPRESSION_RANGE.max}
                                            ariaLabel={tf("export_dialog_flac_level")}
                                            onChange={(next) => setFlacLevel(next)}
                                        />
                                        <AppSliderReadout>{flacLevel}</AppSliderReadout>
                                    </Flex>
                                    <span className="hs-type-caption">
                                        {tf("export_dialog_flac_level_hint")}
                                    </span>
                                </>
                            )}

                            {((format === "wav" &&
                                (wavBitDepth === "i16" || wavBitDepth === "i24")) ||
                                format === "flac") && (
                                <AppField label={tf("export_dialog_dither")}>
                                    <AppSelect
                                        value={dither}
                                        onValueChange={(value) => {
                                            if (value === "none" || value === "tpdf") {
                                                setDither(value);
                                            }
                                        }}
                                        options={[
                                            {
                                                value: "none",
                                                label: tf("export_dialog_dither_none"),
                                            },
                                            {
                                                value: "tpdf",
                                                label: tf("export_dialog_dither_tpdf"),
                                            },
                                        ]}
                                    />
                                </AppField>
                            )}

                            <AppField label={tf("export_dialog_channel_mode")}>
                                <AppSelect
                                    value={channelMode}
                                    onValueChange={(value) => {
                                        if (value === "stereo" || value === "mono") {
                                            setChannelMode(value);
                                        }
                                    }}
                                    options={[
                                        {
                                            value: "stereo",
                                            label: tf("export_dialog_channel_stereo"),
                                        },
                                        {
                                            value: "mono",
                                            label: tf("export_dialog_channel_mono"),
                                        },
                                    ]}
                                />
                            </AppField>
                        </Flex>
                    )}

                    {mode === "project" ? (
                        <>
                            <Flex align="center" gap="2">
                                <span className="hs-type-label shrink-0" style={{ minWidth: 132 }}>
                                    {tf("export_dialog_output_dir")}
                                </span>
                                <TextField.Root
                                    size="2"
                                    value={projectOutputDir}
                                    onFocus={(event) => {
                                        setActiveInputKey("projectOutputDir");
                                        activeInputRef.current = event.target as HTMLInputElement;
                                    }}
                                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                        setProjectOutputDir(event.target.value)
                                    }
                                    style={{ flex: 1 }}
                                />
                                <AppButton size="sm" onClick={() => void browseProjectOutputDir()}>
                                    {tf("export_dialog_browse")}
                                </AppButton>
                            </Flex>

                            <AppField label={tf("export_dialog_project_file_name")}>
                                <TextField.Root
                                    size="2"
                                    value={projectFileName}
                                    onFocus={(event) => {
                                        setActiveInputKey("projectFileName");
                                        activeInputRef.current = event.target as HTMLInputElement;
                                    }}
                                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                        setProjectFileName(event.target.value)
                                    }
                                />
                            </AppField>

                            <Flex gap="2" wrap="wrap" align="center">
                                <span className="hs-type-caption">
                                    {tf("export_pattern_placeholders")}
                                </span>
                                {(["<ProjectName>", "<ProjectFolder>"] as const).map((token) => (
                                    <AppButton
                                        key={token}
                                        size="sm"
                                        onClick={() => applyTokenToActiveInput(token)}
                                    >
                                        {token}
                                    </AppButton>
                                ))}
                            </Flex>

                            {examplePath ? (
                                <span
                                    className="hs-type-muted"
                                    style={{ userSelect: "text", wordBreak: "break-all" }}
                                >
                                    {tf("export_dialog_example_path").replace(
                                        "{path}",
                                        examplePath,
                                    )}
                                </span>
                            ) : null}
                        </>
                    ) : (
                        <>
                            <Flex align="center" gap="2">
                                <span className="hs-type-label shrink-0" style={{ minWidth: 132 }}>
                                    {tf("export_dialog_output_dir")}
                                </span>
                                <TextField.Root
                                    size="2"
                                    value={separatedOutputDir}
                                    onFocus={(event) => {
                                        setActiveInputKey("separatedOutputDir");
                                        activeInputRef.current = event.target as HTMLInputElement;
                                    }}
                                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                        setSeparatedOutputDir(event.target.value)
                                    }
                                    style={{ flex: 1 }}
                                />
                                <AppButton
                                    size="sm"
                                    onClick={() => void browseSeparatedOutputDir()}
                                >
                                    {tf("export_dialog_browse")}
                                </AppButton>
                            </Flex>

                            <AppField label={tf("export_dialog_name_pattern")}>
                                <TextField.Root
                                    size="2"
                                    value={separatedNamePattern}
                                    onFocus={(event) => {
                                        setActiveInputKey("separatedNamePattern");
                                        activeInputRef.current = event.target as HTMLInputElement;
                                    }}
                                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                                        setSeparatedNamePattern(event.target.value)
                                    }
                                />
                            </AppField>

                            <Flex gap="2" wrap="wrap" align="center">
                                <span className="hs-type-caption">
                                    {tf("export_pattern_placeholders")}
                                </span>
                                {[
                                    "<ExportIndex>",
                                    "<TrackIndex>",
                                    "<TrackName>",
                                    "<TrackType>",
                                    "<TrackId>",
                                    "<ProjectName>",
                                    "<ProjectFolder>",
                                ].map((token) => (
                                    <AppButton
                                        key={token}
                                        size="sm"
                                        onClick={() => applyTokenToActiveInput(token)}
                                    >
                                        {token}
                                    </AppButton>
                                ))}
                            </Flex>

                            {examplePath ? (
                                <span
                                    className="hs-type-muted"
                                    style={{ userSelect: "text", wordBreak: "break-all" }}
                                >
                                    {tf("export_dialog_example_path").replace(
                                        "{path}",
                                        examplePath,
                                    )}
                                </span>
                            ) : null}

                            {/*
                             * 外层不滚、内层滚：对话框 body 本身就是
                             * `overflow-y-auto`，这里再叠一个 `max-h-[240px]` 的
                             * 滚动盒，表单变高时就会出现两条竖直滚动条。改为参与
                             * 表单的 flex 布局（AppForm 已是 flex 列），让它吃掉
                             * 剩余高度。
                             */}
                            <div className="hs-scroll-gutter min-h-0 flex-1 overflow-y-auto rounded border border-qt-border bg-qt-base p-2">
                                <span className="hs-type-label font-semibold">
                                    {tf("export_dialog_targets")}
                                </span>
                                <Flex gap="1" mt="2" wrap="wrap">
                                    <AppButton size="sm" onClick={selectAllTargets}>
                                        {tf("export_dialog_select_all")}
                                    </AppButton>
                                    <AppButton size="sm" onClick={clearSelectedTargets}>
                                        {tf("export_dialog_select_none")}
                                    </AppButton>
                                    <AppButton
                                        size="sm"
                                        onClick={selectAllSubTargets}
                                        disabled={
                                            !allTargets.some((target) => target.kind === "sub")
                                        }
                                    >
                                        {tf("export_dialog_select_all_subtracks")}
                                    </AppButton>
                                    <AppButton size="sm" onClick={selectExcludeMutedTargets}>
                                        {tf("export_dialog_select_exclude_muted")}
                                    </AppButton>
                                </Flex>
                                <Flex direction="column" gap="2" mt="2">
                                    {targetGroups.map((group) => (
                                        <div
                                            key={group.id}
                                            className="rounded border border-qt-border bg-qt-window px-2 py-2"
                                        >
                                            <span className="hs-type-label">{group.title}</span>
                                            <AppForm booleanRow="leading">
                                                <Flex direction="column" gap="1" mt="1">
                                                    {group.options.map((target) => (
                                                        <AppSwitchRow
                                                            key={target.id}
                                                            control="checkbox"
                                                            label={target.label}
                                                            checked={selectedTargetIds.includes(
                                                                target.id,
                                                            )}
                                                            onCheckedChange={() =>
                                                                toggleTarget(target.id)
                                                            }
                                                        />
                                                    ))}
                                                </Flex>
                                            </AppForm>
                                        </div>
                                    ))}
                                </Flex>
                            </div>
                        </>
                    )}

                    {errorText ? (
                        <span className="hs-type-body" style={{ color: "var(--qt-danger-text)" }}>
                            {errorText}
                        </span>
                    ) : null}

                    {shouldShowProgress ? (
                        <div className="rounded border border-qt-border bg-qt-window p-2">
                            <ProgressBar
                                percentage={displayProgress}
                                label={progressLabel}
                                completed={exportCompleted}
                                /*
                                 * 长任务必须能取消：一次多目标导出可能跑几分钟，
                                 * 而进度区此前只有读数、没有出口。后端本来就有
                                 * `cancel_export_audio`（页脚的取消也走它），
                                 * 这里只是把它接到进度条上。
                                 */
                                showCancel={!exportCompleted && exportProgress.active}
                                onCancel={() => void cancelRunningExport()}
                            />
                        </div>
                    ) : null}
                </AppForm>
            </AppDialog>
            <AppDialog
                open={conflictDialog.open}
                onOpenChange={(open) => {
                    if (!open) {
                        const resolver = conflictResolverRef.current;
                        conflictResolverRef.current = null;
                        setConflictDialog((prev) => ({ ...prev, open: false }));
                        resolver?.({ choice: "cancel", applyAll: false });
                    }
                }}
                title={
                    conflictDialog.kind === "source-path"
                        ? tf("export_conflict_source_title")
                        : tf("export_conflict_exists_title")
                }
                message={
                    <span
                        style={{
                            userSelect: "text",
                            wordBreak: "break-all",
                            whiteSpace: "pre-wrap",
                        }}
                    >
                        {conflictDialog.kind === "source-path"
                            ? tf("export_conflict_source_desc")
                            : tf("export_conflict_exists_desc")}
                        {"\n"}
                        {conflictDialog.path}
                    </span>
                }
                /*
                 * 只有 source-path 那一类是**不可逆的数据丢失**（导出目标与工程
                 * 媒体同路径，覆写不可逆），给它 danger；"目标已存在"用 skip 就能
                 * 绕开，属于可恢复情形，保持默认样式 —— 到处报警等于没有报警。
                 */
                tone={conflictDialog.kind === "source-path" ? "danger" : "default"}
                size="lg"
                /*
                 * Enter 的默认动作必须显式指定。
                 *
                 * 壳的默认规则是"最后一个非危险动作"，即 overwrite。但 source-path
                 * 一类的文案是**数据丢失警告**（导出目标与工程媒体同路径，覆写不可
                 * 逆）—— 那种情形下让 Enter 直接覆写是危险的默认值。这一类把默认
                 * 动作钉在 skip 上。
                 */
                defaultActionId={conflictDialog.kind === "source-path" ? "skip" : "overwrite"}
                actions={[
                    {
                        id: "skip",
                        label: tf("export_conflict_skip"),
                        intent: conflictDialog.kind === "source-path" ? "primary" : undefined,
                        autoClose: false,
                        onClick: () => {
                            const resolver = conflictResolverRef.current;
                            conflictResolverRef.current = null;
                            setConflictDialog((prev) => ({ ...prev, open: false }));
                            resolver?.({ choice: "skip", applyAll: conflictDialog.applyAll });
                        },
                    },
                    {
                        id: "cancel",
                        label: tf("export_conflict_cancel"),
                        intent: "danger",
                        autoClose: false,
                        onClick: () => {
                            const resolver = conflictResolverRef.current;
                            conflictResolverRef.current = null;
                            setConflictDialog((prev) => ({ ...prev, open: false }));
                            resolver?.({ choice: "cancel", applyAll: false });
                        },
                    },
                    {
                        id: "overwrite",
                        label: tf("export_conflict_overwrite"),
                        intent: conflictDialog.kind === "source-path" ? undefined : "primary",
                        autoClose: false,
                        onClick: () => {
                            const resolver = conflictResolverRef.current;
                            conflictResolverRef.current = null;
                            setConflictDialog((prev) => ({ ...prev, open: false }));
                            resolver?.({
                                choice: "overwrite",
                                applyAll: conflictDialog.applyAll,
                            });
                        },
                    },
                ]}
            >
                <AppForm booleanRow="leading">
                    <AppSwitchRow
                        control="checkbox"
                        label={tf("export_conflict_apply_all")}
                        checked={conflictDialog.applyAll}
                        onCheckedChange={(applyAll) =>
                            setConflictDialog((prev) => ({
                                ...prev,
                                applyAll,
                            }))
                        }
                    />
                </AppForm>
            </AppDialog>
        </>
    );
}
