/*
 * 导出音频配置对话框。
 * 负责收集导出模式、时间范围、输出路径与分轨命名/目标选择，并调用后端统一导出命令。
 */

import { useEffect, useMemo, useRef, useState, type ChangeEvent } from "react";
import { Button, Dialog, Flex, Select, Text, TextField } from "@radix-ui/themes";
import { useI18n } from "../../i18n/I18nProvider";
import { useAppDispatch, useAppSelector } from "../../app/hooks";
import { exportAudioAdvanced } from "../../features/session/sessionSlice";
import { fileBrowserApi } from "../../services/api/fileBrowser";
import { coreApi } from "../../services/api/core";
import { ProgressBar } from "../ProgressBar";
import type { TrackInfo } from "../../features/session/sessionTypes";
import { applySelectWheelChange } from "../../utils/selectWheel";

interface ExportAudioDialogProps {
    open: boolean;
    onOpenChange: (open: boolean) => void;
}

type ExportMode = "project" | "separated";
type ExportRangeKind = "all" | "custom";
type ExportBitDepth = 16 | 24 | 32;

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
        if (Boolean(track.muted)) return true;

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
    const [bitDepth, setBitDepth] = useState<ExportBitDepth>(32);
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

    useEffect(() => {
        if (!open) return;

        setMode("project");
        setRangeKind("all");
        setCustomStartSec("0");
        setCustomEndSec(String(Math.max(0, Math.ceil(session.projectSec))));
        setProjectOutputDir("");
        setProjectFileName("<ProjectName>.wav");
        setSeparatedOutputDir("");
        setSeparatedNamePattern("<ExportIndex>_<TrackName>.wav");
        setSampleRate("48000");
        setBitDepth(32);
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
    }, [open, session.projectSec, targetGroups]);

    useEffect(() => {
        if (!open) return;
        let disposed = false;

        async function loadDefaults() {
            try {
                const defaults = await coreApi.getExportAudioDefaults();
                if (disposed || !defaults?.ok) return;
                setProjectOutputDir(defaults.projectOutputDir ?? "");
                setProjectFileName(defaults.projectFileName ?? "<ProjectName>.wav");
                setSeparatedOutputDir(defaults.separatedOutputDir ?? "");
                setSeparatedNamePattern(
                    defaults.separatedFileName ?? "<ExportIndex>_<TrackName>.wav",
                );
                setSampleRate(String(defaults.sampleRate ?? 48000));
                setBitDepth(
                    defaults.bitDepth === 16 || defaults.bitDepth === 24 || defaults.bitDepth === 32
                        ? defaults.bitDepth
                        : 32,
                );
            } catch {
                // 保持回退默认值。
            }
        }

        void loadDefaults();
        return () => {
            disposed = true;
        };
    }, [open]);

    useEffect(() => {
        if (!open) return;
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            try {
                const mod = await import("@tauri-apps/api/event");
                unlisten = await mod.listen("export_audio_progress", (event: any) => {
                    if (disposed) return;
                    const payload = (event?.payload ?? {}) as {
                        active?: boolean;
                        mode?: "project" | "separated";
                        progress?: number | null;
                        current?: number | null;
                        total?: number | null;
                    };

                    const progressValue =
                        typeof payload.progress === "number" && Number.isFinite(payload.progress)
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
                            typeof payload.current === "number" && Number.isFinite(payload.current)
                                ? Math.max(0, Math.floor(payload.current))
                                : null,
                        total:
                            typeof payload.total === "number" && Number.isFinite(payload.total)
                                ? Math.max(0, Math.floor(payload.total))
                                : null,
                    });
                });
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

    async function resolveExportConflicts(request: any) {
        const plan = await coreApi.previewExportAudioPlan(request as any);
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
                    bitDepth,
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
                        bitDepth,
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
                bitDepth,
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
                    bitDepth,
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
                                {tAny("export_dialog_sample_rate")}
                            </Text>
                            <Select.Root
                                value={sampleRate}
                                onValueChange={(value) => setSampleRate(value)}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: sampleRate,
                                            options: [
                                                "22050",
                                                "32000",
                                                "44100",
                                                "48000",
                                                "88200",
                                                "96000",
                                            ],
                                            onChange: setSampleRate,
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="22050">22050 Hz</Select.Item>
                                    <Select.Item value="32000">32000 Hz</Select.Item>
                                    <Select.Item value="44100">44100 Hz</Select.Item>
                                    <Select.Item value="48000">48000 Hz</Select.Item>
                                    <Select.Item value="88200">88200 Hz</Select.Item>
                                    <Select.Item value="96000">96000 Hz</Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </Flex>

                        <Flex align="center" gap="2">
                            <Text size="2" style={{ minWidth: 132 }}>
                                {tAny("export_dialog_bit_depth")}
                            </Text>
                            <Select.Root
                                value={String(bitDepth)}
                                onValueChange={(value) => {
                                    const parsed = Number(value);
                                    if (parsed === 16 || parsed === 24 || parsed === 32) {
                                        setBitDepth(parsed);
                                    }
                                }}
                            >
                                <Select.Trigger
                                    style={{ flex: 1 }}
                                    onWheel={(event) => {
                                        applySelectWheelChange({
                                            event,
                                            currentValue: String(bitDepth),
                                            options: ["16", "24", "32"],
                                            onChange: (next) => {
                                                const parsed = Number(next);
                                                if (
                                                    parsed === 16 ||
                                                    parsed === 24 ||
                                                    parsed === 32
                                                ) {
                                                    setBitDepth(parsed);
                                                }
                                            },
                                        });
                                    }}
                                />
                                <Select.Content>
                                    <Select.Item value="16">16-bit</Select.Item>
                                    <Select.Item value="24">24-bit</Select.Item>
                                    <Select.Item value="32">32-bit</Select.Item>
                                </Select.Content>
                            </Select.Root>
                        </Flex>

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
                                        占位符：
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
                                        占位符：
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
