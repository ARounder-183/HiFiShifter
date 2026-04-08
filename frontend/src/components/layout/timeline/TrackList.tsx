import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Flex, Box, Text, IconButton, Select } from "@radix-ui/themes";
import { Cross2Icon, PlusIcon } from "@radix-ui/react-icons";
import type { TrackInfo, TrackMeterInfo } from "../../../features/session/sessionTypes";
import { isNoneBinding, isModifierActive } from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import type { MessageKey } from "../../../i18n/messages";
import { MAX_ROW_HEIGHT, MIN_ROW_HEIGHT, TRACK_ADD_ROW_HEIGHT } from "./constants";

/** Color palette options shown when creating a new track. */
const TRACK_COLOR_PALETTE_KEYS: { value: string; key: MessageKey }[] = [
    { value: "#228be6", key: "color_blue" }, // Open Color blue6
    { value: "#ae3ec9", key: "color_purple" }, // Open Color grape6
    { value: "#0ca678", key: "color_green" }, // Open Color teal6
    { value: "#f76707", key: "color_orange" }, // Open Color orange6
    { value: "#d6336c", key: "color_pink" }, // Open Color pink6
    { value: "#15aabf", key: "color_sky_blue" }, // Open Color cyan6
    { value: "#f08c00", key: "color_yellow" }, // Open Color yellow6
    { value: "#f03e3e", key: "color_red" }, // Open Color red6
];
const PITCH_ANALYSIS_ALGO_OPTIONS = ["world_dll", "nsf_hifigan_onnx", "vslib", "none"] as const;

const TRACK_METER_MIN_DB = -48;
const TRACK_METER_MAX_DB = 3;
const TRACK_GAIN_MIN_DB = -60;
const TRACK_GAIN_MAX_DB = 12;
const TRACK_GAIN_WHEEL_STEP_DB = 0.5;
const TRACK_GAIN_WHEEL_FINE_STEP_DB = 0.1;
const TRACK_GAIN_WHEEL_COMMIT_DEBOUNCE_MS = 120;
const TRACK_GAIN_DRAG_DB_PER_PX = 0.2;
const TRACK_GAIN_DRAG_FINE_SCALE = 0.2;
const TRACK_GAIN_DRAG_FINE_PULLBACK_RATIO = 0.35;

type FineAxisDragState = {
    raw: number;
    adjusted: number;
    fineActive: boolean;
};

function advanceFineAxisDrag(
    state: FineAxisDragState,
    nextRaw: number,
    fineActive: boolean,
): number {
    const delta = nextRaw - state.raw;
    if (fineActive && !state.fineActive) {
        state.adjusted += delta * (1 - TRACK_GAIN_DRAG_FINE_PULLBACK_RATIO);
    } else {
        const scale = fineActive ? TRACK_GAIN_DRAG_FINE_SCALE : 1;
        state.adjusted += delta * scale;
    }
    state.raw = nextRaw;
    state.fineActive = fineActive;
    return state.adjusted;
}

function gainToDb(gain: number): number {
    if (!Number.isFinite(gain) || gain <= 1e-4) return TRACK_GAIN_MIN_DB;
    return 20 * Math.log10(gain);
}

function dbToGain(db: number): number {
    if (!Number.isFinite(db) || db <= TRACK_GAIN_MIN_DB) return 0;
    return Math.pow(10, db / 20);
}

function formatGainLabel(gain: number): string {
    const db = gainToDb(gain);
    if (!Number.isFinite(db) || db <= TRACK_GAIN_MIN_DB + 0.05) return "-inf dB";
    const clampedDb = Math.min(TRACK_GAIN_MAX_DB, Math.max(TRACK_GAIN_MIN_DB, db));
    if (Math.abs(clampedDb) < 0.05) return "0.0 dB";
    return `${clampedDb > 0 ? "+" : ""}${clampedDb.toFixed(1)} dB`;
}

function clampGainDb(db: number): number {
    if (!Number.isFinite(db)) return TRACK_GAIN_MIN_DB;
    return Math.min(TRACK_GAIN_MAX_DB, Math.max(TRACK_GAIN_MIN_DB, db));
}

function linearToDb(linear: number): number {
    if (!Number.isFinite(linear) || linear <= 1e-6) return -Infinity;
    return 20 * Math.log10(linear);
}

function meterHeightPercent(linear: number): number {
    const db = linearToDb(linear);
    if (!Number.isFinite(db)) return 0;
    const normalized =
        (Math.min(TRACK_METER_MAX_DB, Math.max(TRACK_METER_MIN_DB, db)) - TRACK_METER_MIN_DB) /
        (TRACK_METER_MAX_DB - TRACK_METER_MIN_DB);
    return normalized * 100;
}

function formatPeakLabel(maxPeakLinear: number, clipped: boolean): string {
    if (clipped || maxPeakLinear >= 1) {
        const overDb = linearToDb(Math.max(1, maxPeakLinear));
        return Number.isFinite(overDb) ? `+${Math.max(0, overDb).toFixed(1)}` : "+0.0";
    }
    const db = linearToDb(maxPeakLinear);
    if (!Number.isFinite(db)) return "-inf";
    return db.toFixed(1);
}

function meterFillClass(peakLinear: number, clipped: boolean): string {
    if (clipped || peakLinear >= 1) return "bg-red-500";
    const db = linearToDb(peakLinear);
    if (db >= -6) return "bg-orange-400";
    if (db >= -18) return "bg-yellow-400";
    return "bg-emerald-400";
}

export const TrackList: React.FC<{
    t: (key: MessageKey) => string;
    tracks: TrackInfo[];
    trackMeters: Record<string, TrackMeterInfo>;
    selectedTrackId: string | null;
    rowHeight: number;
    setRowHeight?: React.Dispatch<React.SetStateAction<number>>;
    verticalZoomKb?: Keybinding;
    paramFineAdjustKb: Keybinding;
    trackVolumeUi: Record<string, number>;
    onSelectTrack: (trackId: string) => void;
    onRemoveTrack: (trackId: string) => void;
    onMoveTrack: (payload: {
        trackId: string;
        targetIndex: number;
        parentTrackId: string | null;
    }) => void;
    onToggleMute: (trackId: string, nextMuted: boolean) => void;
    onToggleSolo: (trackId: string, nextSolo: boolean) => void;
    onToggleCompose: (trackId: string, nextComposeEnabled: boolean) => void;
    onVolumeUiChange: (trackId: string, nextVolume: number) => void;
    onVolumeCommit: (trackId: string, nextVolume: number) => void;
    onAddTrack: () => void;
    onTrackColorChange?: (trackId: string, color: string) => void;
    onAlgoChange?: (trackId: string, algo: string) => void;
    onTrackNameChange?: (trackId: string, name: string) => void;
    onDuplicateTrack?: (trackId: string) => void;
    onScrollTopChange?: (scrollTop: number) => void;
    /** 外部持有该滚动容器的 ref，用于同步右侧轨道区的竖向滚�?*/
    listScrollRef?: React.MutableRefObject<HTMLDivElement | null>;
}> = ({
    t,
    tracks,
    trackMeters,
    selectedTrackId,
    rowHeight,
    setRowHeight,
    verticalZoomKb,
    paramFineAdjustKb,
    trackVolumeUi,
    onSelectTrack,
    onRemoveTrack,
    onMoveTrack,
    onToggleMute,
    onToggleSolo,
    onToggleCompose,
    onVolumeUiChange,
    onVolumeCommit,
    onAddTrack,
    onTrackColorChange,
    onAlgoChange,
    onTrackNameChange,
    onDuplicateTrack,
    onScrollTopChange,
    listScrollRef,
}) => {
    const listRef = useRef<HTMLDivElement | null>(null);
    const rowHeightRef = useRef(rowHeight);
    const pendingVerticalZoomRef = useRef<{
        nextRowHeight: number;
        nextScrollTop: number;
    } | null>(null);
    const panRef = useRef<{
        pointerId: number | null;
        startY: number;
        scrollTop: number;
    } | null>(null);
    const dragRef = useRef<{
        pointerId: number;
        trackId: string;
        startClientX: number;
        startClientY: number;
        hasMoved: boolean;
        originalParentId: string | null;
        originalIndexSelf: number;
    } | null>(null);

    const [dragUi, setDragUi] = useState<{
        draggingTrackId: string;
        overTrackId: string | null;
        mode: "reorder" | "nest";
        indicatorY: number | null;
    } | null>(null);
    const volumeCommitTimersRef = useRef<Record<string, number>>({});

    // 轨道颜色选择器弹出状�?
    const [colorPickerTrackId, setColorPickerTrackId] = useState<string | null>(null);

    // 轨道名称行内编辑状�?
    const [editingTrackId, setEditingTrackId] = useState<string | null>(null);
    const [editingName, setEditingName] = useState("");
    const nameInputRef = useRef<HTMLInputElement | null>(null);

    // 轨道右键菜单状�?
    const [trackCtxMenu, setTrackCtxMenu] = useState<{
        x: number;
        y: number;
        trackId: string;
    } | null>(null);
    const trackCtxMenuRef = useRef<HTMLDivElement | null>(null);

    // 自动修正菜单溢出屏幕
    useLayoutEffect(() => {
        const el = trackCtxMenuRef.current;
        if (!el || !trackCtxMenu) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw) {
            el.style.left = `${Math.max(0, vw - rect.width)}px`;
        }
        if (rect.bottom > vh) {
            el.style.top = `${Math.max(0, vh - rect.height)}px`;
        }
    }, [trackCtxMenu]);

    // 点击其他区域关闭右键菜单
    useEffect(() => {
        if (!trackCtxMenu) return;
        const handler = (e: PointerEvent) => {
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-track-ctx-menu]")) return;
            setTrackCtxMenu(null);
        };
        window.addEventListener("pointerdown", handler, true);
        return () => window.removeEventListener("pointerdown", handler, true);
    }, [trackCtxMenu]);

    function commitTrackName() {
        if (!editingTrackId) return;
        const trimmed = editingName.trim();
        if (trimmed && onTrackNameChange) {
            onTrackNameChange(editingTrackId, trimmed);
        }
        setEditingTrackId(null);
    }

    // 点击其他区域关闭颜色选择�?
    useEffect(() => {
        if (!colorPickerTrackId) return;
        const handler = (e: PointerEvent) => {
            const target = e.target as HTMLElement | null;
            if (target?.closest?.("[data-track-color-picker]")) return;
            setColorPickerTrackId(null);
        };
        window.addEventListener("pointerdown", handler, true);
        return () => window.removeEventListener("pointerdown", handler, true);
    }, [colorPickerTrackId]);

    const parentById = useMemo(() => {
        const m = new Map<string, string | null>();
        for (const tr of tracks) {
            m.set(tr.id, tr.parentId ?? null);
        }
        return m;
    }, [tracks]);

    /** 根轨道数�?*/
    const rootTrackCount = useMemo(
        () => tracks.filter((t) => (t.parentId ?? null) == null).length,
        [tracks],
    );

    const backendTrackVolumeById = useMemo(() => {
        const out: Record<string, number> = {};
        for (const tr of tracks) {
            out[tr.id] = Math.max(0, Math.min(4, Number(tr.volume ?? 1)));
        }
        return out;
    }, [tracks]);

    const currentTrackVolumeById = useMemo(() => {
        const out: Record<string, number> = {};
        for (const tr of tracks) {
            const backendVolume = Math.max(0, Math.min(4, Number(tr.volume ?? 1)));
            const uiOverride = trackVolumeUi[tr.id];
            out[tr.id] = Number.isFinite(uiOverride) ? uiOverride : backendVolume;
        }
        return out;
    }, [tracks, trackVolumeUi]);

    const aggregatedTrackMeters = useMemo(() => {
        const childrenByParent = new Map<string | null, string[]>();
        for (const tr of tracks) {
            const parentId = tr.parentId ?? null;
            const list = childrenByParent.get(parentId);
            if (list) {
                list.push(tr.id);
            } else {
                childrenByParent.set(parentId, [tr.id]);
            }
        }

        const cache = new Map<string, TrackMeterInfo>();
        const visiting = new Set<string>();

        const visit = (trackId: string): TrackMeterInfo => {
            const cached = cache.get(trackId);
            if (cached) return cached;

            if (visiting.has(trackId)) {
                return { peakLinear: 0, maxPeakLinear: 0, clipped: false };
            }
            visiting.add(trackId);

            const own = trackMeters[trackId] ?? {
                peakLinear: 0,
                maxPeakLinear: 0,
                clipped: false,
            };
            const merged: TrackMeterInfo = {
                peakLinear: own.peakLinear,
                maxPeakLinear: own.maxPeakLinear,
                clipped: own.clipped,
            };

            for (const childId of childrenByParent.get(trackId) ?? []) {
                const child = visit(childId);
                if (child.peakLinear > merged.peakLinear) merged.peakLinear = child.peakLinear;
                if (child.maxPeakLinear > merged.maxPeakLinear)
                    merged.maxPeakLinear = child.maxPeakLinear;
                if (child.clipped) merged.clipped = true;
            }

            visiting.delete(trackId);
            cache.set(trackId, merged);
            return merged;
        };

        const out: Record<string, TrackMeterInfo> = {};
        for (const tr of tracks) {
            out[tr.id] = visit(tr.id);
        }
        return out;
    }, [tracks, trackMeters]);

    const meterDisplayByTrackId = useMemo(() => {
        const out: Record<string, TrackMeterInfo> = {};
        for (const tr of tracks) {
            const postGroup = aggregatedTrackMeters[tr.id] ?? {
                peakLinear: 0,
                maxPeakLinear: 0,
                clipped: false,
            };

            let ancestorGain = 1;
            let cur = tr.parentId ?? null;
            let guard = 0;
            while (cur && guard++ < 2048) {
                ancestorGain *= backendTrackVolumeById[cur] ?? 1;
                cur = parentById.get(cur) ?? null;
            }

            if ((tr.parentId ?? null) == null || ancestorGain <= 1e-6) {
                out[tr.id] = postGroup;
                continue;
            }

            const peakLinear = Math.max(0, postGroup.peakLinear / ancestorGain);
            const maxPeakLinear = Math.max(0, postGroup.maxPeakLinear / ancestorGain);
            out[tr.id] = {
                peakLinear,
                maxPeakLinear,
                clipped: peakLinear >= 1 || maxPeakLinear >= 1,
            };
        }
        return out;
    }, [aggregatedTrackMeters, backendTrackVolumeById, parentById, tracks]);

    /**
     * 判断是否不允许删除该轨道�?
     * 当该轨道是根轨道且只剩最后一个根轨道时，禁止删除（否则会导致零轨道）�?
     * 子轨道的删除不会导致零轨道，始终允许�?
     */
    function isLastRootTrack(trackId: string): boolean {
        if (rootTrackCount > 1) return false;
        const track = tracks.find((t) => t.id === trackId);
        return !!track && (track.parentId ?? null) == null;
    }

    useEffect(() => {
        return () => {
            // Safety cleanup.
            dragRef.current = null;
            setDragUi(null);
            const timerIds = Object.values(volumeCommitTimersRef.current);
            for (const timerId of timerIds) {
                window.clearTimeout(timerId);
            }
            volumeCommitTimersRef.current = {};
        };
    }, []);

    function clearPendingVolumeCommit(trackId: string) {
        const timerId = volumeCommitTimersRef.current[trackId];
        if (typeof timerId === "number") {
            window.clearTimeout(timerId);
            delete volumeCommitTimersRef.current[trackId];
        }
    }

    function scheduleVolumeCommit(trackId: string, nextVolume: number) {
        clearPendingVolumeCommit(trackId);
        const timerId = window.setTimeout(() => {
            delete volumeCommitTimersRef.current[trackId];
            onVolumeCommit(trackId, nextVolume);
        }, TRACK_GAIN_WHEEL_COMMIT_DEBOUNCE_MS);
        volumeCommitTimersRef.current[trackId] = timerId;
    }

    useEffect(() => {
        rowHeightRef.current = rowHeight;
    }, [rowHeight]);

    useLayoutEffect(() => {
        const el = listRef.current;
        const pending = pendingVerticalZoomRef.current;
        if (!el || !pending) return;
        if (Math.abs(pending.nextRowHeight - rowHeight) > 1e-9) return;

        pendingVerticalZoomRef.current = null;
        const maxScrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
        const nextScrollTop = Math.max(0, Math.min(maxScrollTop, pending.nextScrollTop));
        el.scrollTop = nextScrollTop;
        onScrollTopChange?.(nextScrollTop);
    }, [rowHeight, onScrollTopChange]);

    function isEditableTarget(target: EventTarget | null) {
        const el = target as HTMLElement | null;
        if (!el) return false;
        const tag = (el.tagName ?? "").toLowerCase();
        if (tag === "input" || tag === "textarea" || tag === "select") return true;
        if (el.isContentEditable) return true;
        if (el.closest?.('input,textarea,select,[contenteditable="true"]')) return true;
        return false;
    }

    function isArrowNavigationKey(key: string): boolean {
        return (
            key === "arrowup" || key === "arrowdown" || key === "arrowleft" || key === "arrowright"
        );
    }

    function startPanPointerLocal(e: React.PointerEvent) {
        // Intercept middle-button mouse to prevent browser native autoscroll
        if (e.pointerType === "mouse" && e.button === 1) {
            e.preventDefault();
            e.stopPropagation();
        }
        if (e.pointerType !== "mouse") return;
        if (e.button !== 1) return;
        if (isEditableTarget(e.target)) return;
        const el = listRef.current;
        if (!el) return;

        panRef.current = {
            pointerId: e.pointerId,
            startY: e.clientY,
            scrollTop: el.scrollTop,
        };

        const prevCursor = document.body.style.cursor;
        const prevSelect = document.body.style.userSelect;
        document.body.style.cursor = "grabbing";
        document.body.style.userSelect = "none";

        try {
            (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
        } catch {}

        function onMove(ev: PointerEvent) {
            const pan = panRef.current;
            const cur = listRef.current;
            if (!pan || !cur) return;
            if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
            cur.scrollTop = pan.scrollTop - (ev.clientY - pan.startY);
            onScrollTopChange?.(cur.scrollTop);
        }

        function end(ev: PointerEvent) {
            const pan = panRef.current;
            if (!pan) return;
            if (pan.pointerId != null && ev.pointerId !== pan.pointerId) return;
            panRef.current = null;
            document.body.style.cursor = prevCursor;
            document.body.style.userSelect = prevSelect;
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", end);
            window.removeEventListener("pointercancel", end);
        }

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", end);
        window.addEventListener("pointercancel", end);
    }

    useEffect(() => {
        const el = listRef.current;
        if (!el) return;

        const handler: EventListener = (evt) => {
            const e = evt as WheelEvent;
            const volumeKnobEl = (e.target as HTMLElement | null)?.closest(
                "[data-track-volume-knob]",
            ) as HTMLElement | null;
            if (volumeKnobEl) {
                const trackId = volumeKnobEl.dataset.trackId;
                if (trackId) {
                    const currentVolume = currentTrackVolumeById[trackId];
                    if (Number.isFinite(currentVolume)) {
                        const rawDelta =
                            Math.abs(e.deltaY) >= Math.abs(e.deltaX) ? e.deltaY : e.deltaX;
                        if (Number.isFinite(rawDelta) && Math.abs(rawDelta) >= 0.01) {
                            const direction = rawDelta < 0 ? 1 : -1;
                            const notches = Math.max(1, Math.round(Math.abs(rawDelta) / 100));
                            const wheelStepDb = isModifierActive(paramFineAdjustKb, e)
                                ? TRACK_GAIN_WHEEL_FINE_STEP_DB
                                : TRACK_GAIN_WHEEL_STEP_DB;
                            const currentDb = gainToDb(currentVolume);
                            const nextDb = Math.max(
                                TRACK_GAIN_MIN_DB,
                                Math.min(
                                    TRACK_GAIN_MAX_DB,
                                    currentDb + direction * wheelStepDb * notches,
                                ),
                            );
                            const nextVolume = dbToGain(nextDb);
                            onVolumeUiChange(trackId, nextVolume);
                            scheduleVolumeCommit(trackId, nextVolume);
                            e.preventDefault();
                            e.stopPropagation();
                            return;
                        }
                    }
                }
            }

            const noModifierPressed = !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey;
            const verticalZoomRequested = (() => {
                if (!verticalZoomKb) return false;
                if (isNoneBinding(verticalZoomKb)) return noModifierPressed;
                return isModifierActive(verticalZoomKb, e);
            })();

            if (verticalZoomRequested && setRowHeight) {
                e.preventDefault();
                const factor = e.deltaY < 0 ? 1.1 : 0.9;
                const baseRowHeight =
                    pendingVerticalZoomRef.current?.nextRowHeight ?? rowHeightRef.current;
                const bounds = el.getBoundingClientRect();
                const pointerY = Math.max(
                    0,
                    Math.min(Math.max(1, bounds.height), e.clientY - bounds.top),
                );
                const rowUnitAtPointer = (el.scrollTop + pointerY) / Math.max(1e-9, baseRowHeight);
                const nextRowHeight = Math.round(
                    Math.max(MIN_ROW_HEIGHT, Math.min(MAX_ROW_HEIGHT, baseRowHeight * factor)),
                );
                if (Math.abs(nextRowHeight - baseRowHeight) < 1e-9) {
                    return;
                }

                pendingVerticalZoomRef.current = {
                    nextRowHeight,
                    nextScrollTop: Math.max(0, rowUnitAtPointer * nextRowHeight - pointerY),
                };
                setRowHeight(nextRowHeight);
                return;
            }

            const useY = Math.abs(e.deltaY) >= Math.abs(e.deltaX);
            const delta = useY ? e.deltaY : e.deltaX;
            if (!Number.isFinite(delta) || Math.abs(delta) < 0.01) return;

            const maxScrollTop = Math.max(0, el.scrollHeight - el.clientHeight);
            if (maxScrollTop <= 0) return;

            const nextScrollTop = Math.max(0, Math.min(maxScrollTop, el.scrollTop + delta));
            if (Math.abs(nextScrollTop - el.scrollTop) < 0.5) return;

            e.preventDefault();
            el.scrollTop = nextScrollTop;
            onScrollTopChange?.(nextScrollTop);
        };

        el.addEventListener("wheel", handler, {
            passive: false,
        } as AddEventListenerOptions);
        return () => {
            el.removeEventListener("wheel", handler);
        };
    }, [
        currentTrackVolumeById,
        onScrollTopChange,
        onVolumeCommit,
        onVolumeUiChange,
        paramFineAdjustKb,
        setRowHeight,
        verticalZoomKb,
    ]);

    function wouldCreateCycle(trackId: string, parentTrackId: string | null) {
        let cur = parentTrackId;
        let guard = 0;
        while (cur && guard++ < 1000) {
            if (cur === trackId) return true;
            cur = parentById.get(cur) ?? null;
        }
        return false;
    }

    function siblingsOf(parentTrackId: string | null): string[] {
        const out: string[] = [];
        for (const tr of tracks) {
            if ((tr.parentId ?? null) === parentTrackId) out.push(tr.id);
        }
        return out;
    }

    function trackAtClientY(clientY: number): {
        track: TrackInfo | null;
        yInRow: number;
        index: number;
    } {
        const el = listRef.current;
        if (!el) return { track: null, yInRow: 0, index: -1 };
        const bounds = el.getBoundingClientRect();
        const y = clientY - bounds.top + el.scrollTop;
        const rawIdx = Math.floor(y / rowHeight);
        // 当鼠标在列表上方时，clamp 到第一个轨道（yInRow=0），使拖拽可以插入到顶部
        if (rawIdx < 0) {
            return { track: tracks[0] ?? null, yInRow: 0, index: 0 };
        }
        const yInRow = y - rawIdx * rowHeight;
        if (rawIdx >= tracks.length) return { track: null, yInRow, index: rawIdx };
        return { track: tracks[rawIdx] ?? null, yInRow, index: rawIdx };
    }

    function beginVolumeKnobDrag(
        e: React.PointerEvent<HTMLButtonElement>,
        trackId: string,
        volume: number,
    ) {
        e.preventDefault();
        e.stopPropagation();
        clearPendingVolumeCommit(trackId);

        const startY = e.clientY;
        const startDb = clampGainDb(gainToDb(volume));
        let lastDb = startDb;
        const fineAxisState: FineAxisDragState = {
            raw: startY,
            adjusted: startY,
            fineActive: isModifierActive(paramFineAdjustKb, e.nativeEvent),
        };

        const onMove = (ev: PointerEvent) => {
            const adjustedY = advanceFineAxisDrag(
                fineAxisState,
                ev.clientY,
                isModifierActive(paramFineAdjustKb, ev),
            );
            const deltaY = startY - adjustedY;
            const nextDb = clampGainDb(startDb + deltaY * TRACK_GAIN_DRAG_DB_PER_PX);
            if (Math.abs(nextDb - lastDb) < 0.01) return;
            lastDb = nextDb;
            onVolumeUiChange(trackId, dbToGain(nextDb));
        };

        const onEnd = () => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onEnd);
            window.removeEventListener("pointercancel", onEnd);
            onVolumeCommit(trackId, dbToGain(lastDb));
        };

        window.addEventListener("pointermove", onMove);
        window.addEventListener("pointerup", onEnd);
        window.addEventListener("pointercancel", onEnd);
    }

    function computeDropSpec(
        draggingTrackId: string,
        clientX: number,
        clientY: number,
    ): {
        parentTrackId: string | null;
        targetIndex: number;
        mode: "reorder" | "nest";
    } {
        const el = listRef.current;
        const bounds = el?.getBoundingClientRect();

        // 当鼠标在列表容器上方时，直接插入到顶层第一个位�?
        if (bounds && clientY < bounds.top && tracks.length > 0) {
            return {
                parentTrackId: null,
                targetIndex: 0,
                mode: "reorder",
            };
        }

        const { track: over, yInRow } = trackAtClientY(clientY);

        // Dropping outside -> append as root.
        if (!over) {
            const roots = siblingsOf(null).filter((id) => id !== draggingTrackId);
            return {
                parentTrackId: null,
                targetIndex: roots.length,
                mode: "reorder",
            };
        }

        const overIndent = Math.max(0, (over.depth ?? 0) * 16);
        const localX = bounds ? clientX - bounds.left : clientX;
        const nest = over.id !== draggingTrackId && localX > 24 + overIndent + 40;

        if (nest) {
            const parentTrackId = over.id;
            if (wouldCreateCycle(draggingTrackId, parentTrackId)) {
                const roots = siblingsOf(null).filter((id) => id !== draggingTrackId);
                return {
                    parentTrackId: null,
                    targetIndex: roots.length,
                    mode: "reorder",
                };
            }
            const children = siblingsOf(parentTrackId).filter((id) => id !== draggingTrackId);
            return {
                parentTrackId,
                targetIndex: children.length,
                mode: "nest",
            };
        }

        let parentTrackId = over.parentId ?? null;
        if (wouldCreateCycle(draggingTrackId, parentTrackId)) {
            parentTrackId = null;
        }

        if (over.id === draggingTrackId) {
            const siblingsIncl = siblingsOf(parentTrackId);
            const indexSelf = Math.max(0, siblingsIncl.indexOf(draggingTrackId));
            return { parentTrackId, targetIndex: indexSelf, mode: "reorder" };
        }

        const siblings = siblingsOf(parentTrackId).filter((id) => id !== draggingTrackId);
        const baseIndex = Math.max(0, siblings.indexOf(over.id));
        // 使用 35% 边缘区域：上 35% 插入到上方，�?35% 插入到下方，中间 30% 保持不动
        const edgeZone = rowHeight * 0.35;
        const insertAfter = yInRow > rowHeight - edgeZone;
        const insertBefore = yInRow < edgeZone;
        // 如果鼠标在中间区域，保持原位不触发重�?
        if (!insertAfter && !insertBefore) {
            const siblingsIncl = siblingsOf(parentTrackId);
            const indexSelf = Math.max(0, siblingsIncl.indexOf(draggingTrackId));
            // 如果不在同一层级，则追加到末�?
            const targetIndex = indexSelf >= 0 ? indexSelf : siblings.length;
            return { parentTrackId, targetIndex, mode: "reorder" };
        }
        const targetIndex = Math.min(siblings.length, baseIndex + (insertAfter ? 1 : 0));
        return { parentTrackId, targetIndex, mode: "reorder" };
    }

    return (
        <Flex direction="column" className="w-64 border-r border-qt-border bg-qt-window shrink-0">
            <Box className="h-6 border-b border-qt-border px-2 flex items-center bg-qt-window shadow-sm z-10">
                <Text size="1" weight="bold" color="gray">
                    {t("tracks")}
                </Text>
            </Box>
            <div
                ref={(el) => {
                    (listRef as React.MutableRefObject<HTMLDivElement | null>).current = el;
                    if (listScrollRef) listScrollRef.current = el;
                }}
                data-track-list-panel
                onFocusCapture={() => {
                    document.body.setAttribute("data-hs-focus-window", "trackHeader");
                }}
                onMouseDownCapture={() => {
                    document.body.setAttribute("data-hs-focus-window", "trackHeader");
                }}
                onPointerDown={(e) => {
                    document.body.setAttribute("data-hs-focus-window", "trackHeader");
                    startPanPointerLocal?.(e);
                }}
                onKeyDownCapture={(e) => {
                    if (isEditableTarget(e.target)) return;
                    const key = e.key.toLowerCase();
                    if (isArrowNavigationKey(key)) {
                        e.preventDefault();
                    }
                }}
                onAuxClick={(e) => {
                    // Prevent native autoscroll overlay on middle click
                    e.preventDefault();
                    e.stopPropagation();
                }}
                onMouseDown={(e) => {
                    if (e.button === 1) {
                        e.preventDefault();
                        e.stopPropagation();
                    }
                }}
                className="flex-1 relative overflow-y-auto custom-scrollbar hide-v-scrollbar"
                onScroll={(e) => {
                    onScrollTopChange?.((e.currentTarget as HTMLDivElement).scrollTop);
                }}
            >
                {dragUi?.mode === "reorder" && typeof dragUi.indicatorY === "number" ? (
                    <div
                        className="absolute left-1 right-1 pointer-events-none z-50"
                        style={{ top: dragUi.indicatorY }}
                    >
                        <div className="h-px bg-qt-highlight" />
                    </div>
                ) : null}
                {tracks.map((track) => {
                    const selected = selectedTrackId === track.id;
                    const depth = Math.max(0, Number(track.depth ?? 0) || 0);
                    const indent = depth * 16;
                    const dragging = dragUi?.draggingTrackId === track.id;
                    const isOver = dragUi?.overTrackId === track.id;
                    const muted = Boolean(track.muted);
                    const solo = Boolean(track.solo);
                    const isRoot = (track.parentId ?? null) == null;
                    const composeEnabled = Boolean(track.composeEnabled);
                    const volume = currentTrackVolumeById[track.id] ?? 1;
                    const meter = meterDisplayByTrackId[track.id];
                    const peakLinear = meter?.peakLinear ?? 0;
                    const maxPeakLinear = meter?.maxPeakLinear ?? 0;
                    const clipped = Boolean(meter?.clipped);
                    const volumeDb = clampGainDb(gainToDb(volume));
                    const knobDeg =
                        volumeDb >= 0
                            ? (volumeDb / TRACK_GAIN_MAX_DB) * 135
                            : (volumeDb / Math.abs(TRACK_GAIN_MIN_DB)) * 135;

                    const guideLines = depth > 0 ? Array.from({ length: depth }) : [];

                    return (
                        <div
                            key={track.id}
                            style={{ height: rowHeight }}
                            className="border-b border-qt-border relative group overflow-hidden"
                            onPointerDownCapture={(e) => {
                                if (e.button !== 0) return;
                                if (selectedTrackId === track.id) return;
                                onSelectTrack(track.id);
                            }}
                            onContextMenu={(e) => {
                                e.preventDefault();
                                setTrackCtxMenu({
                                    x: e.clientX,
                                    y: e.clientY,
                                    trackId: track.id,
                                });
                            }}
                            onPointerDown={(e) => {
                                if (e.button !== 0) return;

                                // If the pointer down starts on an interactive control, do not start a drag.
                                const target = e.target as HTMLElement | null;
                                if (
                                    target?.closest?.(
                                        "button,[role='slider'],input,textarea,select,a",
                                    )
                                ) {
                                    return;
                                }

                                const overSiblings = siblingsOf(track.parentId ?? null);
                                const originalIndexSelf = Math.max(
                                    0,
                                    overSiblings.indexOf(track.id),
                                );

                                dragRef.current = {
                                    pointerId: e.pointerId,
                                    trackId: track.id,
                                    startClientX: e.clientX,
                                    startClientY: e.clientY,
                                    hasMoved: false,
                                    originalParentId: track.parentId ?? null,
                                    originalIndexSelf,
                                };

                                const el = e.currentTarget as HTMLDivElement;
                                el.setPointerCapture(e.pointerId);

                                const prevCursor = document.body.style.cursor;
                                const prevSelect = document.body.style.userSelect;

                                function onMove(ev: PointerEvent) {
                                    const drag = dragRef.current;
                                    if (!drag || drag.pointerId !== e.pointerId) return;

                                    if (!drag.hasMoved) {
                                        const dx = ev.clientX - drag.startClientX;
                                        const dy = ev.clientY - drag.startClientY;
                                        if (dx * dx + dy * dy < 9) {
                                            return;
                                        }
                                        drag.hasMoved = true;
                                        document.body.style.cursor = "grabbing";
                                        document.body.style.userSelect = "none";
                                    }

                                    const spec = computeDropSpec(
                                        drag.trackId,
                                        ev.clientX,
                                        ev.clientY,
                                    );
                                    const overInfo = trackAtClientY(ev.clientY);
                                    const over = overInfo.track;

                                    let indicatorY: number | null = null;
                                    if (spec.mode === "reorder") {
                                        const listBounds = listRef.current?.getBoundingClientRect();
                                        // 鼠标在列表上方时，指示线固定在顶�?
                                        if (listBounds && ev.clientY < listBounds.top) {
                                            indicatorY = 0;
                                        } else {
                                            const idx = overInfo.index;
                                            const edgeZone = rowHeight * 0.35;
                                            if (!Number.isFinite(idx)) {
                                                indicatorY = null;
                                            } else if (!over) {
                                                indicatorY = tracks.length * rowHeight;
                                            } else {
                                                const insertAfter =
                                                    overInfo.yInRow > rowHeight - edgeZone;
                                                const insertBefore = overInfo.yInRow < edgeZone;
                                                if (insertAfter) {
                                                    indicatorY = idx * rowHeight + rowHeight;
                                                } else if (insertBefore) {
                                                    indicatorY = idx * rowHeight;
                                                } else {
                                                    // 中间区域不显示指示线
                                                    indicatorY = null;
                                                }
                                            }
                                        }
                                    }

                                    setDragUi({
                                        draggingTrackId: drag.trackId,
                                        overTrackId: over?.id ?? null,
                                        mode: spec.mode,
                                        indicatorY,
                                    });
                                }

                                function end(ev: PointerEvent) {
                                    const drag = dragRef.current;
                                    if (!drag || drag.pointerId !== e.pointerId) return;
                                    dragRef.current = null;

                                    window.removeEventListener("pointermove", onMove);
                                    window.removeEventListener("pointerup", end);
                                    window.removeEventListener("pointercancel", end);

                                    document.body.style.cursor = prevCursor;
                                    document.body.style.userSelect = prevSelect;

                                    const moved = drag.hasMoved;
                                    setDragUi(null);

                                    if (!moved) {
                                        return;
                                    }

                                    const spec = computeDropSpec(
                                        drag.trackId,
                                        ev.clientX,
                                        ev.clientY,
                                    );

                                    if (
                                        spec.parentTrackId === drag.originalParentId &&
                                        spec.targetIndex === drag.originalIndexSelf
                                    ) {
                                        return;
                                    }

                                    onMoveTrack({
                                        trackId: drag.trackId,
                                        targetIndex: spec.targetIndex,
                                        parentTrackId: spec.parentTrackId,
                                    });
                                }

                                window.addEventListener("pointermove", onMove);
                                window.addEventListener("pointerup", end);
                                window.addEventListener("pointercancel", end);
                            }}
                        >
                            {/* Always-visible left accent bar (pinned to list edge) */}
                            <div
                                className={`absolute left-0 top-0 bottom-0 w-1 transition-opacity ${selected ? "opacity-100" : "opacity-80 group-hover:opacity-90"}`}
                                style={{
                                    backgroundColor: track.color || "var(--qt-highlight)",
                                }}
                            />

                            {/* Left gutter: makes nesting depth visible at a glance */}
                            <div
                                className="absolute left-0 top-0 bottom-0 bg-qt-window pointer-events-none"
                                style={{ width: indent }}
                            >
                                {guideLines.map((_, i) => (
                                    <div
                                        key={i}
                                        className="absolute top-0 bottom-0 border-l border-qt-border opacity-60"
                                        style={{ left: i * 16 + 8 }}
                                    />
                                ))}
                                {depth > 0 ? (
                                    <div
                                        className="absolute border-t border-qt-border opacity-60"
                                        style={{
                                            left: (depth - 1) * 16 + 8,
                                            right: 0,
                                            top: "50%",
                                        }}
                                    />
                                ) : null}
                            </div>

                            {/* Content block: shifted right by depth */}
                            <Box
                                className={`absolute top-0 bottom-0 right-0 bg-qt-base transition-colors overflow-hidden ${selected ? "bg-qt-button-hover" : "hover:bg-qt-button-hover"} ${dragging ? "opacity-60" : ""} ${isOver ? "bg-qt-button-hover" : ""}`}
                                style={{ left: indent }}
                            >
                                {/* Keep a subtle in-row bar too, but don't rely on it */}
                                <div
                                    className={`absolute left-0 top-0 bottom-0 w-1 transition-opacity ${selected ? "opacity-100" : "opacity-10 group-hover:opacity-30"}`}
                                    style={{
                                        backgroundColor: track.color || "var(--qt-highlight)",
                                    }}
                                />

                                {isOver && dragUi?.mode === "nest" ? (
                                    <div
                                        className="absolute inset-0 pointer-events-none"
                                        style={{
                                            backgroundColor:
                                                "color-mix(in oklab, var(--qt-highlight) 14%, transparent)",
                                            border: "1px dashed var(--qt-highlight)",
                                        }}
                                    />
                                ) : null}

                                <Flex height="100%" align="stretch">
                                    <Flex
                                        direction="column"
                                        p="2"
                                        gap="2"
                                        justify="center"
                                        className="min-w-0 flex-1"
                                    >
                                        <Flex justify="between" align="center" gap="2">
                                            <Flex align="center" gap="1" className="min-w-0 flex-1">
                                                {/* ??????????????? */}
                                                <div
                                                    className="relative shrink-0"
                                                    data-track-color-picker
                                                >
                                                    <button
                                                        className="w-3.5 h-3.5 rounded-full border border-white/20 hover:scale-125 transition-transform cursor-pointer"
                                                        style={{
                                                            backgroundColor:
                                                                track.color || "#4f8ef7",
                                                        }}
                                                        title={t("track_change_color")}
                                                        onPointerDown={(e) => e.stopPropagation()}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            setColorPickerTrackId(
                                                                colorPickerTrackId === track.id
                                                                    ? null
                                                                    : track.id,
                                                            );
                                                        }}
                                                    />
                                                    {colorPickerTrackId === track.id && (
                                                        <div
                                                            className="absolute left-0 top-full mt-1 z-50 p-1.5 rounded border border-qt-border bg-qt-window shadow-lg flex gap-1 flex-wrap"
                                                            style={{
                                                                width: 120,
                                                            }}
                                                            data-track-color-picker
                                                        >
                                                            {TRACK_COLOR_PALETTE_KEYS.map((opt) => (
                                                                <button
                                                                    key={opt.value}
                                                                    title={t(opt.key)}
                                                                    className={`w-4 h-4 rounded-full transition-transform hover:scale-125 ${
                                                                        (track.color ||
                                                                            "#4f8ef7") === opt.value
                                                                            ? "ring-2 ring-white/80 scale-110"
                                                                            : ""
                                                                    }`}
                                                                    style={{
                                                                        backgroundColor: opt.value,
                                                                    }}
                                                                    onPointerDown={(e) =>
                                                                        e.stopPropagation()
                                                                    }
                                                                    onClick={(e) => {
                                                                        e.stopPropagation();
                                                                        onTrackColorChange?.(
                                                                            track.id,
                                                                            opt.value,
                                                                        );
                                                                        setColorPickerTrackId(null);
                                                                    }}
                                                                />
                                                            ))}
                                                        </div>
                                                    )}
                                                </div>
                                                {editingTrackId === track.id ? (
                                                    <input
                                                        ref={nameInputRef}
                                                        value={editingName}
                                                        className="bg-transparent outline outline-1 outline-qt-highlight rounded px-0.5 flex-1 min-w-0 text-qt-text text-sm font-medium pr-2"
                                                        onChange={(e) =>
                                                            setEditingName(e.target.value)
                                                        }
                                                        onBlur={commitTrackName}
                                                        onKeyDown={(e) => {
                                                            if (e.key === "Enter") {
                                                                commitTrackName();
                                                            } else if (e.key === "Escape") {
                                                                setEditingTrackId(null);
                                                            }
                                                        }}
                                                        onPointerDown={(e) => e.stopPropagation()}
                                                        onClick={(e) => e.stopPropagation()}
                                                        autoFocus
                                                    />
                                                ) : (
                                                    <Text
                                                        size="2"
                                                        weight="medium"
                                                        className={`text-qt-text truncate pr-2 ${depth > 0 ? "opacity-90" : ""} cursor-text select-none`}
                                                        onPointerDown={(e) => e.stopPropagation()}
                                                        onDoubleClick={(e) => {
                                                            e.stopPropagation();
                                                            setEditingTrackId(track.id);
                                                            setEditingName(track.name);
                                                            setTimeout(() => {
                                                                nameInputRef.current?.select();
                                                            }, 0);
                                                        }}
                                                    >
                                                        {track.name}
                                                    </Text>
                                                )}
                                            </Flex>
                                            {isRoot && composeEnabled && onAlgoChange ? (
                                                <div
                                                    className="shrink-0"
                                                    onPointerDown={(e) => e.stopPropagation()}
                                                >
                                                    <Select.Root
                                                        size="1"
                                                        value={
                                                            PITCH_ANALYSIS_ALGO_OPTIONS.includes(
                                                                track.pitchAnalysisAlgo as
                                                                    | "world_dll"
                                                                    | "nsf_hifigan_onnx"
                                                                    | "vslib"
                                                                    | "none",
                                                            )
                                                                ? track.pitchAnalysisAlgo
                                                                : "nsf_hifigan_onnx"
                                                        }
                                                        onValueChange={(v) => {
                                                            onAlgoChange(track.id, v);
                                                        }}
                                                    >
                                                        <Select.Trigger
                                                            style={{
                                                                minWidth: 80,
                                                            }}
                                                        />
                                                        <Select.Content>
                                                            <Select.Item value="world_dll">
                                                                world
                                                            </Select.Item>
                                                            <Select.Item value="nsf_hifigan_onnx">
                                                                nsf-hifigan
                                                            </Select.Item>
                                                            <Select.Item value="vslib">
                                                                vslib
                                                            </Select.Item>
                                                            <Select.Item value="none">
                                                                {t("none")}
                                                            </Select.Item>
                                                        </Select.Content>
                                                    </Select.Root>
                                                </div>
                                            ) : null}
                                            <IconButton
                                                size="1"
                                                variant="ghost"
                                                color="gray"
                                                className="opacity-0 group-hover:opacity-100"
                                                disabled={isLastRootTrack(track.id)}
                                                onPointerDown={(e) => e.stopPropagation()}
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onRemoveTrack(track.id);
                                                }}
                                            >
                                                <Cross2Icon />
                                            </IconButton>
                                        </Flex>

                                        <div
                                            className="min-w-0 pt-1"
                                            data-track-volume-control
                                            data-track-id={track.id}
                                            onPointerDown={(e) => e.stopPropagation()}
                                            onClick={(e) => e.stopPropagation()}
                                            onDoubleClick={(e) => {
                                                e.stopPropagation();
                                                clearPendingVolumeCommit(track.id);
                                                onVolumeUiChange(track.id, 1);
                                                onVolumeCommit(track.id, 1);
                                            }}
                                        >
                                            <Flex align="center" gap="2">
                                                <button
                                                    type="button"
                                                    className="relative w-8 h-8 rounded-full border border-qt-border bg-qt-window hover:bg-qt-surface transition-colors shrink-0"
                                                    title={formatGainLabel(volume)}
                                                    data-track-volume-knob
                                                    data-track-id={track.id}
                                                    onPointerDown={(e) =>
                                                        beginVolumeKnobDrag(e, track.id, volume)
                                                    }
                                                >
                                                    <span
                                                        className="absolute left-1/2 top-1/2 w-[2px] h-3 -translate-x-1/2 -translate-y-full rounded-full bg-qt-highlight"
                                                        style={{
                                                            transform: `translate(-50%, -100%) rotate(${knobDeg}deg)`,
                                                            transformOrigin: "50% 100%",
                                                        }}
                                                    />
                                                </button>
                                                <Text
                                                    size="1"
                                                    color={
                                                        Math.abs(gainToDb(volume)) < 0.05
                                                            ? "blue"
                                                            : "gray"
                                                    }
                                                    className="leading-none tabular-nums"
                                                >
                                                    {formatGainLabel(volume)}
                                                </Text>
                                            </Flex>
                                        </div>
                                    </Flex>

                                    <Flex
                                        direction="column"
                                        gap="1"
                                        align="center"
                                        justify="center"
                                        className="w-[22px] shrink-0"
                                    >
                                        {isRoot ? (
                                            <IconButton
                                                size="1"
                                                variant={composeEnabled ? "solid" : "ghost"}
                                                color={composeEnabled ? "blue" : "gray"}
                                                title={t("compose")}
                                                onPointerDown={(e) => e.stopPropagation()}
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    onToggleCompose(track.id, !composeEnabled);
                                                }}
                                                style={{
                                                    fontWeight: 700,
                                                    fontSize: 11,
                                                    width: 20,
                                                    height: 20,
                                                }}
                                            >
                                                C
                                            </IconButton>
                                        ) : (
                                            <div className="w-5 h-5" />
                                        )}
                                        <IconButton
                                            size="1"
                                            variant={muted ? "solid" : "ghost"}
                                            color={muted ? "red" : "gray"}
                                            title={muted ? t("clip_unmute") : t("clip_mute")}
                                            onPointerDown={(e) => e.stopPropagation()}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onToggleMute(track.id, !muted);
                                            }}
                                            style={{
                                                fontWeight: 700,
                                                fontSize: 11,
                                                width: 20,
                                                height: 20,
                                            }}
                                        >
                                            M
                                        </IconButton>
                                        <IconButton
                                            size="1"
                                            variant={solo ? "solid" : "ghost"}
                                            color={solo ? "amber" : "gray"}
                                            title={t("solo")}
                                            onPointerDown={(e) => e.stopPropagation()}
                                            onClick={(e) => {
                                                e.stopPropagation();
                                                onToggleSolo(track.id, !solo);
                                            }}
                                            style={{
                                                fontWeight: 700,
                                                fontSize: 11,
                                                width: 20,
                                                height: 20,
                                            }}
                                        >
                                            S
                                        </IconButton>
                                    </Flex>

                                    <div
                                        className="w-[11.25%] min-w-[28px] max-w-[34px] shrink-0"
                                        style={{
                                            background: "var(--qt-meter-rail)",
                                        }}
                                    >
                                        <Flex
                                            direction="column"
                                            align="center"
                                            justify="between"
                                            className="h-full pt-1 pb-0"
                                        >
                                            <Text
                                                size="1"
                                                color={clipped ? "red" : "gray"}
                                                className="leading-none tabular-nums"
                                            >
                                                {formatPeakLabel(maxPeakLinear, clipped)}
                                            </Text>
                                            <div
                                                className="relative h-full w-full"
                                                style={{
                                                    background: "var(--qt-meter-well)",
                                                }}
                                            >
                                                <div
                                                    className={`absolute inset-x-0 bottom-0 transition-[height] duration-75 ${meterFillClass(
                                                        peakLinear,
                                                        clipped,
                                                    )}`}
                                                    style={{
                                                        height: `${meterHeightPercent(peakLinear)}%`,
                                                        maxHeight: "100%",
                                                    }}
                                                />
                                            </div>
                                        </Flex>
                                    </div>
                                </Flex>
                            </Box>
                        </div>
                    );
                })}

                <Flex
                    align="center"
                    justify="center"
                    className="h-8 border-b border-qt-border border-dashed text-qt-text-muted hover:text-qt-text hover:bg-qt-button-hover cursor-pointer transition-colors"
                    style={{ height: TRACK_ADD_ROW_HEIGHT }}
                    onClick={onAddTrack}
                >
                    <PlusIcon className="mr-1" /> <Text size="1">{t("track_add")}</Text>
                </Flex>
            </div>

            {/* 轨道右键菜单 */}
            {trackCtxMenu && (
                <div
                    ref={trackCtxMenuRef}
                    data-track-ctx-menu
                    data-hs-context-menu="1"
                    className="fixed z-50 min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
                    style={{ left: trackCtxMenu.x, top: trackCtxMenu.y }}
                    onPointerDown={(e) => e.stopPropagation()}
                >
                    <button
                        className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors"
                        onClick={() => {
                            onDuplicateTrack?.(trackCtxMenu.trackId);
                            setTrackCtxMenu(null);
                        }}
                    >
                        {t("track_clone")}
                    </button>
                    <button
                        className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors text-red-400 hover:text-red-300"
                        disabled={isLastRootTrack(trackCtxMenu.trackId)}
                        onClick={() => {
                            onRemoveTrack(trackCtxMenu.trackId);
                            setTrackCtxMenu(null);
                        }}
                    >
                        {t("ctx_delete")}
                    </button>
                </div>
            )}
        </Flex>
    );
};
