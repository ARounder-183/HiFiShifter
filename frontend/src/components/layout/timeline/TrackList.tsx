import React, { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { Flex, Box, Text, IconButton, Select } from "@radix-ui/themes";
import { Cross2Icon, PlusIcon } from "@radix-ui/react-icons";
import { shallowEqual } from "react-redux";
import type { TrackInfo, TrackMeterInfo } from "../../../features/session/sessionTypes";
import {
    isNoneBinding,
    isModifierActive,
    selectKeybinding,
    formatKeybinding,
} from "../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../features/keybindings/types";
import type { MessageKey } from "../../../i18n/messages";
import { useAppSelector } from "../../../app/hooks";
import { useVisualPlayhead } from "../../../hooks/useVisualPlayhead";
import { formatCursorTime } from "./timeFormat";
import type { TimeFormatContext } from "./timeFormat";
import { MAX_ROW_HEIGHT, MIN_ROW_HEIGHT, TRACK_ADD_ROW_HEIGHT } from "./constants";
import { advanceFineAxisDrag, type FineAxisDragState } from "./fineAxisDrag";
import { AppTooltipBubble } from "../../AppTooltip";
import { TempoMapCornerButton } from "./TempoMapCornerButton";
import { formatGainDbValue } from "./math";
import { computeVisibleTrackWindow } from "./runtime/timelineWindowing";
import { normalizedTrackColorCss } from "./runtime/timelineCanvasStyle";
import { useAppTheme } from "../../../theme/AppThemeProvider";

/** Color palette options shown when creating a new track.
 * 色值选取与归一化带（s 0.30-0.46、感知亮度 0.50-0.60）对齐：暖色系
 * （橙/黄/红）取更亮的原色，避免亮度归一化把它们抬成"洗白"的粉调；
 * 灰色为默认轨道色。 */
const TRACK_COLOR_PALETTE_KEYS: { value: string; key: MessageKey }[] = [
    { value: "#74787e", key: "color_gray" },
    { value: "#4a8fd1", key: "color_blue" },
    { value: "#7b6bc4", key: "color_purple" },
    { value: "#43a875", key: "color_green" },
    { value: "#d68a52", key: "color_orange" },
    { value: "#d982a8", key: "color_pink" },
    { value: "#bb5fae", key: "color_magenta" },
    { value: "#d4bc55", key: "color_yellow" },
    { value: "#cf5252", key: "color_red" },
];
const PITCH_ANALYSIS_ALGO_OPTIONS = ["world_dll", "nsf_hifigan_onnx", "vslib", "none"] as const;

function splitDigitRuns(text: string): Array<{ text: string; digits: boolean }> {
    const parts: Array<{ text: string; digits: boolean }> = [];
    let current = "";
    let currentDigits = false;
    const flush = () => {
        if (current.length > 0) {
            parts.push({ text: current, digits: currentDigits });
            current = "";
        }
    };
    for (const ch of text) {
        const isDigit = ch >= "0" && ch <= "9";
        if (isDigit !== currentDigits) {
            flush();
            currentDigits = isDigit;
        }
        current += ch;
    }
    flush();
    return parts;
}

/**
 * 将时间文本中的每个数字放入固定宽度的槽位（宽度按当前字体测量），
 * 非数字字符保持自然宽度。这样即使使用不等宽字体，数字变化时文本也不会左右晃动。
 */
const SlotTimeText = React.memo(function SlotTimeText({
    text,
    digitWidthPx,
    className,
    style,
    tooltip,
    selectable = false,
}: {
    text: string;
    digitWidthPx: number | null;
    className?: string;
    style?: React.CSSProperties;
    tooltip?: string;
    /** 允许用户选择这段文本（显式覆盖全局 DAW 禁用文本选择的策略） */
    selectable?: boolean;
}) {
    const parts = React.useMemo(() => splitDigitRuns(text), [text]);
    return (
        <Text
            size="2"
            weight="medium"
            className={selectable ? `${className ?? ""} cursor-text` : className}
            style={
                selectable
                    ? {
                          ...style,
                          userSelect: "text",
                          WebkitUserSelect: "text",
                      }
                    : style
            }
            data-tooltip={tooltip}
            data-hs-selectable={selectable ? "true" : undefined}
            onDoubleClick={
                selectable
                    ? (event) => {
                          // 文本被拆成了多个数字槽位 span，原生双击只会选中一个词；
                          // 这里显式选中整个时间字符串，保证"双击全选"在 WebView 中稳定工作。
                          event.preventDefault();
                          event.stopPropagation();
                          const selection = window.getSelection();
                          if (!selection) return;
                          const range = document.createRange();
                          range.selectNodeContents(event.currentTarget);
                          selection.removeAllRanges();
                          selection.addRange(range);
                      }
                    : undefined
            }
        >
            {parts.map((part, index) =>
                part.digits && digitWidthPx != null && digitWidthPx > 0 ? (
                    <React.Fragment key={index}>
                        {part.text.split("").map((ch, chIndex) => (
                            <span
                                key={chIndex}
                                style={{
                                    display: "inline-block",
                                    width: digitWidthPx,
                                    textAlign: "center",
                                }}
                            >
                                {ch}
                            </span>
                        ))}
                    </React.Fragment>
                ) : (
                    <span key={index}>{part.text}</span>
                ),
            )}
        </Text>
    );
});

const TrackHeaderPlayheadTime = React.memo(function TrackHeaderPlayheadTime() {
    const selector = useAppSelector(
        (state) => ({
            playheadSec: state.session.playheadSec,
            isPlaying: state.session.runtime.isPlaying,
            playbackPositionSec: state.session.runtime.playbackPositionSec,
            primaryTimeUnit: state.session.primaryTimeUnit,
            secondaryTimeUnit: state.session.secondaryTimeUnit,
            bpm: state.session.bpm,
            beats: state.session.beats,
            grid: state.session.grid,
            projectSec: state.session.projectSec,
            show: state.session.showPlayheadTimeInTrackHeader,
            tempoMap: state.session.tempoMap,
        }),
        shallowEqual,
    );
    const [visualSec, setVisualSec] = useState(selector.playheadSec);
    const isTransportAdvancing = selector.isPlaying && selector.playbackPositionSec > 1e-4;
    useVisualPlayhead({
        syncedPlayheadSec: selector.playheadSec,
        isTransportAdvancing,
        onFrame: React.useCallback((sec: number) => setVisualSec(sec), []),
    });
    const timeContext = React.useMemo<TimeFormatContext>(
        () => ({
            bpm: selector.bpm,
            beatsPerBar: Math.max(1, Math.round(selector.beats || 4)),
            grid: selector.grid,
            tempoMap: selector.tempoMap,
        }),
        [selector.bpm, selector.beats, selector.grid, selector.tempoMap],
    );
    const formatted = React.useMemo(
        () =>
            formatCursorTime(
                selector.primaryTimeUnit,
                selector.secondaryTimeUnit,
                visualSec,
                timeContext,
            ),
        [selector.primaryTimeUnit, selector.secondaryTimeUnit, visualSec, timeContext],
    );
    // 为光标时间文本预留固定宽度，并把每个数字放进等宽槽位：
    // 宽度按“当前时间单位下的最长可能文本”用真实字体度量，字体仍完全跟随用户自定义字体。
    const maxLabelSpanRef = useRef<HTMLSpanElement | null>(null);
    const digitProbeRef = useRef<HTMLSpanElement | null>(null);
    const [boxWidth, setBoxWidth] = useState<number | null>(null);
    const [digitWidth, setDigitWidth] = useState<number | null>(null);
    const maxLabel = React.useMemo(() => {
        const maxSec = Math.max(1, selector.projectSec, selector.playheadSec);
        return formatCursorTime(
            selector.primaryTimeUnit,
            selector.secondaryTimeUnit,
            maxSec,
            timeContext,
        ).combined;
    }, [
        selector.primaryTimeUnit,
        selector.secondaryTimeUnit,
        selector.projectSec,
        selector.playheadSec,
        timeContext,
    ]);

    const updateDigitWidth = React.useCallback(() => {
        const probe = digitProbeRef.current;
        if (!probe) return;
        const font = window.getComputedStyle(probe).font;
        const probeWidth = probe.getBoundingClientRect().width;
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        if (!ctx) {
            setDigitWidth(probeWidth);
            return;
        }
        ctx.font = font;
        let max = 0;
        for (const digit of "0123456789") {
            max = Math.max(max, ctx.measureText(digit).width);
        }
        // 槽位宽度至少不能小于页面中“0”的真实渲染宽度，避免个别字体下墨迹溢出。
        const slotWidth = Math.max(max, probeWidth);
        setDigitWidth((prev) =>
            prev != null && Math.abs(prev - slotWidth) < 0.01 ? prev : slotWidth,
        );
    }, []);

    useLayoutEffect(() => {
        const el = maxLabelSpanRef.current;
        if (!el) return;
        const update = () => {
            const width = el.getBoundingClientRect().width;
            // 预留缓冲：既避免亚像素/墨迹溢出被裁掉，也让时间文本两侧不显得拥挤。
            const paddedWidth = Math.ceil(width) + 12;
            setBoxWidth((prev) =>
                prev != null && Math.abs(prev - paddedWidth) < 0.01 ? prev : paddedWidth,
            );
        };
        update();
        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(update);
            observer.observe(el);
            return () => observer.disconnect();
        }
    }, [maxLabel]);

    useLayoutEffect(() => {
        const probe = digitProbeRef.current;
        if (!probe) return;
        updateDigitWidth();
        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(updateDigitWidth);
            observer.observe(probe);
            return () => observer.disconnect();
        }
        if (document.fonts?.ready) {
            void document.fonts.ready.then(updateDigitWidth);
        }
    }, [updateDigitWidth]);

    if (!selector.show) return null;
    return (
        <div className="min-w-0 flex-1 flex justify-end" data-playhead-sec={visualSec.toFixed(6)}>
            <span
                ref={maxLabelSpanRef}
                aria-hidden
                className="absolute invisible whitespace-nowrap"
            >
                <SlotTimeText text={maxLabel} digitWidthPx={digitWidth} className="tabular-nums" />
            </span>
            <Text
                ref={digitProbeRef}
                size="2"
                weight="medium"
                aria-hidden
                className="absolute invisible whitespace-nowrap tabular-nums"
            >
                0
            </Text>
            <SlotTimeText
                text={formatted.combined}
                digitWidthPx={digitWidth}
                selectable
                className="tabular-nums text-qt-text text-right leading-none whitespace-nowrap shrink-0"
                style={
                    boxWidth != null ? { minWidth: boxWidth, display: "inline-block" } : undefined
                }
                tooltip={formatted.combined}
            />
        </div>
    );
});

const TRACK_METER_MIN_DB = -48;
const TRACK_METER_MAX_DB = 3;
const TRACK_GAIN_MIN_DB = -60;
const TRACK_GAIN_MAX_DB = 12;
const TRACK_GAIN_WHEEL_STEP_DB = 0.5;
const TRACK_GAIN_WHEEL_FINE_STEP_DB = 0.1;
const TRACK_GAIN_WHEEL_COMMIT_DEBOUNCE_MS = 120;
const TRACK_GAIN_DRAG_DB_PER_PX = 0.2;

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

type TrackListProps = {
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
    /** “复制拖动”修饰键：按住时拖拽轨道头 = 在放置位置克隆轨道。 */
    copyDragKb?: Keybinding;
    onDuplicateTrackTo?: (payload: {
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
    onCreateTrackBelow?: (trackId: string) => void;
    onTrackColorChange?: (trackId: string, color: string) => void;
    onAlgoChange?: (trackId: string, algo: string) => void;
    onTrackNameChange?: (trackId: string, name: string) => void;
    onDuplicateTrack?: (trackId: string) => void;
    onScrollTopChange?: (scrollTop: number) => void;
    /** 外部持有该滚动容器的 ref，用于同步右侧轨道区的竖向滚�?*/
    listScrollRef?: React.MutableRefObject<HTMLDivElement | null>;
    /** 顶部角落高度（与右侧时间标尺对齐；Tempo Map 行可见时增高）。 */
    headerHeight?: number;
};

const TrackListInner: React.FC<TrackListProps> = ({
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
    copyDragKb,
    onDuplicateTrackTo,
    onToggleMute,
    onToggleSolo,
    onToggleCompose,
    onVolumeUiChange,
    onVolumeCommit,
    onAddTrack,
    onCreateTrackBelow,
    onTrackColorChange,
    onAlgoChange,
    onTrackNameChange,
    onDuplicateTrack,
    onScrollTopChange,
    listScrollRef,
    headerHeight = 48,
}) => {
    const listRef = useRef<HTMLDivElement | null>(null);
    // 轨道头色条/取色预览需要和时间线画布同一套主题化轨道色。
    const { mode } = useAppTheme();
    const darkMode = mode === "dark";
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
    const [volumeHoveredTrackId, setVolumeHoveredTrackId] = useState<string | null>(null);
    const [volumeTooltipPos, setVolumeTooltipPos] = useState<{ x: number; y: number } | null>(null);
    const [volumeDrag, setVolumeDrag] = useState<{ trackId: string; baseDb: number } | null>(null);
    const [editingGainTrackId, setEditingGainTrackId] = useState<string | null>(null);
    const [editingGainValue, setEditingGainValue] = useState("");
    const editingGainInputRef = useRef<HTMLInputElement | null>(null);

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
    const [listScrollTop, setListScrollTop] = useState(0);
    const [listViewportHeight, setListViewportHeight] = useState(0);

    // 轨道右键菜单的快捷键提示（随用户在快捷键设置中的自定义绑定实时变化）。
    const trackAddShortcut = useAppSelector((s) =>
        formatKeybinding(selectKeybinding(s, "track.add"), ""),
    );
    const trackCloneShortcut = useAppSelector((s) =>
        formatKeybinding(selectKeybinding(s, "track.clone"), ""),
    );
    const trackDeleteShortcut = useAppSelector((s) =>
        formatKeybinding(selectKeybinding(s, "track.delete"), ""),
    );

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

    // 名称编辑中：点击输入框以外的任意位置都视为确认并退出编辑。
    // 时间轴画布等区域会在自身的 pointerdown 处理里 preventDefault，
    // 导致输入框收不到 blur 事件；因此这里用 window 捕获阶段的全局
    // 监听兜底，保证点击界面任何其他位置都能确认并退出编辑框。
    const commitTrackNameRef = useRef(commitTrackName);
    useEffect(() => {
        commitTrackNameRef.current = commitTrackName;
    });
    useEffect(() => {
        if (!editingTrackId) return;
        const handler = (e: PointerEvent) => {
            const target = e.target as Node | null;
            if (target && nameInputRef.current?.contains(target)) return;
            commitTrackNameRef.current();
        };
        window.addEventListener("pointerdown", handler, true);
        return () => window.removeEventListener("pointerdown", handler, true);
    }, [editingTrackId]);

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

    function beginTrackGainEdit(trackId: string, volume: number) {
        const db = clampGainDb(gainToDb(volume));
        setEditingGainTrackId(trackId);
        setEditingGainValue(db.toFixed(1));
    }

    function commitTrackGainEdit() {
        if (!editingGainTrackId) return;
        const trackId = editingGainTrackId;
        const parsed = parseFloat(editingGainValue);
        if (!isNaN(parsed)) {
            const nextDb = clampGainDb(parsed);
            const nextVolume = dbToGain(nextDb);
            clearPendingVolumeCommit(trackId);
            onVolumeUiChange(trackId, nextVolume);
            onVolumeCommit(trackId, nextVolume);
        }
        setEditingGainTrackId(null);
    }

    function cancelTrackGainEdit() {
        setEditingGainTrackId(null);
    }

    useEffect(() => {
        rowHeightRef.current = rowHeight;
    }, [rowHeight]);

    useEffect(() => {
        const el = listRef.current;
        if (!el) return;

        const updateViewportHeight = () => {
            setListViewportHeight(el.clientHeight || 0);
        };

        updateViewportHeight();

        if (typeof ResizeObserver !== "undefined") {
            const observer = new ResizeObserver(() => {
                updateViewportHeight();
            });
            observer.observe(el);
            return () => {
                observer.disconnect();
            };
        }

        window.addEventListener("resize", updateViewportHeight);
        return () => {
            window.removeEventListener("resize", updateViewportHeight);
        };
    }, []);

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
        } catch {
            // Pointer capture can fail in some WebView/edge cases; panning still works.
        }

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

        const knobEl = e.currentTarget;
        const startY = e.clientY;
        const startDb = clampGainDb(gainToDb(volume));
        setVolumeDrag({ trackId, baseDb: startDb });
        setVolumeHoveredTrackId(trackId);
        setVolumeTooltipPos({ x: e.clientX, y: e.clientY });
        let lastDb = startDb;
        const fineAxisState: FineAxisDragState = {
            raw: startY,
            adjusted: startY,
            fineActive: isModifierActive(paramFineAdjustKb, e.nativeEvent),
        };

        const onMove = (ev: PointerEvent) => {
            setVolumeTooltipPos({ x: ev.clientX, y: ev.clientY });
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

        const onEnd = (ev: PointerEvent) => {
            window.removeEventListener("pointermove", onMove);
            window.removeEventListener("pointerup", onEnd);
            window.removeEventListener("pointercancel", onEnd);
            setVolumeDrag(null);
            const knobRect = knobEl.getBoundingClientRect();
            const stillOverKnob =
                ev.clientX >= knobRect.left &&
                ev.clientX <= knobRect.right &&
                ev.clientY >= knobRect.top &&
                ev.clientY <= knobRect.bottom;
            setVolumeHoveredTrackId(stillOverKnob ? trackId : null);
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
        copyMode = false,
    ): {
        parentTrackId: string | null;
        targetIndex: number;
        mode: "reorder" | "nest";
    } {
        const el = listRef.current;
        const bounds = el?.getBoundingClientRect();

        // 复制模式下源轨道原地不动、仍在列表中，因此同级列表不做剔除，
        // 插入索引直接对应可见行缝；后端“先克隆（紧贴源）再移动到
        // target_index”的组合会把它映射到完全相同的位置。

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
            const roots = copyMode
                ? siblingsOf(null)
                : siblingsOf(null).filter((id) => id !== draggingTrackId);
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
            // 复制模式不存在自嵌套环（克隆尚未入树），无需环检测。
            if (!copyMode && wouldCreateCycle(draggingTrackId, parentTrackId)) {
                const roots = siblingsOf(null).filter((id) => id !== draggingTrackId);
                return {
                    parentTrackId: null,
                    targetIndex: roots.length,
                    mode: "reorder",
                };
            }
            const children = copyMode
                ? siblingsOf(parentTrackId)
                : siblingsOf(parentTrackId).filter((id) => id !== draggingTrackId);
            return {
                parentTrackId,
                targetIndex: children.length,
                mode: "nest",
            };
        }

        let parentTrackId = over.parentId ?? null;
        if (!copyMode && wouldCreateCycle(draggingTrackId, parentTrackId)) {
            parentTrackId = null;
        }

        if (over.id === draggingTrackId && !copyMode) {
            const siblingsIncl = siblingsOf(parentTrackId);
            const indexSelf = Math.max(0, siblingsIncl.indexOf(draggingTrackId));
            return { parentTrackId, targetIndex: indexSelf, mode: "reorder" };
        }

        const siblings = copyMode
            ? siblingsOf(parentTrackId)
            : siblingsOf(parentTrackId).filter((id) => id !== draggingTrackId);
        const baseIndex = Math.max(0, siblings.indexOf(over.id));
        // 使用 35% 边缘区域：上 35% 插入到上方，�?35% 插入到下方，中间 30% 保持不动
        const edgeZone = rowHeight * 0.35;
        const insertAfter = yInRow > rowHeight - edgeZone;
        const insertBefore = yInRow < edgeZone;

        if (copyMode) {
            // 复制模式：每次放置都产生一个克隆。落在目标行上缘 → 插到它前面；
            // 中间与下缘 → 插到它后面（含源轨道自身行，即“复制到源下方”）。
            return {
                parentTrackId,
                targetIndex: Math.min(siblings.length, baseIndex + (insertBefore ? 0 : 1)),
                mode: "reorder",
            };
        }

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

    const visibleTrackWindow = useMemo(
        () =>
            computeVisibleTrackWindow({
                totalTracks: tracks.length,
                rowHeight,
                scrollTopPx: listScrollTop,
                viewportHeightPx: listViewportHeight,
                overscanRows: 2,
            }),
        [listScrollTop, listViewportHeight, rowHeight, tracks.length],
    );
    const visibleTracks = useMemo(
        () => tracks.slice(visibleTrackWindow.startIndex, visibleTrackWindow.endIndex + 1),
        [tracks, visibleTrackWindow.endIndex, visibleTrackWindow.startIndex],
    );

    const buildVolumeTooltip = (trackId: string, volume: number): string => {
        const db = clampGainDb(gainToDb(volume));
        if (volumeDrag && volumeDrag.trackId === trackId) {
            return t("gain_value_tooltip_drag")
                .replace("{gain}", formatGainDbValue(db))
                .replace("{delta}", formatGainDbValue(db - volumeDrag.baseDb));
        }
        return t("gain_value_tooltip").replace("{gain}", formatGainDbValue(db));
    };

    const volumeTooltipTrackId = volumeDrag?.trackId ?? volumeHoveredTrackId;
    const volumeTooltipTrack = volumeTooltipTrackId
        ? tracks.find((tr) => tr.id === volumeTooltipTrackId)
        : null;
    const volumeTooltipText =
        volumeTooltipTrackId && volumeTooltipTrack
            ? buildVolumeTooltip(
                  volumeTooltipTrackId,
                  currentTrackVolumeById[volumeTooltipTrackId] ?? 1,
              )
            : "";
    const showVolumeTooltip = volumeTooltipTrackId != null && volumeTooltipPos != null;

    return (
        <Flex direction="column" className="w-64 border-r border-qt-border bg-qt-window shrink-0">
            <Box
                className="border-b border-qt-border px-2 flex items-center justify-between gap-2 bg-qt-window shadow-sm z-10 relative"
                style={{ height: headerHeight }}
            >
                <Text size="2" weight="bold" color="gray" className="shrink-0">
                    {t("tracks")}
                </Text>
                <TrackHeaderPlayheadTime />
                {/* 速度映射小按钮（右下角）：显示/创建 或 清空/隐藏。 */}
                <TempoMapCornerButton />
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
                    const nextScrollTop = (e.currentTarget as HTMLDivElement).scrollTop;
                    setListScrollTop(nextScrollTop);
                    onScrollTopChange?.(nextScrollTop);
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
                <div
                    style={{
                        position: "relative",
                        minHeight: tracks.length * rowHeight,
                    }}
                >
                    <div
                        style={{
                            transform: `translateY(${visibleTrackWindow.startIndex * rowHeight}px)`,
                        }}
                    >
                        {visibleTracks.map((track) => {
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
                            const volumeTooltip = buildVolumeTooltip(track.id, volume);

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

                                            // 复制拖动修饰键按住时：预览与放置都按
                                            // “源轨道不剔除”的复制索引计算。
                                            const copyMode = Boolean(
                                                copyDragKb &&
                                                isModifierActive(copyDragKb, ev) &&
                                                onDuplicateTrackTo,
                                            );
                                            const spec = computeDropSpec(
                                                drag.trackId,
                                                ev.clientX,
                                                ev.clientY,
                                                copyMode,
                                            );
                                            const overInfo = trackAtClientY(ev.clientY);
                                            const over = overInfo.track;

                                            let indicatorY: number | null = null;
                                            if (spec.mode === "reorder") {
                                                const listBounds =
                                                    listRef.current?.getBoundingClientRect();
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
                                                        const insertBefore =
                                                            overInfo.yInRow < edgeZone;
                                                        if (insertAfter) {
                                                            indicatorY =
                                                                idx * rowHeight + rowHeight;
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

                                            // 与预览一致：按复制/移动语义计算放置位置。
                                            const copyActive = Boolean(
                                                copyDragKb && isModifierActive(copyDragKb, ev),
                                            );
                                            const spec = computeDropSpec(
                                                drag.trackId,
                                                ev.clientX,
                                                ev.clientY,
                                                copyActive && onDuplicateTrackTo != null,
                                            );

                                            // “复制拖动”修饰键按住时：在放置位置克隆轨道
                                            // （克隆子树移动到拖放位置），源轨道保持原位。
                                            // 与移动不同，克隆到“与源相同的位置”同样有意义，
                                            // 因此跳过同位 no-op 判断。
                                            if (copyActive && onDuplicateTrackTo) {
                                                onDuplicateTrackTo({
                                                    trackId: drag.trackId,
                                                    targetIndex: spec.targetIndex,
                                                    parentTrackId: spec.parentTrackId,
                                                });
                                                return;
                                            }

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
                                    {/* Always-visible left accent bar (pinned to list edge).
                                        显示归一化轨道色：与时间线 Clip 色块同源，
                                        挑的颜色与实际出现的颜色严格一致。 */}
                                    <div
                                        className={`absolute left-0 top-0 bottom-0 w-1 transition-opacity ${selected ? "opacity-100" : "opacity-80 group-hover:opacity-90"}`}
                                        style={{
                                            backgroundColor:
                                                track.color != null
                                                    ? normalizedTrackColorCss(track.color, darkMode)
                                                    : "var(--qt-highlight)",
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
                                                backgroundColor:
                                                    track.color != null
                                                        ? normalizedTrackColorCss(track.color, darkMode)
                                                        : "var(--qt-highlight)",
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
                                                    <Flex
                                                        align="center"
                                                        gap="1"
                                                        className="min-w-0 flex-1"
                                                    >
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
                                                                data-tooltip={t(
                                                                    "track_change_color",
                                                                )}
                                                                onPointerDown={(e) =>
                                                                    e.stopPropagation()
                                                                }
                                                                onClick={(e) => {
                                                                    e.stopPropagation();
                                                                    setColorPickerTrackId(
                                                                        colorPickerTrackId ===
                                                                            track.id
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
                                                                    {TRACK_COLOR_PALETTE_KEYS.map(
                                                                        (opt) => (
                                                                            <button
                                                                                key={opt.value}
                                                                                data-tooltip={t(
                                                                                    opt.key,
                                                                                )}
                                                                                className={`w-4 h-4 rounded-full transition-transform hover:scale-125 ${
                                                                                    (track.color ||
                                                                                        "#4f8ef7") ===
                                                                                    opt.value
                                                                                        ? "ring-2 ring-white/80 scale-110"
                                                                                        : ""
                                                                                }`}
                                                                                style={{
                                                                                    backgroundColor:
                                                                                        opt.value,
                                                                                }}
                                                                                onPointerDown={(
                                                                                    e,
                                                                                ) =>
                                                                                    e.stopPropagation()
                                                                                }
                                                                                onClick={(e) => {
                                                                                    e.stopPropagation();
                                                                                    onTrackColorChange?.(
                                                                                        track.id,
                                                                                        opt.value,
                                                                                    );
                                                                                    setColorPickerTrackId(
                                                                                        null,
                                                                                    );
                                                                                }}
                                                                            />
                                                                        ),
                                                                    )}
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
                                                                onPointerDown={(e) =>
                                                                    e.stopPropagation()
                                                                }
                                                                onClick={(e) => e.stopPropagation()}
                                                                autoFocus
                                                            />
                                                        ) : (
                                                            <Text
                                                                size="2"
                                                                weight="medium"
                                                                className={`text-qt-text truncate pr-2 ${depth > 0 ? "opacity-90" : ""} cursor-text select-none`}
                                                                onPointerDown={(e) =>
                                                                    e.stopPropagation()
                                                                }
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
                                                            onPointerDown={(e) =>
                                                                e.stopPropagation()
                                                            }
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
                                                    onClick={(e) => e.stopPropagation()}
                                                    onDoubleClick={(e) => {
                                                        e.stopPropagation();
                                                        const target =
                                                            e.target instanceof HTMLElement
                                                                ? e.target
                                                                : null;
                                                        if (
                                                            target?.closest?.(
                                                                "[data-track-gain-value]",
                                                            )
                                                        ) {
                                                            beginTrackGainEdit(track.id, volume);
                                                            return;
                                                        }
                                                        clearPendingVolumeCommit(track.id);
                                                        onVolumeUiChange(track.id, 1);
                                                        onVolumeCommit(track.id, 1);
                                                    }}
                                                >
                                                    <Flex align="center" gap="2">
                                                        <button
                                                            type="button"
                                                            className="relative w-8 h-8 rounded-full border border-qt-border bg-qt-window hover:bg-qt-surface transition-colors shrink-0"
                                                            aria-label={volumeTooltip}
                                                            data-track-volume-knob
                                                            data-track-id={track.id}
                                                            onPointerEnter={(e) => {
                                                                setVolumeHoveredTrackId(track.id);
                                                                setVolumeTooltipPos({
                                                                    x: e.clientX,
                                                                    y: e.clientY,
                                                                });
                                                            }}
                                                            onPointerMove={(e) => {
                                                                setVolumeTooltipPos({
                                                                    x: e.clientX,
                                                                    y: e.clientY,
                                                                });
                                                            }}
                                                            onPointerLeave={() => {
                                                                setVolumeHoveredTrackId((prev) =>
                                                                    prev === track.id ? null : prev,
                                                                );
                                                            }}
                                                            onPointerDown={(e) =>
                                                                beginVolumeKnobDrag(
                                                                    e,
                                                                    track.id,
                                                                    volume,
                                                                )
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
                                                        {editingGainTrackId === track.id ? (
                                                            <input
                                                                ref={editingGainInputRef}
                                                                className="w-14 text-xs rounded px-1 outline-none text-left tabular-nums bg-qt-base text-qt-text border border-qt-border"
                                                                value={editingGainValue}
                                                                onChange={(e) =>
                                                                    setEditingGainValue(
                                                                        e.target.value,
                                                                    )
                                                                }
                                                                autoFocus
                                                                onFocus={(e) =>
                                                                    e.currentTarget.select()
                                                                }
                                                                onKeyDown={(e) => {
                                                                    e.stopPropagation();
                                                                    if (e.key === "Enter") {
                                                                        commitTrackGainEdit();
                                                                    }
                                                                    if (e.key === "Escape") {
                                                                        cancelTrackGainEdit();
                                                                    }
                                                                }}
                                                                onBlur={commitTrackGainEdit}
                                                                onPointerDown={(e) =>
                                                                    e.stopPropagation()
                                                                }
                                                                onDoubleClick={(e) =>
                                                                    e.stopPropagation()
                                                                }
                                                            />
                                                        ) : (
                                                            <Text
                                                                size="1"
                                                                color={
                                                                    Math.abs(gainToDb(volume)) <
                                                                    0.05
                                                                        ? "iris"
                                                                        : "gray"
                                                                }
                                                                /* 0.0 dB 用强调色标记"默认增益"，
                                                                   highContrast 保证小字在面板底色上可读 */
                                                                highContrast={
                                                                    Math.abs(gainToDb(volume)) <
                                                                    0.05
                                                                }
                                                                className="leading-none tabular-nums select-none"
                                                                data-track-gain-value
                                                                onPointerDown={(e) =>
                                                                    e.stopPropagation()
                                                                }
                                                            >
                                                                {formatGainLabel(volume)}
                                                            </Text>
                                                        )}
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
                                                        variant={composeEnabled ? "solid" : "surface"}
                                                        color={composeEnabled ? "iris" : "gray"}
                                                        data-tooltip={t("compose")}
                                                        onPointerDown={(e) => e.stopPropagation()}
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            onToggleCompose(
                                                                track.id,
                                                                !composeEnabled,
                                                            );
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
                                                    variant={muted ? "solid" : "surface"}
                                                    color={muted ? "red" : "gray"}
                                                    data-tooltip={
                                                        muted ? t("clip_unmute") : t("clip_mute")
                                                    }
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
                                                    variant={solo ? "solid" : "surface"}
                                                    color={solo ? "amber" : "gray"}
                                                    data-tooltip={t("solo")}
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
                    </div>
                </div>

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
                        className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors flex items-center justify-between gap-3"
                        onClick={() => {
                            onCreateTrackBelow?.(trackCtxMenu.trackId);
                            setTrackCtxMenu(null);
                        }}
                    >
                        <span>{t("track_add")}</span>
                        {trackAddShortcut && (
                            <span className="text-[10px] opacity-50 shrink-0">
                                {trackAddShortcut}
                            </span>
                        )}
                    </button>
                    <button
                        className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors flex items-center justify-between gap-3"
                        onClick={() => {
                            onDuplicateTrack?.(trackCtxMenu.trackId);
                            setTrackCtxMenu(null);
                        }}
                    >
                        <span>{t("track_clone")}</span>
                        {trackCloneShortcut && (
                            <span className="text-[10px] opacity-50 shrink-0">
                                {trackCloneShortcut}
                            </span>
                        )}
                    </button>
                    <button
                        className="w-full text-left px-3 py-1.5 text-sm hover:bg-qt-button-hover transition-colors text-red-400 hover:text-red-300 flex items-center justify-between gap-3"
                        disabled={isLastRootTrack(trackCtxMenu.trackId)}
                        onClick={() => {
                            onRemoveTrack(trackCtxMenu.trackId);
                            setTrackCtxMenu(null);
                        }}
                    >
                        <span>{t("ctx_delete")}</span>
                        {trackDeleteShortcut && (
                            <span className="text-[10px] opacity-50 shrink-0">
                                {trackDeleteShortcut}
                            </span>
                        )}
                    </button>
                </div>
            )}
            <AppTooltipBubble
                text={volumeTooltipText}
                position={showVolumeTooltip ? volumeTooltipPos : null}
            />
        </Flex>
    );
};

export const TrackList = React.memo(TrackListInner);
