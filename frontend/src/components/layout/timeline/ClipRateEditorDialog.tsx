/**
 * 播放速率高级编辑浮层（右键 Clip 右上角的播放速率角标触发）。
 *
 * 上下文菜单样式的固定定位面板（与 ClipContextMenu 同款视觉），而非大弹窗：
 * - 直接输入拉伸倍率（与角标展示值同口径：有效速率）；
 * - 原 BPM 预填 Clip 起始位置的 Tempo 区间 BPM（无 Tempo Map 时为工程 BPM），
 *   新 BPM 按 当前倍率 预填（原 BPM × 倍率）——倍率与原/新 BPM 三者自洽：
 *   打开时所见倍率与角标一致，改新 BPM 即“把这段当成新 BPM 来播”；
 * - “自动调整 Clip 长度”开关（默认开）：源窗口保持不变，时长按新旧速率
 *   反比缩放；
 * - 多选 Clip 时批量应用到所有可编辑目标。
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { useI18n } from "../../../i18n/I18nProvider";
import { useAppSelector } from "../../../app/hooks";
import { isModifierActive, selectKeybinding } from "../../../features/keybindings/keybindingsSlice";
import { tempoAtSec, clampBpm } from "../../../utils/tempoMap";
import type { TempoMap } from "../../../utils/tempoMap";
import { parsePlaybackRateInput } from "./runtime/timelineCanvasStyle";
import { formatEditNumber } from "./math";
import {
    formatDurationUnit,
    formatFadeLengthTooltip,
    parseDurationInput,
    type FadeLengthFormatContext,
} from "./timeFormat";
import type { ClipInfo } from "../../../features/session/sessionTypes";

const FALLBACK_BEATS_PER_BAR = 4;

// 滚轮步进：普通 / 精细（修饰键 = modifier.paramFineAdjust）。
const RATE_WHEEL_STEP = 0.1;
const RATE_FINE_STEP = 0.01;
const BPM_WHEEL_STEP = 1;
const BPM_FINE_STEP = 0.1;
// 有效速率上下限（与 set_clip_state reducer 的 clamp 同口径）。
const MIN_RATE = 0.1;
const MAX_RATE = 10;

export interface ClipRateEditorPosition {
    x: number;
    y: number;
}

export interface ClipRateEditorDialogProps {
    open: boolean;
    /** 目标 Clip（打开时由父级从会话状态解析；null 时不渲染）。 */
    clip: ClipInfo | null;
    /** 右键触发时的屏幕坐标（浮层锚点）。 */
    position: ClipRateEditorPosition | null;
    tempoMap: TempoMap | null;
    /** 工程 BPM（无 Tempo Map 区间时的回退值）。 */
    projectBpm: number;
    /** 批量应用提示：多选时可编辑目标数（含当前 Clip）。 */
    targetCount: number;
    /** 主/副时间单位的时长格式化上下文（与淡化 ToolTips 同源）。 */
    formatCtx: FadeLengthFormatContext;
    onApply: (rate: number, adjustLength: boolean, durationSec: number | null) => void;
    onOpenChange: (open: boolean) => void;
}

/** 数字输入失败解析时的占位语义：保留字符串、提交时校验。 */
function parseBpmText(raw: string): number | null {
    const value = Number(raw.trim());
    if (!Number.isFinite(value) || value <= 0) return null;
    return clampBpm(value);
}

/** 面板内容：仅在 open 时按目标 Clip 挂载（key = clip.id），预填一次。 */
function ClipRateEditorFields({
    clip,
    position,
    tempoMap,
    projectBpm,
    targetCount,
    formatCtx,
    onApply,
    onClose,
}: {
    clip: ClipInfo;
    position: ClipRateEditorPosition;
    tempoMap: TempoMap | null;
    projectBpm: number;
    targetCount: number;
    formatCtx: FadeLengthFormatContext;
    onApply: (rate: number, adjustLength: boolean, durationSec: number | null) => void;
    onClose: () => void;
}) {
    const { t } = useI18n();
    const tAny = t as (key: string) => string;
    const menuRef = useRef<HTMLDivElement | null>(null);
    // 精细调整修饰键（与 FadeContextMenu 的滑轮步进同一来源）。
    const fineAdjustKb = useAppSelector((state) =>
        selectKeybinding(state, "modifier.paramFineAdjust"),
    );

    const currentBpm = tempoAtSec(tempoMap, clip.startSec, {
        bpm: projectBpm,
        beatsPerBar: FALLBACK_BEATS_PER_BAR,
    }).bpm;
    const prefilledOldBpm = formatEditNumber(currentBpm);

    const [rateText, setRateText] = useState(() => formatEditNumber(clip.playbackRate));
    const [oldBpmText, setOldBpmText] = useState(prefilledOldBpm);
    // 自洽预填：新 BPM = 原 BPM × 当前倍率（与速率字段在打开时一致）。
    const [newBpmText, setNewBpmText] = useState(() =>
        formatEditNumber(currentBpm * clip.playbackRate),
    );
    const [autoLength, setAutoLength] = useState(true);
    // 时长：允许以主/副时间单位格式输入；解析经 parseDurationInput。
    const [durationText, setDurationText] = useState(() =>
        formatDurationUnit(formatCtx.primaryTimeUnit, Number(clip.lengthSec) || 0, formatCtx),
    );
    const [durationEdited, setDurationEdited] = useState(false);
    const [rateEdited, setRateEdited] = useState(false);

    const parsedDuration = useMemo(
        () =>
            parseDurationInput(
                durationText,
                formatCtx.primaryTimeUnit,
                formatCtx.secondaryTimeUnit,
                formatCtx,
            ),
        [durationText, formatCtx],
    );

    const parsed = useMemo(() => {
        const rate = parsePlaybackRateInput(rateText);
        const oldBpm = parseBpmText(oldBpmText);
        const newBpm = parseBpmText(newBpmText);
        return { rate, oldBpm, newBpm };
    }, [rateText, oldBpmText, newBpmText]);

    const rateChanged =
        rateEdited && parsed.rate != null
            ? Math.abs(parsed.rate - clip.playbackRate) > 1e-9
            : false;
    const durationChanged =
        durationEdited && parsedDuration != null
            ? Math.abs(parsedDuration - (Number(clip.lengthSec) || 0)) > 1e-9
            : false;

    const canApply =
        (!rateEdited || parsed.rate != null) &&
        (!durationEdited || parsedDuration != null) &&
        (rateChanged || durationChanged);

    const previewSec = useMemo(() => {
        if (durationChanged && parsedDuration != null) return parsedDuration;
        if (rateChanged && parsed.rate != null && autoLength) {
            const oldRate = Number(clip.playbackRate) || 1;
            return ((Number(clip.lengthSec) || 0) * oldRate) / parsed.rate;
        }
        return Number(clip.lengthSec) || 0;
    }, [clip, parsed.rate, autoLength, rateChanged, durationChanged, parsedDuration]);

    function applyRate(nextRate: number) {
        setRateEdited(true);
        setRateText(formatEditNumber(nextRate));
    }

    // ── 滚轮步进（普通 / 精细修饰键）────────────────────────────────────
    // 与 FadeContextMenu 的曲率滑块同一模式：deltaY 向上 = 增大。
    // 倍率：0.1 / 精细 0.01；BPM：1 / 精细 0.1。写入走格式化值（滚轮不是
    // 打字，可以直接吸附到步进精度），并保持三字段联动。

    function updateRateValue(next: number) {
        const clamped = Math.min(MAX_RATE, Math.max(MIN_RATE, next));
        setRateText(formatEditNumber(clamped));
        const oldBpm = parseBpmText(oldBpmText);
        if (oldBpm != null) {
            // 与手动改倍率的联动路径同口径：formatEditNumber 保留 6 位精度。
            // 展示级取整（如 toFixed(2)）会让新 BPM 与倍率/时长不再精确对应，
            // 且连续滚轮步进时每次都重新舍入，漂移会累积。
            setNewBpmText(formatEditNumber(oldBpm * clamped));
        }
    }

    function updateOldBpmValue(next: number) {
        const clamped = clampBpm(next);
        setOldBpmText(formatEditNumber(clamped));
        const newBpm = parseBpmText(newBpmText);
        if (newBpm != null && clamped > 1e-6) {
            applyRate(newBpm / clamped);
        }
    }

    function updateNewBpmValue(next: number) {
        const clamped = clampBpm(next);
        setNewBpmText(formatEditNumber(clamped));
        const oldBpm = parseBpmText(oldBpmText);
        if (oldBpm != null && oldBpm > 1e-6) {
            applyRate(clamped / oldBpm);
        }
    }

    // 视口边缘夹紧（与 ClipContextMenu 同款）。
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        if (rect.right > window.innerWidth) {
            el.style.left = `${Math.max(0, window.innerWidth - rect.width)}px`;
        }
        if (rect.bottom > window.innerHeight) {
            el.style.top = `${Math.max(0, window.innerHeight - rect.height)}px`;
        }
    }, [position]);

    // 点击面板以外（捕获阶段）→ 关闭；Escape → 关闭。
    useEffect(() => {
        const onDocPointerDown = (event: PointerEvent) => {
            const target = event.target instanceof Element ? event.target : null;
            if (menuRef.current && target && menuRef.current.contains(target)) return;
            onClose();
        };
        const onKey = (event: KeyboardEvent) => {
            if (event.key === "Escape") onClose();
        };
        document.addEventListener("pointerdown", onDocPointerDown, true);
        window.addEventListener("keydown", onKey);
        return () => {
            document.removeEventListener("pointerdown", onDocPointerDown, true);
            window.removeEventListener("keydown", onKey);
        };
    }, [onClose]);

    return (
        <div
            ref={menuRef}
            role="menu"
            data-hs-floating-menu="1"
            data-hs-context-menu="1"
            className="fixed z-[999] w-[248px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-2 px-3 flex flex-col gap-2"
            style={{ left: position.x, top: position.y }}
            onPointerDown={(e) => e.stopPropagation()}
            onContextMenu={(e) => e.preventDefault()}
        >
            <div className="text-[12px] font-medium">{tAny("clip_rate_editor_title")}</div>
            <div className="text-[10px] text-qt-text/60 leading-snug">
                {tAny("clip_rate_editor_bpm_hint")}
            </div>

            <label className="flex flex-col gap-1">
                <span className="text-[10px] text-qt-text/60">{tAny("clip_rate_editor_rate")}</span>
                <input
                    className="w-full text-xs rounded px-2 py-1 outline-none bg-black/20 border border-qt-border"
                    value={rateText}
                    onChange={(e) => {
                        setRateEdited(true);
                        setRateText(e.target.value);
                        // 手动改倍率 → 新 BPM 跟随（保持 原 BPM 不变）。
                        const rate = parsePlaybackRateInput(e.target.value);
                        const oldBpm = parseBpmText(oldBpmText);
                        if (rate != null && oldBpm != null) {
                            setNewBpmText(formatEditNumber(oldBpm * rate));
                        }
                    }}
                    onKeyDown={(e) => {
                        e.stopPropagation();
                        if (e.key === "Enter" && canApply) {
                            onApply(
                                parsed.rate ?? clip.playbackRate,
                                autoLength && !durationChanged,
                                durationChanged ? parsedDuration : null,
                            );
                            onClose();
                        }
                    }}
                    onWheel={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        setRateEdited(true);
                        const fine = isModifierActive(fineAdjustKb, e.nativeEvent);
                        const step = fine ? RATE_FINE_STEP : RATE_WHEEL_STEP;
                        const current = parsePlaybackRateInput(rateText) ?? clip.playbackRate;
                        updateRateValue(current + step * (e.deltaY < 0 ? 1 : -1));
                    }}
                />
            </label>

            <div className="flex gap-2">
                <label className="flex-1 flex flex-col gap-1">
                    <span className="text-[10px] text-qt-text/60">
                        {tAny("clip_rate_editor_old_bpm")}
                    </span>
                    <input
                        className="w-full text-xs rounded px-2 py-1 outline-none bg-black/20 border border-qt-border"
                        value={oldBpmText}
                        onChange={(e) => {
                            setOldBpmText(e.target.value);
                            // 修改原 BPM → 倍率 = 新 / 原。
                            const oldBpm = parseBpmText(e.target.value);
                            const newBpm = parseBpmText(newBpmText);
                            if (oldBpm != null && newBpm != null && oldBpm > 1e-6) {
                                applyRate(newBpm / oldBpm);
                            }
                        }}
                        onWheel={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            const fine = isModifierActive(fineAdjustKb, e.nativeEvent);
                            const step = fine ? BPM_FINE_STEP : BPM_WHEEL_STEP;
                            const current = parseBpmText(oldBpmText) ?? currentBpm;
                            updateOldBpmValue(current + step * (e.deltaY < 0 ? 1 : -1));
                        }}
                    />
                </label>
                <label className="flex-1 flex flex-col gap-1">
                    <span className="text-[10px] text-qt-text/60">
                        {tAny("clip_rate_editor_new_bpm")}
                    </span>
                    <input
                        className="w-full text-xs rounded px-2 py-1 outline-none bg-black/20 border border-qt-border"
                        value={newBpmText}
                        onChange={(e) => {
                            setNewBpmText(e.target.value);
                            // 修改新 BPM → 倍率 = 新 / 原。
                            const oldBpm = parseBpmText(oldBpmText);
                            const newBpm = parseBpmText(e.target.value);
                            if (oldBpm != null && newBpm != null && oldBpm > 1e-6) {
                                applyRate(newBpm / oldBpm);
                            }
                        }}
                        onWheel={(e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            const fine = isModifierActive(fineAdjustKb, e.nativeEvent);
                            const step = fine ? BPM_FINE_STEP : BPM_WHEEL_STEP;
                            const current = parseBpmText(newBpmText) ?? currentBpm;
                            updateNewBpmValue(current + step * (e.deltaY < 0 ? 1 : -1));
                        }}
                    />
                </label>
            </div>

            <label className="flex flex-col gap-1">
                <span className="text-[10px] text-qt-text/60">
                    {tAny("clip_rate_editor_duration")}
                    {": "}
                    {formatFadeLengthTooltip(Number(clip.lengthSec) || 0, formatCtx)}
                </span>
                <input
                    className={`w-full text-xs rounded px-2 py-1 outline-none bg-black/20 border ${
                        durationEdited && parsedDuration == null
                            ? "border-red-400/80"
                            : "border-qt-border"
                    }`}
                    value={durationText}
                    onChange={(e) => {
                        setDurationEdited(true);
                        setDurationText(e.target.value);
                        // 时长即拉伸：源窗口保持不变，由时长反推有效速率，
                        // 并联动 倍率 / 新 BPM 字段（三者始终自洽）。
                        const dur = parseDurationInput(
                            e.target.value,
                            formatCtx.primaryTimeUnit,
                            formatCtx.secondaryTimeUnit,
                            formatCtx,
                        );
                        const oldLengthSec = Number(clip.lengthSec) || 0;
                        const oldRate = Number(clip.playbackRate) || 1;
                        if (dur != null && dur > 1e-6 && oldRate > 1e-6) {
                            const implied = Math.min(
                                10,
                                Math.max(0.1, (oldLengthSec * oldRate) / dur),
                            );
                            setRateEdited(true);
                            updateRateValue(implied);
                        }
                    }}
                    onKeyDown={(e) => {
                        e.stopPropagation();
                        if (e.key === "Enter" && canApply) {
                            onApply(
                                parsed.rate ?? clip.playbackRate,
                                autoLength,
                                durationChanged && parsedDuration != null ? parsedDuration : null,
                            );
                            onClose();
                        }
                    }}
                />
                <span className="text-[10px] text-qt-text/60 tabular-nums">
                    {formatFadeLengthTooltip(previewSec, formatCtx)}
                </span>
            </label>

            <label className="flex items-center gap-2 select-none">
                <input
                    type="checkbox"
                    className="accent-current"
                    checked={autoLength}
                    onChange={(e) => setAutoLength(e.target.checked)}
                />
                <span className="text-[11px]">{tAny("clip_rate_editor_auto_length")}</span>
            </label>

            <div className="text-[10px] text-qt-text/60">
                {tAny("clip_rate_editor_result")}
                {": "}
                {formatFadeLengthTooltip(previewSec, formatCtx)}
                {!autoLength && !durationChanged
                    ? ` (${tAny("clip_rate_editor_keep_length")})`
                    : ""}
            </div>

            {targetCount > 1 ? (
                <div className="text-[10px] text-qt-text/60">
                    {tAny("clip_rate_editor_multi").replace("{count}", String(targetCount))}
                </div>
            ) : null}

            <div className="flex justify-end pt-1">
                <button
                    role="menuitem"
                    className="px-2 py-1 text-[11px] rounded bg-qt-button-hover/60 hover:bg-qt-button-hover disabled:opacity-40"
                    disabled={!canApply}
                    onClick={() => {
                        onApply(
                            parsed.rate ?? clip.playbackRate,
                            autoLength && !durationChanged,
                            durationChanged && parsedDuration != null ? parsedDuration : null,
                        );
                        onClose();
                    }}
                >
                    {tAny("clip_rate_editor_apply")}
                </button>
            </div>
        </div>
    );
}

export function ClipRateEditorDialog({
    open,
    clip,
    position,
    tempoMap,
    projectBpm,
    targetCount,
    formatCtx,
    onApply,
    onOpenChange,
}: ClipRateEditorDialogProps) {
    if (!open || !clip || !position) return null;
    return (
        <ClipRateEditorFields
            key={clip.id}
            clip={clip}
            position={position}
            tempoMap={tempoMap}
            projectBpm={projectBpm}
            targetCount={targetCount}
            formatCtx={formatCtx}
            onApply={onApply}
            onClose={() => onOpenChange(false)}
        />
    );
}
