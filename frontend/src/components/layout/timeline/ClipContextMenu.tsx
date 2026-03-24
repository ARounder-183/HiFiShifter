import React, { useLayoutEffect, useRef } from "react";
import type {
    ClipInfo,
    FadeCurveType,
} from "../../../features/session/sessionTypes";
import { useI18n } from "../../../i18n/I18nProvider";
import type { MessageKey } from "../../../i18n/messages";
import { useAppSelector } from "../../../app/hooks";
import {
    selectKeybinding,
    formatKeybinding,
} from "../../../features/keybindings/keybindingsSlice";
import { sortAndFilterFadedClips } from "./clipFadeContext";

// ── 单条菜单项 ──────────────────────────────────────────────────────────────
const MenuItem: React.FC<{
    label: string;
    shortcut?: string;
    disabled?: boolean;
    danger?: boolean;
    onClick: () => void;
}> = ({ label, shortcut, disabled, danger, onClick }) => (
    <button
        className={`px-3 py-1.5 text-left w-full text-[12px] transition-colors flex items-center justify-between gap-3
            ${
                disabled
                    ? "opacity-40 cursor-default"
                    : danger
                      ? "hover:bg-red-500/20 text-red-400"
                      : "hover:bg-qt-button-hover"
            }`}
        disabled={disabled}
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
            e.stopPropagation();
            onClick();
        }}
    >
        <span>{label}</span>
        {shortcut && (
            <span className="text-[10px] opacity-50 shrink-0">{shortcut}</span>
        )}
    </button>
);

const Divider: React.FC = () => (
    <div className="my-1 border-t border-qt-border" />
);

// ── 渐变曲线选项 ────────────────────────────────────────────────────────────
const CURVE_OPTION_KEYS: { value: FadeCurveType; key: MessageKey }[] = [
    { value: "linear", key: "fade_curve_linear" },
    { value: "sine", key: "fade_curve_sine" },
    { value: "exponential", key: "fade_curve_exponential" },
    { value: "logarithmic", key: "fade_curve_logarithmic" },
    { value: "scurve", key: "fade_curve_scurve" },
];

const FadeCurveRow: React.FC<{
    label: string;
    current: FadeCurveType;
    onSelect: (c: FadeCurveType) => void;
    t: (key: MessageKey) => string;
}> = ({ label, current, onSelect, t }) => (
    <div className="px-3 py-1.5 flex items-center gap-1.5 flex-wrap">
        <span className="text-[11px] text-qt-text/60 mr-1 shrink-0">
            {label}
        </span>
        {CURVE_OPTION_KEYS.map((opt) => (
            <button
                key={opt.value}
                title={t(opt.key)}
                className={`px-1.5 py-0.5 rounded text-[10px] transition-colors
                    ${
                        current === opt.value
                            ? "bg-qt-highlight text-white"
                            : "bg-qt-button hover:bg-qt-button-hover text-qt-text/80"
                    }`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                    e.stopPropagation();
                    onSelect(opt.value);
                }}
            >
                {t(opt.key)}
            </button>
        ))}
    </div>
);

// ── 主组件 ──────────────────────────────────────────────────────────────────
export const ClipContextMenu: React.FC<{
    x: number;
    y: number;
    /** 右键点击的 clip */
    clip: ClipInfo;
    /** 多个 clip 列表（含 clip 本身），长度 >= 2 时进入多选模式 */
    selectedClips: ClipInfo[];
    /** 与当前 clip 在同轨道上重叠的其他 clip */
    overlappingClips?: ClipInfo[];
    /** 播放头是否在 clip 范围内（用于分割按钮启用判断）*/
    playheadInClip: boolean;
    canSplitSelected: boolean;
    onClose: () => void;
    onDelete: (ids: string[]) => void;
    onMute: (ids: string[], muted: boolean) => void;
    onRename: (clipId: string) => void;
    onCopy: (ids: string[]) => void;
    onCut: (ids: string[]) => void;
    onReplace: (ids: string[]) => void;
    onSplit: (clipIds: string[]) => void;
    onGlue: (ids: string[]) => void;
    onNormalize: (ids: string[]) => void;
    onFadeCurveChange?: (
        clipId: string,
        target: "in" | "out",
        curve: FadeCurveType,
    ) => void;
}> = ({
    x,
    y,
    clip,
    selectedClips,
    overlappingClips = [],
    playheadInClip,
    canSplitSelected,
    onClose,
    onDelete,
    onMute,
    onRename,
    onCopy,
    onCut,
    onReplace,
    onSplit,
    onGlue,
    onNormalize,
    onFadeCurveChange,
}) => {
    const { t } = useI18n();
    const menuRef = useRef<HTMLDivElement>(null);
    const isMulti = selectedClips.length >= 2;
    const ids = isMulti ? selectedClips.map((c) => c.id) : [clip.id];

    const normalizeKb = useAppSelector((state) =>
        selectKeybinding(state, "clip.normalize"),
    );
    const normalizeShortcut = normalizeKb
        ? formatKeybinding(normalizeKb, "")
        : undefined;

    // 胶合：仅同轨且多选时可用
    const glueDisabled =
        !isMulti ||
        (() => {
            const trackId = selectedClips[0]?.trackId;
            return !trackId || selectedClips.some((c) => c.trackId !== trackId);
        })();

    // 多选中是否全部静音
    const allMuted = isMulti ? selectedClips.every((c) => c.muted) : clip.muted;

    function close() {
        onClose();
    }

    // Clamp menu position to viewport edges
    useLayoutEffect(() => {
        const el = menuRef.current;
        if (!el) return;
        const rect = el.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw)
            el.style.left = `${Math.max(0, vw - rect.width)}px`;
        if (rect.bottom > vh)
            el.style.top = `${Math.max(0, vh - rect.height)}px`;
    }, [x, y]);

    return (
        <div
            ref={menuRef}
            data-hs-context-menu="1"
            className="fixed z-50 min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {isMulti ? (
                // ── 多选菜单 ──
                <>
                    <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                        {t("ctx_selected_n").replace(
                            "{n}",
                            String(selectedClips.length),
                        )}
                    </div>
                    <Divider />
                    <MenuItem
                        label={t("ctx_delete_all")}
                        danger
                        onClick={() => {
                            onDelete(ids);
                            close();
                        }}
                    />
                    <MenuItem
                        label={
                            allMuted ? t("ctx_unmute_all") : t("ctx_mute_all")
                        }
                        onClick={() => {
                            onMute(ids, !allMuted);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_copy_all")}
                        onClick={() => {
                            onCopy(ids);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_cut_all")}
                        onClick={() => {
                            onCut(ids);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_replace_all")}
                        onClick={() => {
                            onReplace(ids);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_split_at_playhead")}
                        disabled={!canSplitSelected}
                        onClick={() => {
                            onSplit(ids);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_normalize_all")}
                        shortcut={normalizeShortcut}
                        onClick={() => {
                            onNormalize(ids);
                            close();
                        }}
                    />
                    <Divider />
                    <MenuItem
                        label={t("glue")}
                        disabled={glueDisabled}
                        onClick={() => {
                            onGlue(ids);
                            close();
                        }}
                    />
                </>
            ) : (
                // ── 单选菜单 ──
                <>
                    <MenuItem
                        label={t("ctx_delete")}
                        danger
                        onClick={() => {
                            onDelete([clip.id]);
                            close();
                        }}
                    />
                    <MenuItem
                        label={clip.muted ? t("clip_unmute") : t("clip_mute")}
                        onClick={() => {
                            onMute([clip.id], !clip.muted);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_rename")}
                        onClick={() => {
                            onRename(clip.id);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_copy")}
                        onClick={() => {
                            onCopy([clip.id]);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_cut")}
                        onClick={() => {
                            onCut([clip.id]);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_replace")}
                        onClick={() => {
                            onReplace([clip.id]);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_split_at_playhead")}
                        disabled={!playheadInClip}
                        onClick={() => {
                            onSplit([clip.id]);
                            close();
                        }}
                    />
                    <MenuItem
                        label={t("ctx_normalize")}
                        shortcut={normalizeShortcut}
                        onClick={() => {
                            onNormalize([clip.id]);
                            close();
                        }}
                    />
                    {onFadeCurveChange &&
                        (() => {
                            const fadedClips = sortAndFilterFadedClips({
                                clip,
                                overlappingClips,
                            });
                            if (fadedClips.length === 0) return null;
                            const showHeader = fadedClips.length > 1;
                            return (
                                <>
                                    <Divider />
                                    {showHeader && (
                                        <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                                            {t("overlapping_clips_header").replace(
                                                "{n}",
                                                String(fadedClips.length),
                                            )}
                                        </div>
                                    )}
                                    {fadedClips.map((fc) => (
                                        <React.Fragment key={fc.id}>
                                            {showHeader && (
                                                <div className="px-3 pt-1 text-[10px] text-qt-text/40 truncate">
                                                    {fc.name || fc.id}
                                                </div>
                                            )}
                                            {fc.fadeInSec > 0 && (
                                                <FadeCurveRow
                                                    label={t("fade_in")}
                                                    current={
                                                        (fc.fadeInCurve as FadeCurveType) ??
                                                        "sine"
                                                    }
                                                    onSelect={(c) => {
                                                        onFadeCurveChange(fc.id, "in", c);
                                                    }}
                                                    t={t}
                                                />
                                            )}
                                            {fc.fadeOutSec > 0 && (
                                                <FadeCurveRow
                                                    label={t("fade_out")}
                                                    current={
                                                        (fc.fadeOutCurve as FadeCurveType) ??
                                                        "sine"
                                                    }
                                                    onSelect={(c) => {
                                                        onFadeCurveChange(fc.id, "out", c);
                                                    }}
                                                    t={t}
                                                />
                                            )}
                                        </React.Fragment>
                                    ))}
                                </>
                            );
                        })()}
                </>
            )}
        </div>
    );
};
