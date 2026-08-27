import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { FadeShapeIcon } from "./FadeShapeIcon";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { useI18n } from "../../../i18n/I18nProvider";
import type { MessageKey } from "../../../i18n/messages";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { selectKeybinding, formatKeybinding } from "../../../features/keybindings/keybindingsSlice";
import {
    addClipTakeFromMediaRemote,
    cycleClipTakesRemote,
    duplicateClipTakeRemote,
    explodeClipTakesRemote,
    packClipsIntoTakesRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipActiveTakeRemote,
} from "../../../features/session/sessionSlice";
import { webApi } from "../../../services/webviewApi";
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
        role="menuitem"
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
        {shortcut && <span className="text-[10px] opacity-50 shrink-0">{shortcut}</span>}
    </button>
);

const Divider: React.FC = () => <div className="my-1 border-t border-qt-border" />;

/** 一级菜单中的二级子菜单；悬停或点击均可展开。 */
const SubMenu: React.FC<{
    label: string;
    disabled?: boolean;
    badge?: string;
    children: React.ReactNode;
}> = ({ label, disabled = false, badge, children }) => {
    const [open, setOpen] = useState(false);
    const wrapperRef = useRef<HTMLDivElement>(null);
    const panelRef = useRef<HTMLDivElement>(null);

    useLayoutEffect(() => {
        if (!open) return;
        const panel = panelRef.current;
        if (!panel) return;
        panel.style.left = "calc(100% - 4px)";
        panel.style.right = "auto";
        panel.style.top = "-5px";
        panel.style.bottom = "auto";

        const rect = panel.getBoundingClientRect();
        const vw = window.innerWidth;
        const vh = window.innerHeight;
        if (rect.right > vw - 4) {
            panel.style.left = "auto";
            panel.style.right = "calc(100% - 4px)";
        }
        if (rect.bottom > vh - 4) {
            panel.style.top = "auto";
            panel.style.bottom = "-5px";
        }
    }, [open]);

    return (
        <div
            ref={wrapperRef}
            className="relative"
            onMouseEnter={() => {
                if (!disabled) setOpen(true);
            }}
            onMouseLeave={() => setOpen(false)}
        >
            <button
                className={`px-3 py-1.5 text-left w-full text-[12px] transition-colors flex items-center justify-between gap-3
                    ${disabled ? "opacity-40 cursor-default" : "hover:bg-qt-button-hover"}`}
                disabled={disabled}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                    e.stopPropagation();
                    if (!disabled) setOpen((value) => !value);
                }}
                aria-haspopup="menu"
                aria-expanded={open}
            >
                <span className="flex items-center gap-2 min-w-0">
                    <span className="truncate">{label}</span>
                    {badge && (
                        <span className="text-[10px] leading-none rounded bg-black/20 px-1 py-0.5 opacity-70">
                            {badge}
                        </span>
                    )}
                </span>
                <svg
                    width="12"
                    height="12"
                    viewBox="0 0 15 15"
                    fill="none"
                    aria-hidden="true"
                    className="opacity-50 shrink-0"
                >
                    <path
                        d="M6 3.5L10 7.5L6 11.5"
                        stroke="currentColor"
                        strokeWidth="1.2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                    />
                </svg>
            </button>
            {open && !disabled && (
                <div
                    ref={panelRef}
                    role="menu"
                    data-hs-context-menu="1"
                    className="absolute z-[60] min-w-[190px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
                    onPointerDown={(e) => e.stopPropagation()}
                    onClick={(e) => e.stopPropagation()}
                >
                    {children}
                </div>
            )}
        </div>
    );
};

function effectiveFadeSecondsOf(clip: ClipInfo): { in: number; out: number } {
    return {
        in: (clip.autoFadeInSec ?? 0) > 0 ? (clip.autoFadeInSec ?? 0) : clip.fadeInSec,
        out: (clip.autoFadeOutSec ?? 0) > 0 ? (clip.autoFadeOutSec ?? 0) : clip.fadeOutSec,
    };
}

// ── REAPER 七预设淡变形状 ────────────────────────────────────────────────
// 菜单顺序与 REAPER 7.x 淡变右键菜单一致（Linear / Fast Start / Fast End /
// Fast Start Steep / Fast End Steep / Slow Start/End (Steep)），形状 id 与
// timeline/reaperFade.ts FADE_PRESETS 对应。
const FADE_SHAPE_OPTIONS: { shape: number; key: MessageKey }[] = [
    { shape: 0, key: "fade_shape_linear" },
    { shape: 1, key: "fade_shape_fast_start" },
    { shape: 2, key: "fade_shape_fast_end" },
    { shape: 3, key: "fade_shape_fast_start_steep" },
    { shape: 4, key: "fade_shape_fast_end_steep" },
    { shape: 5, key: "fade_shape_slow_start_end" },
    { shape: 6, key: "fade_shape_slow_start_end_steep" },
];

const FadeShapeRow: React.FC<{
    label: string;
    current: number;
    /** 本行是淡出（图标水平镜像，曲线方向与画布一致）。 */
    isOut?: boolean;
    onSelect: (shape: number) => void;
    t: (key: MessageKey) => string;
}> = ({ label, current, isOut = false, onSelect, t }) => (
    <div className="px-3 py-1.5 flex items-center gap-1 flex-wrap">
        <span className="text-[11px] text-qt-text/60 mr-1 shrink-0">{label}</span>
        {FADE_SHAPE_OPTIONS.map((opt) => (
            <button
                key={opt.key}
                data-tooltip={t(opt.key)}
                className={`p-0.5 rounded transition-colors leading-none
                    ${
                        // 小数变体（如 1.1）按基础族高亮（REAPER 同语义）。
                        Math.trunc(current) === opt.shape
                            ? "bg-qt-highlight text-white"
                            : "bg-qt-button hover:bg-qt-button-hover text-qt-text/80"
                    }`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                    e.stopPropagation();
                    onSelect(opt.shape);
                }}
            >
                <FadeShapeIcon shape={opt.shape} mirrored={isOut} />
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
    onReplaceMidi?: (ids: string[]) => void;
    onQuickExport: (ids: string[]) => void;
    onSplit: (clipIds: string[]) => void;
    onGlue: (ids: string[]) => void;
    onGroup?: (ids: string[]) => void;
    onUngroup?: (ids: string[]) => void;
    onConvertToPitchRef?: (ids: string[]) => void;
    onUpdatePitchRef?: (ids: string[]) => void;
    onExportMidi?: (ids: string[]) => void;
    onNormalize: (ids: string[]) => void;
    onToggleReverse: (ids: string[], reversed: boolean) => void;
    onToggleLoop?: (ids: string[], loopEnabled: boolean) => void;
    /** 切换淡入/淡出的 REAPER 形状预设（保留曲率 dir 不变）。 */
    onFadeShapeChange?: (clipId: string, target: "in" | "out", shape: number) => void;
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
    onReplaceMidi,
    onQuickExport,
    onSplit,
    onGlue,
    onGroup,
    onUngroup,
    onConvertToPitchRef,
    onUpdatePitchRef,
    onExportMidi,
    onNormalize,
    onToggleReverse,
    onToggleLoop,
    onFadeShapeChange,
}) => {
    const { t } = useI18n();
    const dispatch = useAppDispatch();
    const menuRef = useRef<HTMLDivElement>(null);
    /** Take 重命名的内联输入草稿（替代 window.prompt）。 */
    const [takeRenameDraft, setTakeRenameDraft] = useState<{
        takeId: string;
        value: string;
    } | null>(null);
    const ids = selectedClips.length >= 2 ? selectedClips.map((c) => c.id) : [clip.id];
    const isMulti = ids.length >= 2;
    const isSingle = !isMulti;

    // 音高参考块判断
    const isPitch = (c: ClipInfo) => c.midiNoteCount != null;
    const takes = Array.isArray(clip.takes) ? clip.takes : [];
    const activeTake = takes.find((take) => take.id === clip.activeTakeId) ?? takes[0];
    const allPitchAdjustment = selectedClips.length > 0 && selectedClips.every(isPitch);
    const hasPitchAdjustment = selectedClips.some(isPitch);
    const audioOnlyIds = selectedClips.filter((c) => !isPitch(c)).map((c) => c.id);
    const pitchOnlyIds = selectedClips.filter(isPitch).map((c) => c.id);

    const normalizeKb = useAppSelector((state) => selectKeybinding(state, "clip.normalize"));
    const normalizeShortcut = normalizeKb ? formatKeybinding(normalizeKb, "") : undefined;

    // 胶合：仅同轨且多选时可用，且不能混合音高参考块和常规音频块
    const hasMixedTypes = hasPitchAdjustment && !allPitchAdjustment;
    const glueDisabled =
        !isMulti ||
        hasMixedTypes ||
        (() => {
            const trackId = selectedClips[0]?.trackId;
            return !trackId || selectedClips.some((c) => c.trackId !== trackId);
        })();

    // 多选中是否全部静音
    const allMuted = isMulti ? selectedClips.every((c) => c.muted) : clip.muted;
    const allReversed = isMulti ? selectedClips.every((c) => c.reversed) : clip.reversed;
    // 多选中是否已全部启用 Loop（循环源）
    const allLooped = isMulti ? selectedClips.every((c) => c.loopEnabled) : clip.loopEnabled;

    // 编组 / 解组
    const hasGroup = selectedClips.some((c) => c.groupId != null);

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
        if (rect.right > vw) el.style.left = `${Math.max(0, vw - rect.width)}px`;
        if (rect.bottom > vh) el.style.top = `${Math.max(0, vh - rect.height)}px`;
    }, [x, y]);

    // Escape 关闭菜单（键盘可达性）；输入框内的 Escape 由其自身的
    // onKeyDown stopPropagation 拦截，不会触发这里。
    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                onClose();
            }
        };
        window.addEventListener("keydown", onKey);
        return () => window.removeEventListener("keydown", onKey);
    }, [onClose]);

    return (
        <div
            ref={menuRef}
            role="menu"
            data-hs-context-menu="1"
            data-hs-floating-menu="1"
            className="fixed z-50 min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {isMulti && (
                <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                    {t("ctx_selected_n").replace("{n}", String(selectedClips.length))}
                </div>
            )}

            <MenuItem
                label={isMulti ? t("ctx_delete_all") : t("ctx_delete")}
                danger
                onClick={() => {
                    onDelete(ids);
                    close();
                }}
            />
            <MenuItem
                label={
                    allMuted
                        ? isMulti
                            ? t("ctx_unmute_all")
                            : t("clip_unmute")
                        : isMulti
                          ? t("ctx_mute_all")
                          : t("clip_mute")
                }
                onClick={() => {
                    onMute(ids, !allMuted);
                    close();
                }}
            />
            <Divider />
            <SubMenu
                label={t("clip_takes")}
                badge={takes.length > 1 ? String(takes.length) : undefined}
            >
                {isMulti && (
                    <MenuItem
                        label={t("clip_pack_into_takes")}
                        onClick={() => {
                            void dispatch(packClipsIntoTakesRemote({ clipIds: ids }));
                            close();
                        }}
                    />
                )}
                {isSingle && (
                    <>
                        {takes.map((take) => (
                            <MenuItem
                                key={take.id}
                                // 活跃 take 用 ● 标记；非活跃用 em-space（U+2003，
                                // 不会被 HTML 空白折叠）保持对齐。
                                label={`${
                                    take.id === clip.activeTakeId
                                        ? t("clip_take_active_mark")
                                        : "\u2003"
                                } ${take.name || take.id}`}
                                disabled={takes.length <= 1}
                                onClick={() => {
                                    // 点击已激活的 take 是 no-op：跳过 dispatch，
                                    // 避免无谓的乐观切换+回滚快照+全量快照刷新。
                                    if (take.id === clip.activeTakeId) {
                                        close();
                                        return;
                                    }
                                    void dispatch(
                                        setClipActiveTakeRemote({
                                            clipId: clip.id,
                                            takeId: take.id,
                                        }),
                                    );
                                    close();
                                }}
                            />
                        ))}
                        {takes.length > 1 && (
                            <>
                                <Divider />
                                <MenuItem
                                    label={t("clip_take_cycle_prev")}
                                    onClick={() => {
                                        void dispatch(
                                            cycleClipTakesRemote({
                                                clipIds: [clip.id],
                                                direction: -1,
                                            }),
                                        );
                                        close();
                                    }}
                                />
                                <MenuItem
                                    label={t("clip_take_cycle_next")}
                                    onClick={() => {
                                        void dispatch(
                                            cycleClipTakesRemote({
                                                clipIds: [clip.id],
                                                direction: 1,
                                            }),
                                        );
                                        close();
                                    }}
                                />
                            </>
                        )}
                        <Divider />
                        <MenuItem
                            label={t("clip_take_add")}
                            onClick={() => {
                                void (async () => {
                                    const picked = await webApi.openAudioDialog();
                                    const path =
                                        picked && typeof picked === "object" && "path" in picked
                                            ? String((picked as { path?: unknown }).path ?? "")
                                            : "";
                                    if (path) {
                                        void dispatch(
                                            addClipTakeFromMediaRemote({
                                                clipId: clip.id,
                                                sourcePath: path,
                                            }),
                                        );
                                    }
                                })();
                                close();
                            }}
                        />
                        <MenuItem
                            label={t("clip_take_duplicate")}
                            disabled={!activeTake}
                            onClick={() => {
                                if (!activeTake) return;
                                void dispatch(
                                    duplicateClipTakeRemote({
                                        clipId: clip.id,
                                        takeId: activeTake.id,
                                    }),
                                );
                                close();
                            }}
                        />
                        <MenuItem
                            label={t("clip_take_rename")}
                            disabled={!activeTake}
                            onClick={() => {
                                if (!activeTake) return;
                                // 内联输入替代 window.prompt：Tauri/WKWebView 下
                                // 脚本对话框普遍不可用（静默返回 null），且会同步
                                // 阻塞 UI 线程。
                                setTakeRenameDraft({
                                    takeId: activeTake.id,
                                    value: activeTake.name || "",
                                });
                            }}
                        />
                        {takeRenameDraft && (
                            <div className="px-3 py-1.5" onPointerDown={(e) => e.stopPropagation()}>
                                <input
                                    autoFocus
                                    role="menuitem"
                                    aria-label={t("clip_take_rename")}
                                    className="w-full bg-qt-window text-[12px] border border-qt-border rounded px-2 py-1 outline-none focus:border-qt-highlight text-qt-text"
                                    value={takeRenameDraft.value}
                                    onChange={(e) =>
                                        setTakeRenameDraft({
                                            ...takeRenameDraft,
                                            value: e.target.value,
                                        })
                                    }
                                    onKeyDown={(e) => {
                                        // 先于窗口级 Escape 关闭处理。
                                        e.stopPropagation();
                                        if (e.key === "Enter") {
                                            const next = takeRenameDraft.value.trim();
                                            if (next) {
                                                void dispatch(
                                                    renameClipTakeRemote({
                                                        clipId: clip.id,
                                                        takeId: takeRenameDraft.takeId,
                                                        name: next,
                                                    }),
                                                );
                                            }
                                            setTakeRenameDraft(null);
                                            close();
                                        } else if (e.key === "Escape") {
                                            setTakeRenameDraft(null);
                                        }
                                    }}
                                />
                            </div>
                        )}
                        <MenuItem
                            label={t("clip_take_remove")}
                            danger
                            disabled={takes.length <= 1 || !activeTake}
                            onClick={() => {
                                if (!activeTake) return;
                                void dispatch(
                                    removeClipTakeRemote({
                                        clipId: clip.id,
                                        takeId: activeTake.id,
                                    }),
                                );
                                close();
                            }}
                        />
                        {takes.length > 1 && (
                            <MenuItem
                                label={t("clip_take_explode")}
                                onClick={() => {
                                    void dispatch(explodeClipTakesRemote({ clipId: clip.id }));
                                    close();
                                }}
                            />
                        )}
                    </>
                )}
            </SubMenu>
            <MenuItem
                label={
                    allReversed
                        ? isMulti
                            ? t("ctx_unreverse_selected")
                            : t("ctx_unreverse")
                        : isMulti
                          ? t("ctx_reverse_selected")
                          : t("ctx_reverse")
                }
                onClick={() => {
                    onToggleReverse(ids, !allReversed);
                    close();
                }}
            />
            {onToggleLoop && (
                <MenuItem
                    label={
                        allLooped
                            ? isMulti
                                ? t("ctx_unloop_selected")
                                : t("ctx_unloop")
                            : isMulti
                              ? t("ctx_loop_selected")
                              : t("ctx_loop")
                    }
                    onClick={() => {
                        onToggleLoop(ids, !allLooped);
                        close();
                    }}
                />
            )}
            {isSingle && (
                <MenuItem
                    label={t("ctx_rename")}
                    onClick={() => {
                        onRename(clip.id);
                        close();
                    }}
                />
            )}
            <MenuItem
                label={isMulti ? t("ctx_copy_all") : t("ctx_copy")}
                onClick={() => {
                    onCopy(ids);
                    close();
                }}
            />
            <MenuItem
                label={isMulti ? t("ctx_cut_all") : t("ctx_cut")}
                onClick={() => {
                    onCut(ids);
                    close();
                }}
            />
            {!allPitchAdjustment && (
                <MenuItem
                    label={isMulti ? t("ctx_replace_all") : t("ctx_replace")}
                    onClick={() => {
                        onReplace(hasPitchAdjustment ? audioOnlyIds : ids);
                        close();
                    }}
                />
            )}
            {hasPitchAdjustment && onReplaceMidi && (
                <MenuItem
                    label={isMulti ? t("ctx_replace_midi_all") : t("ctx_replace_midi")}
                    onClick={() => {
                        onReplaceMidi(pitchOnlyIds);
                        close();
                    }}
                />
            )}
            {!allPitchAdjustment && (
                <MenuItem
                    label={t("ctx_quick_export")}
                    onClick={() => {
                        onQuickExport(hasPitchAdjustment ? audioOnlyIds : ids);
                        close();
                    }}
                />
            )}
            <MenuItem
                label={t("ctx_split_at_playhead")}
                disabled={isMulti ? !canSplitSelected : !playheadInClip}
                onClick={() => {
                    onSplit(ids);
                    close();
                }}
            />
            <MenuItem
                label={isMulti ? t("ctx_normalize_all") : t("ctx_normalize")}
                shortcut={normalizeShortcut}
                onClick={() => {
                    onNormalize(ids);
                    close();
                }}
            />

            {(isMulti || hasGroup) && (
                <>
                    <Divider />
                    {isMulti && !hasGroup && (
                        <MenuItem
                            label={t("group")}
                            onClick={() => {
                                onGroup?.(ids);
                                close();
                            }}
                        />
                    )}
                    {hasGroup && (
                        <MenuItem
                            label={t("ungroup")}
                            onClick={() => {
                                onUngroup?.(ids);
                                close();
                            }}
                        />
                    )}
                </>
            )}
            {isMulti && (
                <MenuItem
                    label={t("glue")}
                    disabled={glueDisabled}
                    onClick={() => {
                        onGlue(ids);
                        close();
                    }}
                />
            )}

            {!allPitchAdjustment && (
                <>
                    <Divider />
                    <MenuItem
                        label={t("ctx_convert_to_pitch_ref")}
                        onClick={() => {
                            const audioIds = selectedClips
                                .filter((c) => !isPitch(c))
                                .map((c) => c.id);
                            if (audioIds.length > 0) {
                                onConvertToPitchRef?.(audioIds);
                            }
                            close();
                        }}
                    />
                </>
            )}

            {allPitchAdjustment && onUpdatePitchRef && (
                <>
                    <Divider />
                    <MenuItem
                        label={t("ctx_update_pitch_ref")}
                        onClick={() => {
                            if (pitchOnlyIds.length > 0) {
                                onUpdatePitchRef(pitchOnlyIds);
                            }
                            close();
                        }}
                    />
                </>
            )}

            {onExportMidi && (
                <MenuItem
                    label={t("ctx_export_midi")}
                    onClick={() => {
                        onExportMidi(ids);
                        close();
                    }}
                />
            )}

            {onFadeShapeChange &&
                (() => {
                    const fadedClips = isSingle
                        ? sortAndFilterFadedClips({
                              clip,
                              overlappingClips,
                          })
                        : sortAndFilterFadedClips({
                              clip: selectedClips[0] ?? clip,
                              overlappingClips: selectedClips.slice(1),
                          });
                    if (fadedClips.length === 0) return null;

                    const showHeader = isMulti || fadedClips.length > 1;

                    return (
                        <>
                            <Divider />
                            {showHeader && (
                                <div className="px-3 py-1 text-[11px] text-qt-text/50 select-none">
                                    {isMulti
                                        ? t("ctx_selected_n").replace(
                                              "{n}",
                                              String(fadedClips.length),
                                          )
                                        : t("overlapping_clips_header").replace(
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
                                    {effectiveFadeSecondsOf(fc).in > 0 && (
                                        <FadeShapeRow
                                            label={t("fade_in")}
                                            current={Number.isFinite(fc.fadeInShape) ? fc.fadeInShape : 0}
                                            isOut={false}
                                            onSelect={(shape) => {
                                                onFadeShapeChange?.(fc.id, "in", shape);
                                            }}
                                            t={t}
                                        />
                                    )}
                                    {effectiveFadeSecondsOf(fc).out > 0 && (
                                        <FadeShapeRow
                                            label={t("fade_out")}
                                            current={Number.isFinite(fc.fadeOutShape) ? fc.fadeOutShape : 0}
                                            isOut={true}
                                            onSelect={(shape) => {
                                                onFadeShapeChange?.(fc.id, "out", shape);
                                            }}
                                            t={t}
                                        />
                                    )}
                                </React.Fragment>
                            ))}
                        </>
                    );
                })()}
        </div>
    );
};
