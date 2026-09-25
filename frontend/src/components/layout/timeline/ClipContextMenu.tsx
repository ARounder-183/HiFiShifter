import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { FadeShapeIcon } from "./FadeShapeIcon";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import { useI18n } from "../../../i18n/I18nProvider";
import type { MessageKey } from "../../../i18n/messages";
import { useAppDispatch, useAppSelector } from "../../../app/hooks";
import { selectKeybinding, formatKeybinding } from "../../../features/keybindings/keybindingsSlice";
import type { ActionId } from "../../../features/keybindings/types";
import {
    addClipTakeFromMediaRemote,
    cycleClipTakesRemote,
    duplicateClipTakeRemote,
    explodeClipTakesRemote,
    packClipsIntoTakesRemote,
    removeClipTakeRemote,
    renameClipTakeRemote,
    setClipActiveTakeRemote,
    setClipTakeChannelModeRemote,
    setClipTakeReversedRemote,
} from "../../../features/session/sessionSlice";
import {
    CHANNEL_MODE_OPTIONS,
    channelModeI18nKey,
    channelModeShortLabel,
    normalizeChannelMode,
    nextChannelMode,
} from "../../../utils/channelMode";
import { webApi } from "../../../services/webviewApi";
import { sharedFadeShape, sortAndFilterFadedClips } from "./clipFadeContext";

// ── 单条菜单项 ──────────────────────────────────────────────────────────────
const MenuItem: React.FC<{
    label: string;
    shortcut?: string;
    disabled?: boolean;
    danger?: boolean;
    /** 悬停 / 禁用原因提示。 */
    title?: string;
    onClick: () => void;
}> = ({ label, shortcut, disabled, danger, title, onClick }) => (
    <button
        role="menuitem"
        data-tooltip={title}
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

/**
 * 读取动作当前生效的快捷键文本（跟随用户在快捷键设置中的自定义绑定）。
 * 未绑定（None binding）时返回 undefined，菜单项不显示快捷键。
 */
function useMenuShortcut(actionId: ActionId): string | undefined {
    const kb = useAppSelector((state) => selectKeybinding(state, actionId));
    return formatKeybinding(kb, "") || undefined;
}

/**
 * Take 行菜单项：点击行切换 active take；行尾“倒放”小按钮翻转**该 Take
 * 自身**的播放方向 —— 不切换 active take、不受“同步编辑所有 Take”设置
 * 影响（对单个 Take 的内容操作，窗口换算由后端
 * `set_clip_take_reversed` 按消费窗口完成）。
 */
const TakeMenuItem: React.FC<{
    label: string;
    disabled?: boolean;
    reversed: boolean;
    reverseLabel: string;
    /** 声道模式（0..=4，对齐 REAPER CHANMODE）；MIDI take 等无声道语义时省略按钮。 */
    channelMode?: number;
    modeLabel?: string;
    modeTitle?: string;
    onSwitch: () => void;
    onToggleReverse: () => void;
    onCycleChannelMode?: () => void;
}> = ({
    label,
    disabled = false,
    reversed,
    reverseLabel,
    channelMode,
    modeLabel,
    modeTitle,
    onSwitch,
    onToggleReverse,
    onCycleChannelMode,
}) => (
    // 行内布局：标签 flex-1 + 尾随两个 shrink-0 按钮（flex 兄弟，绝不定
    // 位）—— 任何语言下按钮互不重叠、不挤压标签；标签超长时 truncate
    // 兜底（面板宽度已随内容展开，见 SubMenu 的 width:max-content）。
    <div className="flex items-center w-full gap-1 pr-1.5">
        <button
            role="menuitem"
            className={`px-3 py-1.5 text-left flex-1 min-w-0 text-[12px] transition-colors rounded
                ${disabled ? "opacity-40 cursor-default" : "hover:bg-qt-button-hover"}`}
            disabled={disabled}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
                e.stopPropagation();
                onSwitch();
            }}
        >
            <span className="block truncate">{label}</span>
        </button>
        {onCycleChannelMode != null && modeLabel != null && (
            <button
                role="menuitem"
                aria-label={`${modeTitle ?? modeLabel}: ${label}`}
                data-tooltip={modeTitle}
                className={`shrink-0 px-1.5 py-0.5 text-[10px] leading-none rounded border transition-colors
                    ${
                        (channelMode ?? 0) !== 0
                            ? "border-qt-highlight text-qt-highlight"
                            : "border-qt-border text-qt-text-muted opacity-70"
                    } hover:bg-qt-button-hover`}
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => {
                    e.stopPropagation();
                    onCycleChannelMode();
                }}
            >
                {modeLabel}
            </button>
        )}
        <button
            role="menuitem"
            aria-label={`${reverseLabel}: ${label}`}
            data-tooltip={reverseLabel}
            className={`shrink-0 px-1.5 py-0.5 text-[10px] leading-none rounded border transition-colors
                ${
                    reversed
                        ? "border-qt-highlight text-qt-highlight"
                        : "border-qt-border text-qt-text-muted opacity-70"
                } hover:bg-qt-button-hover`}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
                e.stopPropagation();
                onToggleReverse();
            }}
        >
            {reverseLabel}
        </button>
    </div>
);

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
        // 宽度随内容展开：绝对定位面板的宽度默认被包含块（触发项宽度）封顶，
        // Take 行等长文本会因此换行。max-content 展开后若超出视口，按最终
        // 锚定侧的可用空间收口 —— 行内标签以 truncate 兜底。
        panel.style.width = "max-content";
        panel.style.maxWidth = "none";

        const vw = window.innerWidth;
        const vh = window.innerHeight;
        let rect = panel.getBoundingClientRect();
        if (rect.right > vw - 4) {
            panel.style.left = "auto";
            panel.style.right = "calc(100% - 4px)";
        }
        rect = panel.getBoundingClientRect();
        const anchoredLeft = panel.style.left !== "auto";
        const availableWidth = anchoredLeft ? vw - 8 - rect.left : rect.right - 8;
        if (rect.width > availableWidth) {
            panel.style.maxWidth = `${Math.max(160, Math.floor(availableWidth))}px`;
        }

        rect = panel.getBoundingClientRect();
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
    /** 当前形状；`null` = 各 Clip 不一致，不预选任何一项。 */
    current: number | null;
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
                        current !== null && Math.trunc(current) === opt.shape
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
    /** 把所选 Clip 的时间范围并入参数编辑器选区（只作用于同一根轨道组）。 */
    onAddToParamSelection?: (ids: string[]) => void;
    onNormalize: (ids: string[]) => void;
    /** 打开"静音检测"对话框（多选时作用于全部所选 Clip 中含音频源者）。 */
    onSilenceDetection?: (ids: string[]) => void;
    onToggleReverse: (ids: string[], reversed: boolean) => void;
    onToggleLoop?: (ids: string[], loopEnabled: boolean) => void;
    /**
     * 批量设置所选 Clip 的声道模式（作用范围跟随全局"同步编辑所有 Take"设置）。
     */
    onSetChannelMode?: (ids: string[], mode: number) => void;
    /** 扫描并把所选 Clip 里的"假立体声"Take 折叠为单声道。 */
    onScanFakeStereo?: (ids: string[]) => void;
    /**
     * 切换淡入/淡出的 REAPER 形状预设（保留曲率 dir 不变）。
     *
     * 接收**一组 Clip**：多选时形状行只给一行，选择即批量应用到全部所选 Clip
     * （单次 IPC + 单个撤销步）；单选时传入长度为 1 的数组。
     */
    onFadeShapeChange?: (clipIds: string[], target: "in" | "out", shape: number) => void;
    /** 打开"编辑播放速率"浮层（锚点 = 菜单位置）。与倍率角标右键同一浮层；
     *  多选时以右键的 clip 为 anchor 批量应用（提交管线内聚）。 */
    onEditRate?: (clipId: string, screenX: number, screenY: number) => void;
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
    onAddToParamSelection,
    onNormalize,
    onSilenceDetection,
    onToggleReverse,
    onToggleLoop,
    onSetChannelMode,
    onScanFakeStereo,
    onFadeShapeChange,
    onEditRate,
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

    // 菜单项右侧的快捷键提示：从快捷键注册表读取当前生效的绑定
    //（用户自定义后菜单同步跟随），未绑定动作的菜单项不显示。
    const normalizeShortcut = useMenuShortcut("clip.normalize");
    const deleteShortcut = useMenuShortcut("clip.delete");
    const copyShortcut = useMenuShortcut("clip.copy");
    const cutShortcut = useMenuShortcut("clip.cut");
    const splitShortcut = useMenuShortcut("clip.split");
    const groupShortcut = useMenuShortcut("clip.group");
    const ungroupShortcut = useMenuShortcut("clip.ungroup");
    const cycleTakeNextShortcut = useMenuShortcut("clip.cycleTake");
    const cycleTakePrevShortcut = useMenuShortcut("clip.cycleTakePrev");
    const addToParamSelectionShortcut = useMenuShortcut("edit.addClipsToParamSelection");

    // 胶合：仅同轨且多选时可用，且不能混合音高参考块和常规音频块
    const hasMixedTypes = hasPitchAdjustment && !allPitchAdjustment;
    const glueDisabled =
        !isMulti ||
        hasMixedTypes ||
        (() => {
            const trackId = selectedClips[0]?.trackId;
            return !trackId || selectedClips.some((c) => c.trackId !== trackId);
        })();

    // 静音检测：至少一个目标 Clip 的活跃 Take 含音频源才可用。
    const hasAudioTake = (c: ClipInfo) => (c.takes ?? []).some((tk) => !!tk.sourcePath);
    const silenceEligible = isMulti ? selectedClips.some(hasAudioTake) : hasAudioTake(clip);

    // 多选中是否全部静音
    const allMuted = isMulti ? selectedClips.every((c) => c.muted) : clip.muted;
    const allReversed = isMulti ? selectedClips.every((c) => c.reversed) : clip.reversed;
    // 多选中是否已全部启用 Loop（循环源）
    const allLooped = isMulti ? selectedClips.every((c) => c.loopEnabled) : clip.loopEnabled;
    // 多选中共同的声道模式（不一致时为 null，子菜单不显示选中标记）。
    const commonChannelMode = (() => {
        const modes = new Set(selectedClips.map((c) => normalizeChannelMode(c.channelMode)));
        return modes.size === 1 ? normalizeChannelMode(clip.channelMode) : null;
    })();

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
            className="fixed z-[999] min-w-[140px] rounded border border-qt-border bg-qt-window text-qt-text shadow-lg py-1"
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
                shortcut={deleteShortcut}
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
                            <TakeMenuItem
                                key={take.id}
                                // 活跃 take 用 ● 标记；非活跃用 em-space（U+2003，
                                // 不会被 HTML 空白折叠）保持对齐。
                                label={`${
                                    take.id === clip.activeTakeId
                                        ? t("clip_take_active_mark")
                                        : "\u2003"
                                } ${take.name || take.id}`}
                                disabled={takes.length <= 1}
                                reversed={Boolean(take.reversed)}
                                reverseLabel={t("clip_take_reverse")}
                                {...(take.sourcePath
                                    ? {
                                          channelMode: take.channelMode,
                                          modeLabel: channelModeShortLabel(take.channelMode),
                                          modeTitle: `${t("clip_take_channel_mode")}: ${t(
                                              channelModeI18nKey(take.channelMode),
                                          )} (${t("clip_take_channel_mode_cycle")})`,
                                          onCycleChannelMode: () => {
                                              // 不关闭菜单：连续切换声道无需反复
                                              // 重开；乐观更新即时刷新按钮状态。
                                              void dispatch(
                                                  setClipTakeChannelModeRemote({
                                                      clipId: clip.id,
                                                      takeId: take.id,
                                                      channelMode: nextChannelMode(
                                                          take.channelMode,
                                                      ),
                                                  }),
                                              );
                                          },
                                      }
                                    : {})}
                                onSwitch={() => {
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
                                onToggleReverse={() => {
                                    // 不关闭菜单：与声道按钮同口径，连续翻转。
                                    void dispatch(
                                        setClipTakeReversedRemote({
                                            clipId: clip.id,
                                            takeId: take.id,
                                            reversed: !take.reversed,
                                        }),
                                    );
                                }}
                            />
                        ))}
                        {takes.length > 1 && (
                            <>
                                <Divider />
                                <MenuItem
                                    label={t("clip_take_cycle_prev")}
                                    shortcut={cycleTakePrevShortcut}
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
                                    shortcut={cycleTakeNextShortcut}
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
                {/* 替换素材：作用对象是**活跃 Take** 的源媒体，因此归入 Take 范畴
                    （与"添加媒体为 Take"相邻）。多选时对每个 Clip 的活跃 Take 生效。 */}
                {(!allPitchAdjustment || (hasPitchAdjustment && onReplaceMidi)) && <Divider />}
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
            </SubMenu>
            {(onSetChannelMode || onScanFakeStereo) && (
                <SubMenu
                    label={t("ctx_channel_mode")}
                    badge={
                        commonChannelMode === null
                            ? undefined
                            : channelModeShortLabel(commonChannelMode)
                    }
                >
                    {CHANNEL_MODE_OPTIONS.map((option) => (
                        <MenuItem
                            key={option.value}
                            // ● 标记当前共同模式；多选模式不一致时全部留空。
                            label={`${commonChannelMode === option.value ? "●" : "\u2003"} ${t(
                                option.i18nKey,
                            )}`}
                            shortcut={option.shortLabel}
                            onClick={() => {
                                onSetChannelMode?.(ids, option.value);
                                close();
                            }}
                        />
                    ))}
                    {onScanFakeStereo && (
                        <>
                            <Divider />
                            <MenuItem
                                label={t("ctx_scan_fake_stereo")}
                                title={t("ctx_scan_fake_stereo_hint")}
                                onClick={() => {
                                    onScanFakeStereo(ids);
                                    close();
                                }}
                            />
                        </>
                    )}
                </SubMenu>
            )}
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
            <Divider />
            <MenuItem
                label={isMulti ? t("ctx_copy_all") : t("ctx_copy")}
                shortcut={copyShortcut}
                onClick={() => {
                    onCopy(ids);
                    close();
                }}
            />
            <MenuItem
                label={isMulti ? t("ctx_cut_all") : t("ctx_cut")}
                shortcut={cutShortcut}
                onClick={() => {
                    onCut(ids);
                    close();
                }}
            />
            <MenuItem
                label={t("ctx_split_at_playhead")}
                shortcut={splitShortcut}
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
            {onAddToParamSelection && (
                <MenuItem
                    label={t("ctx_add_to_param_selection")}
                    shortcut={addToParamSelectionShortcut}
                    onClick={() => {
                        onAddToParamSelection(ids);
                        close();
                    }}
                />
            )}
            {/* ── 内容工具 ───────────────────────────────────────────────
                「播放速率」「静音检测」「音高参考」「导出」都是**单个动作**，
                折叠成子菜单只会多一次悬停/点击，因此平铺在一级。每段自带
                前置分隔线，段不存在时不留空分隔。 */}
            {(onEditRate || onSilenceDetection) && (
                <>
                    <Divider />
                    {onEditRate && (
                        <MenuItem
                            label={t("ctx_edit_rate")}
                            onClick={() => {
                                // 锚点 = 菜单弹出位置：菜单关闭后浮层原地展开。
                                // 多选时右键的 clip 即 anchor（提交走 getBulkEditableClipIds 批量管线）。
                                onEditRate(clip.id, x, y);
                                close();
                            }}
                        />
                    )}
                    {onSilenceDetection && (
                        <MenuItem
                            label={t("ctx_silence_detection")}
                            disabled={!silenceEligible}
                            data-tooltip={
                                silenceEligible ? undefined : t("silence_no_audio_source")
                            }
                            onClick={() => {
                                onSilenceDetection(ids);
                                close();
                            }}
                        />
                    )}
                </>
            )}
            {((!allPitchAdjustment && onConvertToPitchRef) ||
                (allPitchAdjustment && onUpdatePitchRef)) && (
                <>
                    <Divider />
                    {!allPitchAdjustment && onConvertToPitchRef && (
                        <MenuItem
                            label={t("ctx_convert_to_pitch_ref")}
                            onClick={() => {
                                const audioIds = selectedClips
                                    .filter((c) => !isPitch(c))
                                    .map((c) => c.id);
                                if (audioIds.length > 0) {
                                    onConvertToPitchRef(audioIds);
                                }
                                close();
                            }}
                        />
                    )}
                    {allPitchAdjustment && onUpdatePitchRef && (
                        <MenuItem
                            label={t("ctx_update_pitch_ref")}
                            onClick={() => {
                                if (pitchOnlyIds.length > 0) {
                                    onUpdatePitchRef(pitchOnlyIds);
                                }
                                close();
                            }}
                        />
                    )}
                </>
            )}
            {(!allPitchAdjustment || onExportMidi) && (
                <>
                    <Divider />
                    {!allPitchAdjustment && (
                        <MenuItem
                            label={t("ctx_quick_export")}
                            onClick={() => {
                                onQuickExport(hasPitchAdjustment ? audioOnlyIds : ids);
                                close();
                            }}
                        />
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
                </>
            )}
            {(isMulti || hasGroup) && (
                <>
                    <Divider />
                    <SubMenu label={t("ctx_group")}>
                        {isMulti && !hasGroup && (
                            <MenuItem
                                label={t("group")}
                                shortcut={groupShortcut}
                                onClick={() => {
                                    onGroup?.(ids);
                                    close();
                                }}
                            />
                        )}
                        {hasGroup && (
                            <MenuItem
                                label={t("ungroup")}
                                shortcut={ungroupShortcut}
                                onClick={() => {
                                    onUngroup?.(ids);
                                    close();
                                }}
                            />
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
                    </SubMenu>
                </>
            )}

            {onFadeShapeChange &&
                (() => {
                    // 多选：**每个方向只给一行**，选择即批量应用到全部所选 Clip。
                    // 旧实现逐个 Clip 列举（还带名字表头），选项行数随选择数线性
                    // 膨胀，实际使用时几乎不可读，且"给某一个 Clip 单独换形状"
                    // 在多选语境下并无意义。
                    if (isMulti) {
                        const fadeInTargets = selectedClips.filter(
                            (c) => effectiveFadeSecondsOf(c).in > 0,
                        );
                        const fadeOutTargets = selectedClips.filter(
                            (c) => effectiveFadeSecondsOf(c).out > 0,
                        );
                        if (fadeInTargets.length === 0 && fadeOutTargets.length === 0) {
                            return null;
                        }
                        return (
                            <>
                                <Divider />
                                {fadeInTargets.length > 0 && (
                                    <FadeShapeRow
                                        label={t("fade_in")}
                                        // 各 Clip 形状不一致时不预选任何一项（null）。
                                        current={sharedFadeShape(fadeInTargets, "in")}
                                        isOut={false}
                                        onSelect={(shape) => {
                                            onFadeShapeChange(ids, "in", shape);
                                        }}
                                        t={t}
                                    />
                                )}
                                {fadeOutTargets.length > 0 && (
                                    <FadeShapeRow
                                        label={t("fade_out")}
                                        current={sharedFadeShape(fadeOutTargets, "out")}
                                        isOut={true}
                                        onSelect={(shape) => {
                                            onFadeShapeChange(ids, "out", shape);
                                        }}
                                        t={t}
                                    />
                                )}
                            </>
                        );
                    }

                    // 单选：保持"本 Clip + 相邻重叠 Clip 各自一行"。这里逐个指定
                    // 形状是有意义的 —— 交叉淡化两侧通常要用互补的曲线。
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
                                    {effectiveFadeSecondsOf(fc).in > 0 && (
                                        <FadeShapeRow
                                            label={t("fade_in")}
                                            current={
                                                Number.isFinite(fc.fadeInShape) ? fc.fadeInShape : 0
                                            }
                                            isOut={false}
                                            onSelect={(shape) => {
                                                onFadeShapeChange([fc.id], "in", shape);
                                            }}
                                            t={t}
                                        />
                                    )}
                                    {effectiveFadeSecondsOf(fc).out > 0 && (
                                        <FadeShapeRow
                                            label={t("fade_out")}
                                            current={
                                                Number.isFinite(fc.fadeOutShape)
                                                    ? fc.fadeOutShape
                                                    : 0
                                            }
                                            isOut={true}
                                            onSelect={(shape) => {
                                                onFadeShapeChange([fc.id], "out", shape);
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
