// 片段右键菜单：**内容**由 clipMenuModel 决定（纯数据），本文件只负责画出来。
//
// 【为什么菜单里没有第二行说明】见方案 §3.1.1 的 D1.1：结构性不可用的项**不渲染**，
// 情境性不可用的项只用 disabled 表示 —— 两者都不写「为什么」。原因在这里对用户
// 不可操作（换不了设置、换不了宿主），代价却是整张菜单被撑成两倍高。
import React, { useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import {
    isPluginMode,
    canSplitHostClips,
    canClipboardHostClips,
    canEditHostClips,
    canEditFadeShape,
    canImportMidiAsClip,
    canEditHostTakes,
    canGlueClips,
    canDetectSilence,
    canScanFakeStereo,
    canReverseHostClips,
    canBridgeHostTakePitch,
    canExportAudio,
} from "../../../services/hostCapabilities";
import { FadeShapeIcon } from "./FadeShapeIcon";
import type { ClipInfo } from "../../../features/session/sessionTypes";

import { AppSubMenu } from "../../../ui";
import { useMenuKeyboard } from "../../../ui/useMenuKeyboard";
import { useI18n } from "../../../i18n/I18nProvider";
import type { MessageKey } from "../../../i18n/messages";
import { useAppDispatch } from "../../../app/hooks";
import { useMenuShortcut } from "../../../ui/useMenuShortcut";
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
    channelModeI18nKey,
    channelModeShortLabel,
    nextChannelMode,
} from "../../../utils/channelMode";
import { webApi } from "../../../services/webviewApi";
import {
    buildClipMenuModel,
    type ClipMenuEntry,
    type ClipMenuFadeRow,
    type ClipMenuTakeRow,
} from "./clipMenuModel";

// ── 单条菜单项 ──────────────────────────────────────────────────────────────
// 壳与项的取值全部来自共享样式模型（.hs-menu*，见 src/index.css）——本文件只
// 提供结构与内容。
const MenuItem: React.FC<{
    label: string;
    shortcut?: string;
    disabled?: boolean;
    danger?: boolean;
    onClick: () => void;
}> = ({ label, shortcut, disabled, danger, onClick }) => (
    <button
        role="menuitem"
        disabled={disabled}
        data-danger={danger ? "1" : undefined}
        className="hs-menu__item"
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => {
            e.stopPropagation();
            onClick();
        }}
    >
        <span className="flex min-w-0 flex-col items-start">
            <span className="hs-menu__label-text">{label}</span>
        </span>
        <span className="hs-menu__trail">{shortcut ? <span>{shortcut}</span> : null}</span>
    </button>
);

const Divider: React.FC = () => <div className="hs-menu__separator" role="separator" />;

/**
 * Take 行菜单项：点击行切换 active take；行尾「倒放」小按钮翻转**该 Take 自身**
 * 的播放方向 —— 不切换 active take、不受「同步编辑所有 Take」设置影响（对单个
 * Take 的内容操作，窗口换算由后端 set_clip_take_reversed 按消费窗口完成）。
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
    // 行内布局：标签 flex-1 + 尾随两个 shrink-0 按钮（flex 兄弟，绝不定位）——
    // 任何语言下按钮互不重叠、不挤压标签；标签超长时 truncate 兜底。
    <div className="flex items-center w-full gap-1 pr-1.5">
        <button
            role="menuitem"
            className="hs-menu__item flex-1 min-w-0"
            disabled={disabled}
            onPointerDown={(e) => e.stopPropagation()}
            onClick={(e) => {
                e.stopPropagation();
                onSwitch();
            }}
        >
            <span className="hs-menu__label-text">{label}</span>
        </button>
        {onCycleChannelMode != null && modeLabel != null && (
            <button
                role="menuitem"
                aria-label={String(modeTitle ?? modeLabel) + ": " + label}
                data-tooltip={modeTitle}
                className={
                    "shrink-0 px-1.5 py-0.5 text-qt-micro leading-none rounded border transition-colors " +
                    ((channelMode ?? 0) !== 0
                        ? "border-qt-highlight text-qt-highlight"
                        : "border-qt-border text-qt-text-muted opacity-70") +
                    " hover:bg-qt-menu-item-hover"
                }
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
            aria-label={reverseLabel + ": " + label}
            data-tooltip={reverseLabel}
            className={
                "shrink-0 px-1.5 py-0.5 text-qt-micro leading-none rounded border transition-colors " +
                (reversed
                    ? "border-qt-highlight text-qt-highlight"
                    : "border-qt-border text-qt-text-muted opacity-70") +
                " hover:bg-qt-menu-item-hover disabled:opacity-40"
            }
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
    /** 当前形状；null = 各 Clip 不一致，不预选任何一项。 */
    current: number | null;
    /** 本行是淡出（图标水平镜像，曲线方向与画布一致）。 */
    isOut?: boolean;
    onSelect: (shape: number) => void;
    t: (key: MessageKey) => string;
}> = ({ label, current, isOut = false, onSelect, t }) => (
    <div className="hs-menu__body flex items-center gap-1 flex-wrap">
        <span className="hs-type-caption mr-1 shrink-0">{label}</span>
        {FADE_SHAPE_OPTIONS.map((opt) => (
            <button
                key={opt.key}
                data-tooltip={t(opt.key)}
                className={
                    "p-0.5 rounded transition-colors leading-none " +
                    // 小数变体（如 1.1）按基础族高亮（REAPER 同语义）。
                    (current !== null && Math.trunc(current) === opt.shape
                        ? "bg-qt-highlight text-white"
                        : "bg-qt-button hover:bg-qt-button-hover text-qt-text/80")
                }
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
    /** 打开「静音检测」对话框（多选时作用于全部所选 Clip 中含音频源者）。 */
    onSilenceDetection?: (ids: string[]) => void;
    onToggleReverse: (ids: string[], reversed: boolean) => void;
    onToggleLoop?: (ids: string[], loopEnabled: boolean) => void;
    /** 批量设置所选 Clip 的声道模式（作用范围跟随全局「同步编辑所有 Take」设置）。 */
    onSetChannelMode?: (ids: string[], mode: number) => void;
    /** 扫描并把所选 Clip 里的「假立体声」Take 折叠为单声道。 */
    onScanFakeStereo?: (ids: string[]) => void;
    /**
     * 切换淡入/淡出的 REAPER 形状预设（保留曲率 dir 不变）。
     *
     * 接收**一组 Clip**：多选时形状行只给一行，选择即批量应用到全部所选 Clip
     * （单次 IPC + 单个撤销步）；单选时传入长度为 1 的数组。
     */
    onFadeShapeChange?: (clipIds: string[], target: "in" | "out", shape: number) => void;
    /** 打开「编辑播放速率」浮层（锚点 = 菜单位置）。与倍率角标右键同一浮层；
     *  多选时以右键的 clip 为 anchor 批量应用（提交管线内聚）。 */
    onEditRate?: (clipId: string, screenX: number, screenY: number) => void;
    /** REAPER take 音高 → HiFiShifter 音高参数线（一次性迁移）。 */
    onImportHostPitch?: (ids: string[]) => void;
    /** HiFiShifter 音高参数线 → REAPER take 音高包络（一次性迁移）。 */
    onExportHostPitch?: (ids: string[]) => void;
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
    onImportHostPitch,
    onExportHostPitch,
}) => {
    const { t } = useI18n();
    const dispatch = useAppDispatch();
    const menuRef = useRef<HTMLDivElement>(null);
    useMenuKeyboard(menuRef);
    /** Take 重命名的内联输入草稿（替代 window.prompt）。 */
    const [takeRenameDraft, setTakeRenameDraft] = useState<{
        takeId: string;
        value: string;
    } | null>(null);

    const ids = selectedClips.length >= 2 ? selectedClips.map((c) => c.id) : [clip.id];
    const isMulti = ids.length >= 2;
    const allMuted = isMulti ? selectedClips.every((c) => c.muted) : clip.muted;
    const allReversed = isMulti ? selectedClips.every((c) => c.reversed) : clip.reversed;
    const allLooped = isMulti ? selectedClips.every((c) => c.loopEnabled) : clip.loopEnabled;

    const audioOnlyIds = selectedClips.filter((c) => c.midiNoteCount == null).map((c) => c.id);
    const pitchOnlyIds = selectedClips.filter((c) => c.midiNoteCount != null).map((c) => c.id);
    const anyPitch = pitchOnlyIds.length > 0;

    // 菜单项右侧的快捷键提示：从快捷键注册表读取当前生效的绑定
    //（用户自定义后菜单同步跟随），未绑定动作的菜单项不显示。
    const shortcuts = {
        normalize: useMenuShortcut("clip.normalize"),
        delete: useMenuShortcut("clip.delete"),
        copy: useMenuShortcut("clip.copy"),
        cut: useMenuShortcut("clip.cut"),
        split: useMenuShortcut("clip.split"),
        group: useMenuShortcut("clip.group"),
        ungroup: useMenuShortcut("clip.ungroup"),
        cycleTakeNext: useMenuShortcut("clip.cycleTake"),
        cycleTakePrev: useMenuShortcut("clip.cycleTakePrev"),
        addToParamSelection: useMenuShortcut("edit.addClipsToParamSelection"),
    };

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
        if (rect.right > vw) el.style.left = Math.max(0, vw - rect.width) + "px";
        if (rect.bottom > vh) el.style.top = Math.max(0, vh - rect.height) + "px";
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

    const entries = buildClipMenuModel({
        clip,
        selectedClips,
        overlappingClips,
        playheadInClip,
        canSplitSelected,
        pluginMode: isPluginMode(),
        caps: {
            editClips: canEditHostClips(),
            clipboard: canClipboardHostClips(),
            split: canSplitHostClips(),
            takes: canEditHostTakes(),
            glue: canGlueClips(),
            silenceDetection: canDetectSilence(),
            fakeStereoScan: canScanFakeStereo(),
            reverse: canReverseHostClips(),
            quickExport: canExportAudio(),
            pitchBridge:
                canBridgeHostTakePitch() &&
                (onImportHostPitch != null || onExportHostPitch != null),
            midiAsClip: canImportMidiAsClip(),
            fadeShape: canEditFadeShape(),
        },
        available: {
            replaceMidi: onReplaceMidi != null,
            group: onGroup != null,
            ungroup: onUngroup != null,
            convertToPitchRef: onConvertToPitchRef != null,
            updatePitchRef: onUpdatePitchRef != null,
            exportMidi: onExportMidi != null,
            addToParamSelection: onAddToParamSelection != null,
            silenceDetection: onSilenceDetection != null,
            scanFakeStereo: onScanFakeStereo != null,
            toggleLoop: onToggleLoop != null,
            channelMode: onSetChannelMode != null,
            editRate: onEditRate != null,
            fadeShape: onFadeShapeChange != null,
        },
        shortcuts,
        t,
    });

    /**
     * 动作标识 → 回调。
     *
     * 【为什么按 key 取而不是把闭包塞进模型】模型必须能在 node 环境里被单测，
     * 所以它只产出**数据**（key + 文案 + 禁用态）；"点了做什么"是渲染层的知识。
     */
    const actions: Record<string, () => void> = {
        delete: () => onDelete(ids),
        "delete-all": () => onDelete(ids),
        mute: () => onMute(ids, !allMuted),
        "take-pack": () => void dispatch(packClipsIntoTakesRemote({ clipIds: ids })),
        "take-cycle-prev": () =>
            void dispatch(cycleClipTakesRemote({ clipIds: [clip.id], direction: -1 })),
        "take-cycle-next": () =>
            void dispatch(cycleClipTakesRemote({ clipIds: [clip.id], direction: 1 })),
        "take-add": () => {
            void (async () => {
                const picked = await webApi.openAudioDialog();
                const path =
                    picked && typeof picked === "object" && "path" in picked
                        ? String((picked as { path?: unknown }).path ?? "")
                        : "";
                if (path) {
                    void dispatch(
                        addClipTakeFromMediaRemote({ clipId: clip.id, sourcePath: path }),
                    );
                }
            })();
        },
        "take-duplicate": () => {
            const active =
                clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
            if (!active) return;
            void dispatch(duplicateClipTakeRemote({ clipId: clip.id, takeId: active.id }));
        },
        "take-rename": () => {
            const active =
                clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
            if (!active) return;
            // 内联输入替代 window.prompt：Tauri/WKWebView 下脚本对话框普遍不可用
            //（静默返回 null），且会同步阻塞 UI 线程。
            setTakeRenameDraft({ takeId: active.id, value: active.name || "" });
        },
        "take-remove": () => {
            const active =
                clip.takes?.find((take) => take.id === clip.activeTakeId) ?? clip.takes?.[0];
            if (!active) return;
            void dispatch(removeClipTakeRemote({ clipId: clip.id, takeId: active.id }));
        },
        "take-explode": () => void dispatch(explodeClipTakesRemote({ clipId: clip.id })),
        replace: () => onReplace(anyPitch ? audioOnlyIds : ids),
        "replace-all": () => onReplace(anyPitch ? audioOnlyIds : ids),
        "replace-midi": () => onReplaceMidi?.(pitchOnlyIds),
        "replace-midi-all": () => onReplaceMidi?.(pitchOnlyIds),
        "scan-fake-stereo": () => onScanFakeStereo?.(ids),
        reverse: () => onToggleReverse(ids, !allReversed),
        loop: () => onToggleLoop?.(ids, !allLooped),
        copy: () => onCopy(ids),
        "copy-all": () => onCopy(ids),
        cut: () => onCut(ids),
        "cut-all": () => onCut(ids),
        split: () => onSplit(ids),
        normalize: () => onNormalize(ids),
        "normalize-all": () => onNormalize(ids),
        "add-to-param-selection": () => onAddToParamSelection?.(ids),
        "edit-rate": () => onEditRate?.(clip.id, x, y),
        "silence-detection": () => onSilenceDetection?.(ids),
        "pitch-reference-convert": () => {
            if (audioOnlyIds.length > 0) onConvertToPitchRef?.(audioOnlyIds);
        },
        "pitch-reference-update": () => {
            if (pitchOnlyIds.length > 0) onUpdatePitchRef?.(pitchOnlyIds);
        },
        "quick-export": () => onQuickExport(anyPitch ? audioOnlyIds : ids),
        "export-midi": () => onExportMidi?.(ids),
        group: () => onGroup?.(ids),
        ungroup: () => onUngroup?.(ids),
        glue: () => onGlue(ids),
        "host-pitch-import": () => onImportHostPitch?.(ids),
        "host-pitch-export": () => onExportHostPitch?.(ids),
    };
    // 声道模式是一族同形项：按前缀解析出模式号，避免在模型里塞 5 个闭包。
    const runAction = (key: string) => {
        if (key.startsWith("channel-mode-")) {
            const mode = Number(key.slice("channel-mode-".length));
            onSetChannelMode?.(ids, mode);
            close();
            return;
        }
        if (key === "take-rename") {
            // 重命名要展开内联输入，**不关闭菜单**。
            actions[key]?.();
            return;
        }
        actions[key]?.();
        close();
    };

    const renderTakeRow = (row: ClipMenuTakeRow) => {
        // 行尾两个按钮各自独立：方向翻转**不关闭**菜单（与声道按钮同口径，连续翻转）。
        return (
            <TakeMenuItem
                key={row.key}
                label={row.label}
                disabled={!row.switchable}
                reversed={row.reversed}
                reverseLabel={t("clip_take_reverse")}
                {...(row.channelMode !== null
                    ? {
                          channelMode: row.channelMode,
                          modeLabel: channelModeShortLabel(row.channelMode),
                          modeTitle:
                              t("clip_take_channel_mode") +
                              ": " +
                              t(channelModeI18nKey(row.channelMode)) +
                              " (" +
                              t("clip_take_channel_mode_cycle") +
                              ")",
                          onCycleChannelMode: () => {
                              void dispatch(
                                  setClipTakeChannelModeRemote({
                                      clipId: clip.id,
                                      takeId: row.takeId,
                                      channelMode: nextChannelMode(row.channelMode ?? 0),
                                  }),
                              );
                          },
                      }
                    : {})}
                onSwitch={() => {
                    // 点击已激活的 take 是 no-op：跳过 dispatch，避免无谓的乐观切换 +
                    // 回滚快照 + 全量快照刷新。
                    if (row.takeId !== clip.activeTakeId) {
                        void dispatch(
                            setClipActiveTakeRemote({ clipId: clip.id, takeId: row.takeId }),
                        );
                    }
                    close();
                }}
                onToggleReverse={() => {
                    const take = clip.takes?.find((item) => item.id === row.takeId);
                    if (!take) return;
                    void dispatch(
                        setClipTakeReversedRemote({
                            clipId: clip.id,
                            takeId: row.takeId,
                            reversed: !take.reversed,
                        }),
                    );
                }}
            />
        );
    };

    const renderFadeRow = (row: ClipMenuFadeRow) => (
        <FadeShapeRow
            key={row.key}
            label={row.direction === "in" ? t("fade_in") : t("fade_out")}
            current={row.current}
            isOut={row.direction === "out"}
            onSelect={(shape) => {
                onFadeShapeChange?.(row.clipId === null ? ids : [row.clipId], row.direction, shape);
            }}
            t={t}
        />
    );

    /**
     * Take 重命名的内联输入。
     *
     * 【为什么挂在 Take 子菜单里】它是**该子菜单**的一次改写动作，弹到菜单底部会与
     * 触发它的那一行隔开好几段距离，用户看不出输入框属于谁。
     */
    const renameInput = takeRenameDraft ? (
        <div className="hs-menu__body" onPointerDown={(e) => e.stopPropagation()}>
            <input
                autoFocus
                role="menuitem"
                aria-label={t("clip_take_rename")}
                className="w-full bg-qt-window text-qt-sm border border-qt-border rounded px-2 py-1 outline-hidden focus:border-qt-highlight text-qt-text"
                value={takeRenameDraft.value}
                onChange={(e) => setTakeRenameDraft({ ...takeRenameDraft, value: e.target.value })}
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
    ) : null;

    const renderEntry = (entry: ClipMenuEntry): React.ReactNode => {
        switch (entry.kind) {
            case "separator":
                return <Divider key={entry.key} />;
            case "heading":
                return (
                    <div key={entry.key} className="hs-menu__label">
                        {entry.text}
                    </div>
                );
            case "item":
                return (
                    <MenuItem
                        key={entry.key}
                        label={entry.label}
                        shortcut={entry.shortcut}
                        disabled={entry.disabled}
                        danger={entry.danger}
                        onClick={() => runAction(entry.key)}
                    />
                );
            case "submenu":
                return (
                    <AppSubMenu key={entry.key} label={entry.label} badge={entry.badge}>
                        {entry.children.map(renderEntry)}
                        {entry.key === "takes" ? renameInput : null}
                    </AppSubMenu>
                );
            case "takeRow":
                return renderTakeRow(entry);
            case "fadeRow":
                return renderFadeRow(entry);
            default:
                return null;
        }
    };

    // 菜单挂到 document.body：弹出面留在布局盒里会被沿途任何一层
    // overflow: hidden 裁掉（见 src/index.css 的 .hs-menu--submenu）。
    // 由组件自己 portal（而不是让调用方 portal）—— 表面归组件所有，这样
    // 「菜单表面必须挂到 body」这条不变量对每个菜单都成立、也能被门禁检查。
    return createPortal(
        <div
            ref={menuRef}
            role="menu"
            data-hs-context-menu="1"
            data-hs-floating-menu="1"
            // --no-scroll：本菜单里挂着 AppSubMenu 的子面板，而子面板是**绝对定位**
            // 的后代 —— 壳一旦 overflow-y: auto 就会把它裁掉。子面板自己会滚动。
            className="hs-menu hs-menu--no-scroll"
            style={{ left: x, top: y }}
            onPointerDown={(e) => e.stopPropagation()}
        >
            {entries.map(renderEntry)}
        </div>,
        document.body,
    );
};
