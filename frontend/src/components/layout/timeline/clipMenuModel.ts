// 片段右键菜单的**纯模型**：输入「片段状态 + 能力位」，输出条目树。
//
// 【为什么抽成纯函数】此前这张菜单是「一棵 JSX 树 + 二十来处内联 gate() 调用」：
// 判据散在渲染里，无法在 node 环境下单测，于是「哪些项在插件里不该出现」只能靠人眼
// 数。抽出来之后，菜单的**内容**（哪一项在、哪一项禁用）是数据，渲染只是把数据画
// 出来 —— 测试直接断言数据，不必挂 React、Redux 与 i18n Provider。
//
// 【为什么分隔线由模型收敛】「隐藏不可用项」落地后最常见的回归是空段与连续两条
// 分隔线（旧代码里有 6 处 Divider 是无条件写的）。把收敛做在模型里，渲染端
// 不需要知道任何规则，也不可能忘掉。
import { isClipSplittableAtSec } from "./splitTargets";
import type { ClipInfo } from "../../../features/session/sessionTypes";
import type { MessageKey } from "../../../i18n/messages";
import {
    CHANNEL_MODE_OPTIONS,
    channelModeShortLabel,
    normalizeChannelMode,
} from "../../../utils/channelMode";
import { sharedFadeShape, sortAndFilterFadedClips } from "./clipFadeContext";

// ── 条目类型 ────────────────────────────────────────────────────────────────

/** 一条可点击的菜单项。key 是**动作标识**，渲染端据此取回调。 */
export interface ClipMenuItem {
    kind: "item";
    key: string;
    label: string;
    shortcut?: string;
    disabled?: boolean;
    danger?: boolean;
}

/** 一层子菜单。children 已经过分隔线收敛，渲染端不必再管。 */
export interface ClipMenuSubmenu {
    kind: "submenu";
    key: string;
    label: string;
    badge?: string;
    children: ClipMenuEntry[];
}

/** 不可点击的说明行（多选计数、淡变分组标题、插件里的只读 take 名）。 */
export interface ClipMenuHeading {
    kind: "heading";
    key: string;
    text: string;
}

export interface ClipMenuSeparator {
    kind: "separator";
    key: string;
}

/**
 * Take 行：行点击切换 active take，行尾还有两个**独立**的小按钮。
 *
 * 【为什么不是普通 item】行尾按钮与行点击是两件事（切换 take / 翻转该 take 的
 * 方向 / 循环声道模式），而且按钮不关闭菜单。把它压成 item 会丢掉这两个控件。
 */
export interface ClipMenuTakeRow {
    kind: "takeRow";
    key: string;
    takeId: string;
    label: string;
    /** 行点击（切换 active take）可用。 */
    switchable: boolean;
    reversed: boolean;
    /** 行尾「倒放」按钮可用。 */
    reverseToggle: boolean;
    /**
     * 行尾声道按钮的当前模式；null = **不渲染**按钮（该 take 没有音频源）。
     */
    channelMode: number | null;
}

/** 淡变形状行（七个预设图标）。 */
export interface ClipMenuFadeRow {
    kind: "fadeRow";
    key: string;
    direction: "in" | "out";
    /** null = 多选：选择即批量应用到全部所选片段。 */
    clipId: string | null;
    /** 当前形状；null = 各片段不一致，不预选任何一项。 */
    current: number | null;
}

export type ClipMenuEntry =
    | ClipMenuItem
    | ClipMenuSubmenu
    | ClipMenuHeading
    | ClipMenuSeparator
    | ClipMenuTakeRow
    | ClipMenuFadeRow;

// ── 输入 ────────────────────────────────────────────────────────────────────

/**
 * 能力位：**全部**来自 services/hostCapabilities（它又只读后端的
 * hostCapabilities 快照）。这里不出现任何 isPluginMode() 之类的形态判断 ——
 * 形态判断一旦进入菜单，就会与后端能力位分叉。
 */
export interface ClipMenuCaps {
    /** 宿主片段几何可写（静音 / 循环 / 声道 / 速率 / 编组 / 编辑播放速率）。 */
    editClips: boolean;
    /** 宿主 item 剪贴板（删除 / 复制 / 剪切）。 */
    clipboard: boolean;
    /** 宿主分割。 */
    split: boolean;
    /** 宿主 take 结构可写（切换 / 增删 take）。 */
    takes: boolean;
    /** 胶合（需要新建宿主 item）。 */
    glue: boolean;
    /** 静音检测。 */
    silenceDetection: boolean;
    /** 假立体声扫描。 */
    fakeStereoScan: boolean;
    /** 倒放写回。 */
    reverse: boolean;
    /** 能把处理结果导出到文件（快速导出）。 */
    quickExport: boolean;
    /** REAPER take 音高互转。 */
    pitchBridge: boolean;
    /** 能把 MIDI 建成片段（决定「替换 MIDI」是否出现）。 */
    midiAsClip: boolean;
    /** 淡变形状轴可写。 */
    fadeShape: boolean;
}

/** 哪些可选回调真的传进来了（父组件按场景给，菜单不能凭空造入口）。 */
export interface ClipMenuAvailability {
    replaceMidi?: boolean;
    group?: boolean;
    ungroup?: boolean;
    convertToPitchRef?: boolean;
    updatePitchRef?: boolean;
    exportMidi?: boolean;
    addToParamSelection?: boolean;
    silenceDetection?: boolean;
    scanFakeStereo?: boolean;
    toggleLoop?: boolean;
    channelMode?: boolean;
    editRate?: boolean;
    fadeShape?: boolean;
}

/** 菜单项右侧的快捷键提示（来自快捷键注册表，用户自定义后同步跟随）。 */
export interface ClipMenuShortcuts {
    delete?: string;
    copy?: string;
    cut?: string;
    split?: string;
    group?: string;
    ungroup?: string;
    normalize?: string;
    cycleTakeNext?: string;
    cycleTakePrev?: string;
    addToParamSelection?: string;
}

export interface ClipMenuInput {
    clip: ClipInfo;
    selectedClips: ClipInfo[];
    overlappingClips?: ClipInfo[];
    /** 播放头是否落在片段内（分割的**情境性**前提）。 */
    playheadInClip: boolean;
    canSplitSelected: boolean;
    /**
     * 右键 / 长按的位置（工程秒）。`null` / 缺省 = 不知道落点。
     *
     * 【为什么菜单需要它】「在播放头处分割」在触摸屏上不可用 —— 手机上
     * 没有光标，用户无法表达「切这里」。长按位置**就是**用户指着的地方，
     * 它必须能作为落点传给后端（`splitClipIdsAtPlayhead(ids, splitSecOverride)`
     * 早已支持显式落点，只是此前没有任何调用点传它）。
     */
    pressSec?: number | null;
    /** 插件形态。**只**用于「take 子菜单的结构项收起、行点击禁用」这一件事。 */
    pluginMode: boolean;
    caps: ClipMenuCaps;
    available: ClipMenuAvailability;
    shortcuts?: ClipMenuShortcuts;
    t: (key: MessageKey) => string;
}

// ── 分隔线收敛 ──────────────────────────────────────────────────────────────

/**
 * 去掉「段首 / 段尾 / 连续」的分隔线，并丢掉**空子菜单**。
 *
 * 【为什么要递归】子菜单内部的条目同样可能被整段隐藏（插件里的 Take 子菜单只剩
 * 一行），只在顶层收敛会把问题推到下一层。
 */
export function collapseClipMenuEntries(entries: readonly ClipMenuEntry[]): ClipMenuEntry[] {
    const out: ClipMenuEntry[] = [];
    for (const entry of entries) {
        if (entry.kind === "separator") {
            // 段首与连续两条都不留：分隔线只用来**分开**两段内容。
            if (out.length === 0 || out[out.length - 1].kind === "separator") continue;
            out.push(entry);
            continue;
        }
        if (entry.kind === "submenu") {
            const children = collapseClipMenuEntries(entry.children);
            // 空子菜单是「入口存在但点开什么都没有」，比不渲染更糟。
            if (children.length === 0) continue;
            out.push({ ...entry, children });
            continue;
        }
        out.push(entry);
    }
    while (out.length > 0 && out[out.length - 1].kind === "separator") out.pop();
    return out;
}

// ── 纯判定 ──────────────────────────────────────────────────────────────────

export const isPitchClip = (clip: ClipInfo): boolean => clip.midiNoteCount != null;

/** 某个片段是否至少有一个 take 带音频源（静音检测的前提）。 */
export const hasAudioTake = (clip: ClipInfo): boolean =>
    (clip.takes ?? []).some((take) => !!take.sourcePath);

/** 淡变的**有效**长度：自动淡变优先（与画布、与形状行的判定同口径）。 */
export function effectiveFadeSeconds(clip: ClipInfo): { in: number; out: number } {
    return {
        in: (clip.autoFadeInSec ?? 0) > 0 ? (clip.autoFadeInSec ?? 0) : clip.fadeInSec,
        out: (clip.autoFadeOutSec ?? 0) > 0 ? (clip.autoFadeOutSec ?? 0) : clip.fadeOutSec,
    };
}

// ── 构建 ────────────────────────────────────────────────────────────────────

/**
 * 构建整张片段右键菜单。
 *
 * 三条规则（方案 §3.1.1 的 D1.1）：
 *
 * 1. **结构性不可用**（这条运行形态里永远做不到）⇒ **不出现**。原因对用户不可操作，
 *    占位只会把菜单撑成两倍高、把能用的项淹掉。
 * 2. **情境性不可用**（用户动一下就能满足）⇒ 出现且 disabled，**不写第二行**。
 *    位置关系（播放头在不在片段里）肉眼可见，小字只是噪声。
 * 3. 每一条判据都来自 caps，不来自 pluginMode。
 */
export function buildClipMenuModel(input: ClipMenuInput): ClipMenuEntry[] {
    const {
        clip,
        selectedClips,
        overlappingClips = [],
        playheadInClip,
        canSplitSelected,
        pressSec = null,
        pluginMode,
        caps,
        available,
        shortcuts = {},
        t,
    } = input;

    const isMulti = selectedClips.length >= 2;
    const takes = Array.isArray(clip.takes) ? clip.takes : [];
    const activeTake = takes.find((take) => take.id === clip.activeTakeId) ?? takes[0];

    const allPitch = selectedClips.length > 0 && selectedClips.every(isPitchClip);
    const anyPitch = selectedClips.some(isPitchClip);
    const hasMixedTypes = anyPitch && !allPitch;

    const allMuted = isMulti ? selectedClips.every((c) => c.muted) : clip.muted;
    const allReversed = isMulti ? selectedClips.every((c) => c.reversed) : clip.reversed;
    const allLooped = isMulti ? selectedClips.every((c) => c.loopEnabled) : clip.loopEnabled;
    const commonChannelMode = (() => {
        const modes = new Set(selectedClips.map((c) => normalizeChannelMode(c.channelMode)));
        return modes.size === 1 ? normalizeChannelMode(clip.channelMode) : null;
    })();
    const hasGroup = selectedClips.some((c) => c.groupId != null);

    /** 胶合：仅同轨多选、且不混合音高参考块与常规音频块。 */
    const glueDisabled =
        !isMulti ||
        hasMixedTypes ||
        !selectedClips[0]?.trackId ||
        selectedClips.some((c) => c.trackId !== selectedClips[0].trackId);

    const silenceEligible = isMulti ? selectedClips.some(hasAudioTake) : hasAudioTake(clip);

    const entries: ClipMenuEntry[] = [];
    /** 追加一段：段内至少一条条目才会留下分隔线（收敛在末尾统一做）。 */
    const section = (build: () => ClipMenuEntry[]) => {
        const items = build();
        if (items.length === 0) return;
        entries.push({ kind: "separator", key: "sep-" + entries.length });
        entries.push(...items);
    };
    const item = (
        key: string,
        label: string,
        extra: Partial<Pick<ClipMenuItem, "shortcut" | "disabled" | "danger">> = {},
    ): ClipMenuItem => ({ kind: "item", key, label, ...extra });

    if (isMulti) {
        entries.push({
            kind: "heading",
            key: "selection-count",
            text: t("ctx_selected_n").replace("{n}", String(selectedClips.length)),
        });
    }

    // ── 顶段：删除 / 静音 ──────────────────────────────────────────────────
    if (caps.clipboard) {
        entries.push(
            item(
                isMulti ? "delete-all" : "delete",
                isMulti ? t("ctx_delete_all") : t("ctx_delete"),
                { shortcut: shortcuts.delete, danger: true },
            ),
        );
    }
    if (caps.editClips) {
        const mutedLabel = allMuted
            ? isMulti
                ? t("ctx_unmute_all")
                : t("clip_unmute")
            : isMulti
              ? t("ctx_mute_all")
              : t("clip_mute");
        entries.push(item("mute", mutedLabel));
    }

    // ── Take ───────────────────────────────────────────────────────────────
    const takeChildren: ClipMenuEntry[] = [];
    if (pluginMode) {
        // 【插件里 Take 的边界：结构做不到，属性做得到】切换 / 增删 / 打包 take 会
        // 重建 ARA region ↔ take 的一一对应，是 `canEditHostTakes` 的结构边界。
        // 但**逐 take 的方向与声道模式**落在 take 属性上（chunk 改写 / take 指针的
        // `I_CHANMODE`），结构上做得到 —— 行照常渲染，行尾按钮是可操作项；行点击
        // （切换 active take）恒禁用。
        //
        // 【方向写不回去时退回只读列表】逐 take 倒放与 clip 级倒放共用同一个能力位
        // （chunk 改写 / 宿主动作是同一族）；能力位为 false 的宿主上按钮点不动，
        // 退回标题行 —— "宁可不少显示一个入口，也不露一个注定失败的"。
        if (caps.reverse) {
            for (const take of takes) {
                takeChildren.push({
                    kind: "takeRow",
                    key: "take-" + take.id,
                    takeId: take.id,
                    label:
                        (take.id === clip.activeTakeId ? t("clip_take_active_mark") : "\u2003") +
                        " " +
                        (take.name || take.id),
                    switchable: false,
                    reversed: Boolean(take.reversed),
                    reverseToggle: caps.reverse,
                    channelMode: take.sourcePath ? normalizeChannelMode(take.channelMode) : null,
                });
            }
        } else {
            for (const take of takes) {
                takeChildren.push({
                    kind: "heading",
                    key: "take-" + take.id,
                    text:
                        (take.id === clip.activeTakeId ? t("clip_take_active_mark") : "\u2003") +
                        " " +
                        (take.name || take.id),
                });
            }
        }
        takeChildren.push(item("take-rename", t("clip_take_rename"), { disabled: !activeTake }));
    } else {
        if (isMulti) {
            takeChildren.push(
                item("take-pack", t("clip_pack_into_takes"), { disabled: !caps.takes }),
            );
        }
        if (!isMulti) {
            for (const take of takes) {
                takeChildren.push({
                    kind: "takeRow",
                    key: "take-" + take.id,
                    takeId: take.id,
                    label:
                        (take.id === clip.activeTakeId ? t("clip_take_active_mark") : "\u2003") +
                        " " +
                        (take.name || take.id),
                    switchable: takes.length > 1 && caps.takes,
                    reversed: Boolean(take.reversed),
                    reverseToggle: caps.reverse,
                    channelMode: take.sourcePath ? normalizeChannelMode(take.channelMode) : null,
                });
            }
            if (takes.length > 1) {
                takeChildren.push(
                    { kind: "separator", key: "sep-take-cycle" },
                    item("take-cycle-prev", t("clip_take_cycle_prev"), {
                        shortcut: shortcuts.cycleTakePrev,
                        disabled: !caps.takes,
                    }),
                    item("take-cycle-next", t("clip_take_cycle_next"), {
                        shortcut: shortcuts.cycleTakeNext,
                        disabled: !caps.takes,
                    }),
                );
            }
            takeChildren.push(
                { kind: "separator", key: "sep-take-edit" },
                item("take-add", t("clip_take_add"), { disabled: !caps.takes }),
                item("take-duplicate", t("clip_take_duplicate"), {
                    disabled: !activeTake || !caps.takes,
                }),
                item("take-rename", t("clip_take_rename"), { disabled: !activeTake }),
                item("take-remove", t("clip_take_remove"), {
                    danger: true,
                    disabled: takes.length <= 1 || !activeTake || !caps.takes,
                }),
            );
            if (takes.length > 1) {
                takeChildren.push(
                    item("take-explode", t("clip_take_explode"), { disabled: !caps.takes }),
                );
            }
            // 替换素材：作用对象是**活跃 take** 的源媒体，因此归入 Take 范畴。
            if (!allPitch) {
                takeChildren.push(
                    item(
                        isMulti ? "replace-all" : "replace",
                        isMulti ? t("ctx_replace_all") : t("ctx_replace"),
                        { disabled: !caps.takes },
                    ),
                );
            }
            if (anyPitch && available.replaceMidi && caps.midiAsClip) {
                takeChildren.push(
                    item(
                        isMulti ? "replace-midi-all" : "replace-midi",
                        isMulti ? t("ctx_replace_midi_all") : t("ctx_replace_midi"),
                    ),
                );
            }
        }
    }
    entries.push({
        kind: "submenu",
        key: "takes",
        label: t("clip_takes"),
        badge: takes.length > 1 ? String(takes.length) : undefined,
        children: takeChildren,
    });

    // ── 声道模式 / 倒放 / 循环 ─────────────────────────────────────────────
    if (available.channelMode && caps.editClips) {
        const channelChildren: ClipMenuEntry[] = CHANNEL_MODE_OPTIONS.map((option) =>
            item(
                "channel-mode-" + String(option.value),
                (commonChannelMode === option.value ? "●" : "\u2003") + " " + t(option.i18nKey),
                { shortcut: option.shortLabel },
            ),
        );
        // 插件里没有「扫描假立体声」：它要读整份媒体的 PCM 才能判定（见
        // canScanFakeStereo），而声道模式本身可写 —— 两者不能共用一道门。
        if (available.scanFakeStereo && caps.fakeStereoScan) {
            channelChildren.push(
                { kind: "separator", key: "sep-scan" },
                item("scan-fake-stereo", t("ctx_scan_fake_stereo")),
            );
        }
        entries.push({
            kind: "submenu",
            key: "channel-mode",
            label: t("ctx_channel_mode"),
            badge:
                commonChannelMode === null ? undefined : channelModeShortLabel(commonChannelMode),
            children: channelChildren,
        });
    }

    if (caps.reverse) {
        entries.push(
            item(
                "reverse",
                allReversed
                    ? isMulti
                        ? t("ctx_unreverse_selected")
                        : t("ctx_unreverse")
                    : isMulti
                      ? t("ctx_reverse_selected")
                      : t("ctx_reverse"),
            ),
        );
    }

    if (available.toggleLoop && caps.editClips) {
        entries.push(
            item(
                "loop",
                allLooped
                    ? isMulti
                        ? t("ctx_unloop_selected")
                        : t("ctx_unloop")
                    : isMulti
                      ? t("ctx_loop_selected")
                      : t("ctx_loop"),
            ),
        );
    }

    // ── 剪贴板 / 分割 / 规格化 ─────────────────────────────────────────────
    section(() => {
        const out: ClipMenuEntry[] = [];
        if (caps.clipboard) {
            out.push(
                item(isMulti ? "copy-all" : "copy", isMulti ? t("ctx_copy_all") : t("ctx_copy"), {
                    shortcut: shortcuts.copy,
                }),
                item(isMulti ? "cut-all" : "cut", isMulti ? t("ctx_cut_all") : t("ctx_cut"), {
                    shortcut: shortcuts.cut,
                }),
            );
        }
        if (caps.split) {
            // 「在此处分割」：落点 = 右键 / 长按位置。
            //
            // 【为什么它是第一条】触摸屏上没有光标，用户唯一能表达「切这里」的
            // 方式就是**指着那里**。把这条放在「在播放头处分割」之上，是为了让
            // 手机上第一眼看到的就是可用的那一条（播放头在手机上通常离手指很远）。
            const pressInClip =
                pressSec != null && Number.isFinite(pressSec)
                    ? isMulti
                        ? selectedClips.some((c) => isClipSplittableAtSec(c, pressSec))
                        : isClipSplittableAtSec(clip, pressSec)
                    : false;
            if (pressSec != null && Number.isFinite(pressSec)) {
                out.push(item("split-here", t("ctx_split_here"), { disabled: !pressInClip }));
            }
            // 【为什么这里只禁用、不隐藏】播放头不在片段内是**用户一步就能改**的状态；
            // 结构性的「宿主没有分割通道」已经由 caps.split 挡在上面。
            out.push(
                item("split", t("ctx_split_at_playhead"), {
                    shortcut: shortcuts.split,
                    disabled: isMulti ? !canSplitSelected : !playheadInClip,
                }),
            );
        }
        out.push(
            item(
                isMulti ? "normalize-all" : "normalize",
                isMulti ? t("ctx_normalize_all") : t("ctx_normalize"),
                { shortcut: shortcuts.normalize },
            ),
        );
        if (available.addToParamSelection) {
            out.push(
                item("add-to-param-selection", t("ctx_add_to_param_selection"), {
                    shortcut: shortcuts.addToParamSelection,
                }),
            );
        }
        return out;
    });

    // ── 内容工具：速率 / 静音检测 ──────────────────────────────────────────
    section(() => {
        const out: ClipMenuEntry[] = [];
        if (available.editRate && caps.editClips) out.push(item("edit-rate", t("ctx_edit_rate")));
        if (available.silenceDetection && caps.silenceDetection) {
            out.push(
                item("silence-detection", t("ctx_silence_detection"), {
                    // 情境性：所选片段都没有音频源 ⇒ 出现但禁用（单行，无说明）。
                    disabled: !silenceEligible,
                }),
            );
        }
        return out;
    });

    // ── 音高参考 ───────────────────────────────────────────────────────────
    section(() => {
        if (!allPitch && available.convertToPitchRef) {
            return [item("pitch-reference-convert", t("ctx_convert_to_pitch_ref"))];
        }
        if (allPitch && available.updatePitchRef) {
            return [item("pitch-reference-update", t("ctx_update_pitch_ref"))];
        }
        return [];
    });

    // ── 导出 ───────────────────────────────────────────────────────────────
    section(() => {
        const out: ClipMenuEntry[] = [];
        if (!allPitch && caps.quickExport) out.push(item("quick-export", t("ctx_quick_export")));
        if (available.exportMidi) out.push(item("export-midi", t("ctx_export_midi")));
        return out;
    });

    // ── 编组 ───────────────────────────────────────────────────────────────
    if (isMulti || hasGroup) {
        section(() => [
            {
                kind: "submenu",
                key: "group",
                label: t("ctx_group"),
                children: [
                    ...(isMulti && !hasGroup && caps.editClips
                        ? [item("group", t("common_group"), { shortcut: shortcuts.group })]
                        : []),
                    ...(hasGroup && caps.editClips
                        ? [item("ungroup", t("common_ungroup"), { shortcut: shortcuts.ungroup })]
                        : []),
                    ...(isMulti && caps.glue
                        ? [item("glue", t("common_glue"), { disabled: glueDisabled })]
                        : []),
                ],
            },
        ]);
    }

    // ── REAPER take 音高互转 ───────────────────────────────────────────────
    if (caps.pitchBridge) {
        section(() => [
            {
                kind: "submenu",
                key: "host-pitch",
                label: t("ctx_host_pitch"),
                children: [
                    item("host-pitch-import", t("ctx_host_pitch_import")),
                    item("host-pitch-export", t("ctx_host_pitch_export")),
                ],
            },
        ]);
    }

    // ── 淡变形状 ───────────────────────────────────────────────────────────
    if (available.fadeShape && caps.fadeShape) {
        section(() => {
            if (isMulti) {
                const fadeInTargets = selectedClips.filter((c) => effectiveFadeSeconds(c).in > 0);
                const fadeOutTargets = selectedClips.filter((c) => effectiveFadeSeconds(c).out > 0);
                const out: ClipMenuEntry[] = [];
                if (fadeInTargets.length > 0) {
                    out.push({
                        kind: "fadeRow",
                        key: "fade-in",
                        direction: "in",
                        clipId: null,
                        current: sharedFadeShape(fadeInTargets, "in"),
                    });
                }
                if (fadeOutTargets.length > 0) {
                    out.push({
                        kind: "fadeRow",
                        key: "fade-out",
                        direction: "out",
                        clipId: null,
                        current: sharedFadeShape(fadeOutTargets, "out"),
                    });
                }
                return out;
            }
            const faded = sortAndFilterFadedClips({ clip, overlappingClips });
            if (faded.length === 0) return [];
            const out: ClipMenuEntry[] = [];
            if (faded.length > 1) {
                out.push({
                    kind: "heading",
                    key: "fade-header",
                    text: t("overlapping_clips_header").replace("{n}", String(faded.length)),
                });
            }
            for (const fadedClip of faded) {
                if (faded.length > 1) {
                    out.push({
                        kind: "heading",
                        key: "fade-name-" + fadedClip.id,
                        text: fadedClip.name || fadedClip.id,
                    });
                }
                const effective = effectiveFadeSeconds(fadedClip);
                if (effective.in > 0) {
                    out.push({
                        kind: "fadeRow",
                        key: "fade-in-" + fadedClip.id,
                        direction: "in",
                        clipId: fadedClip.id,
                        current: Number.isFinite(fadedClip.fadeInShape) ? fadedClip.fadeInShape : 0,
                    });
                }
                if (effective.out > 0) {
                    out.push({
                        kind: "fadeRow",
                        key: "fade-out-" + fadedClip.id,
                        direction: "out",
                        clipId: fadedClip.id,
                        current: Number.isFinite(fadedClip.fadeOutShape)
                            ? fadedClip.fadeOutShape
                            : 0,
                    });
                }
            }
            return out;
        });
    }

    return collapseClipMenuEntries(entries);
}
