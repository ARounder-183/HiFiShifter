// 时间轴内核淡变 hover / 预览 / 提交与右键菜单。拆自 TimelinePanel。
import React from "react";
import { isPluginMode, canEditFadeCurvature } from "../../../../services/hostCapabilities";
import { hostMediaState, hostMediaReason } from "../../../../features/ara/hostAudio";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import {
    resolveCurvatureEditBase,
    resolveCurvePointer,
    solveNearestCurveAxes,
    solveNearestCurveDir,
} from "../reaperFade";
import { hostFadeGainForAxes } from "../hostFadeDisplay";
import {
    buildCrossfadeGripInfoContent,
    buildSingleFadeInfoContent,
    publishFadeRichTooltip,
    type FadeInfoDelta,
    type FadeLabelLookup,
} from "../fadeTooltipText";
import type { HostFadeMetadata } from "../../../../types/api";
import {
    buildSnapOffsetInfoText,
    type TimeValueFormatContext,
    type TimeValueLabelLookup,
} from "../timeValueText";
import { effectiveFadeSec } from "../kernel/interaction/fadeTargets";
import type { ClipHitRegion } from "../kernel/interaction/hitTest";
import type { ClipHeaderControl } from "../kernel/interaction/clipHeaderControls";
import {
    setClipsStateBulkRemote,
    setClipFades,
    checkpointHistory,
    setClipAutoFades,
} from "../../../../features/session/sessionSlice";
import { batch } from "react-redux";
import { formatGainDbValue } from "../math";
import { resolveKernelEditParticipants, type KernelEditParticipant } from "../hooks/kernelEditSet";
import { computeInitialCrossfadeSides } from "../hooks/autoCrossfade";
import { applyBulkFadeValue } from "../hooks/bulkClipEdit";
import { clampGainDb, gainToDbSafe } from "../../../../utils/gainDb";
import {
    clipHasNonlinearStretch,
    clipHasTakeEnvelope,
    clipTooltipTitle,
} from "../../../../features/session/sessionTypes";
import type { useTimelineState } from "../hooks/useTimelineState";
import type { useI18n } from "../../../../i18n/I18nProvider";
import type { AppDispatch } from "../../../../app/store";

type Deps = {
    state: ReturnType<typeof useTimelineState>;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
    timeValueFormatCtx: TimeValueFormatContext;
    t: ReturnType<typeof useI18n>["t"];
    tVars: ReturnType<typeof useI18n>["tVars"];
    beginKernelGestureInteraction: () => void;
    endKernelGestureInteraction: () => void;
};

type FadeSideInfo = {
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
    delta?: FadeInfoDelta;
    hostFades?: HostFadeMetadata;
};

export function useTimelineKernelFade({
    state,
    dispatch,
    multiSelectedClipIds,
    timeValueFormatCtx,
    t,
    tVars,
    beginKernelGestureInteraction,
    endKernelGestureInteraction,
}: Deps) {
    const { sessionRef, fadeCurvatureKb, pxPerSecRef } = state;
    const kernelFadeOriginRef = React.useRef<{
        clipId: string;
        fadeInSec: number;
        fadeOutSec: number;
        /**
         * 本次淡变作用于的全部 clip（多选集合）。
         *
         * 旧实现 `useEditDrag` 的 `supportsGroupExpansion` **排除 fade**——
         * 淡变拖拽不展开编组，只作用于选中的 clip。
         */
        participants: KernelEditParticipant[];
        /** 各参与者的按下时基准（回滚与批量换算用）。 */
        baseById: Map<
            string,
            {
                fadeInSec: number;
                fadeOutSec: number;
                lengthSec: number;
                /** 曲率与形状：曲率拖拽会改它们，取消必须能还原。 */
                fadeInDir: number;
                fadeOutDir: number;
                fadeInShape: number;
                fadeOutShape: number;
            }
        >;
        /**
         * 各参与者按下时的**自动交叉淡化**长度（取消时一并还原）。
         *
         * 预览期会把拖拽侧的自动值清零（手动 fade 必须赢过自动值，见预览处的说明），
         * 取消就必须把它放回去——否则用户按 Esc 之后，那条自动交叉淡化会永久消失。
         */
        autoBaseById: Map<string, { autoFadeInSec: number; autoFadeOutSec: number }>;
        /** 自动交叉淡化：受影响集合 / 编辑前重叠关系 / 可调整侧（与裁切同源）。 */
        xfadeClipIds: string[];
        initialCrossfadeSides: ReturnType<typeof computeInitialCrossfadeSides>;
        editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
    } | null>(null);

    /**
     * 内核 clip 悬停 → 发布浮标内容（旧实现各 `data-tooltip` 的等价物）。
     *
     * 与淡变浮标共用**同一个锚点**（内核容器）：旧实现把这些文案挂在各自的 DOM
     * 元素上，内核自绘后那些元素不存在，因此必须由命中结果反推"指针下是什么控件"。
     * 两条通道互斥（宿主保证）：淡变控件命中时走富内容通道，其余走这里。
     *
     * 文案与取法逐条对齐旧实现：
     * - 名称区 → MIDI 前辍 + 显示名，或源文件路径（`clip.sourcePath`）；
     * - 静音 → 按当前状态给「静音 / 取消静音」；
     * - 锁链 → 按该组当前是否被禁用给「启用 / 禁用编组」；
     * - 速率 / 增益徽标 → 各自的静态提示（数值由双击后的输入框给出）；
     * - 共振峰 → 面板标题；
     * - 增益旋钮 → 带**实时数值**的提示（拖动中带增量，见增益预览）；
     * - SnapOffset 手柄 → 拖动调整提示。
     *
     * @param args 命中信息；不在任何 clip 上时为 null。
     * @returns 无返回值。
     */
    const handleKernelClipHover = React.useCallback(
        (
            args: {
                clipId: string;
                region: ClipHitRegion;
                headerControl: ClipHeaderControl | null;
            } | null,
        ) => {
            const anchor =
                typeof document === "undefined"
                    ? null
                    : (document.querySelector(
                          "[data-hs-fade-tooltip-anchor]",
                      ) as HTMLElement | null);
            if (anchor === null) return;
            if (args === null) {
                publishFadeRichTooltip(anchor, null);
                return;
            }
            const session = sessionRef.current;
            const clip = session.clips.find((item) => item.id === args.clipId);
            if (clip === undefined) {
                publishFadeRichTooltip(anchor, null);
                return;
            }
            let text: string | null = null;
            switch (args.headerControl) {
                case "name": {
                    // 名称行只给名字，不给路径（见 `clipTooltipTitle`）。
                    const label = clipTooltipTitle(clip);
                    const base =
                        clip.midiNoteCount != null
                            ? `${t("clip_type_midi_prefix")} ${label}`.trim()
                            : label;
                    // 插件不渲染非线性拉伸 / take 包络。事先在浮标里说清楚，而不是让用户
                    // 点了分割/拖边再撞一个"operation refused"（独立 App 会物化它们，
                    // 因此只在插件模式下提示）。
                    let described = base;
                    if (isPluginMode() && clipHasNonlinearStretch(clip)) {
                        described = `${described} · ${t("clip_nonlinear_stretch_unsupported")}`;
                    }
                    if (isPluginMode() && clipHasTakeEnvelope(clip)) {
                        described = `${described} · ${t("clip_take_envelope_unsupported")}`;
                    }
                    // 倒放标记：宿主报告方向后必须让人看得见 —— 否则"这个 take 倒放了"
                    // 只活在数据里，用户会以为它没倒放（插件里尤其如此：没有波形可看）。
                    text = clip.reversed ? `${described} · ${t("clip_take_reverse")}` : described;
                    break;
                }
                case "mute":
                    text = clip.muted ? t("clip_unmute") : t("clip_mute");
                    break;
                case "chain": {
                    const groupId = clip.groupId;
                    if (groupId != null && groupId !== "") {
                        const disabled = session.disabledGroupIds.includes(groupId);
                        text = disabled ? t("enable_group") : t("disable_group");
                    }
                    break;
                }
                case "rate-label":
                    text = t("clip_badge_rate_tip");
                    break;
                case "gain-label":
                    text = t("clip_badge_gain_tip");
                    break;
                case "formant":
                    text = t("clip_formant_title");
                    break;
                case "gain-knob":
                    text = t("gain_value_tooltip").replace(
                        "{gain}",
                        formatGainDbValue(clampGainDb(gainToDbSafe(clip.gain))),
                    );
                    break;
                default:
                    // SnapOffset 手柄没有 header 控件，用分区识别。
                    //
                    // 文案的三种形态（偏移为 0 只给标签 / 两行 / 拖拽带增量）由
                    // `buildSnapOffsetInfoText` 一处判定，悬停这里只是"无位移"那一支。
                    // 偏移是**时长**口径、位置是**时刻**口径（感知 Tempo Map），
                    // 两者不可互换，见 `timeValueText` 文件头。
                    if (args.region === "snap-offset-handle") {
                        const offset = Math.max(0, Number(clip.snapOffsetSec) || 0);
                        text = buildSnapOffsetInfoText({
                            offsetSec: offset,
                            positionSec: (Number(clip.startSec) || 0) + offset,
                            deltaSec: null,
                            formatCtx: timeValueFormatCtx,
                            t: tVars as TimeValueLabelLookup,
                        });
                    }
                    break;
            }
            /*
             * 无源占位的统一提示：悬停 clip 本体时说明"为什么里面没有内容"。
             *
             * 只在没有更具体的控件文案时才用（`text === null` 即命中 body / 无控件
             * 区域）—— 悬停名称、静音等控件时，那些文案更贴近用户当刻的意图。
             *
             * 【为什么按状态分文案】一个布尔承载不了四种成因：刚分割后的**在途**、
             * folder 父轨的**永远拿不到**、倒放的**宿主在处理**、以及真的在等。
             * 此前它们共用一句"等待 REAPER 提供音频（未分配 ARA 区域）"，
             * 于是"我刚分割了一下"看起来像"插件坏了"。
             */
            if (text === null) {
                switch (hostMediaState(clip)) {
                    case "pending":
                        text = t("ara_clip_syncing_host_audio");
                        break;
                    case "unavailable":
                        // 【为什么按原因分开】"等待 REAPER 提供音频"回答不了用户最需要
                        // 的问题：该等、该改用法、还是该撤销。`take_switched`（宿主换了
                        // 当前 Take 而 ARA 未重认领，典型来源"倒放 Item 为新 Take"）是
                        // **可操作**的，必须给出下一步；`source_read_failed`（已授权却读不出
                        // 样本）同样需要明确告知，不能与"还在等分配 region"混为一谈；其余退回
                        // 通用文案。
                        text = (() => {
                            const reason = hostMediaReason(clip);
                            if (reason === "take_switched") {
                                return t("ara_clip_host_take_switched");
                            }
                            if (reason === "source_read_failed") {
                                return t("ara_clip_host_source_read_failed");
                            }
                            if (reason === "source_revoked") {
                                return t("ara_clip_host_source_revoked");
                            }
                            return t("ara_clip_waiting_for_host_audio");
                        })();
                        break;
                    default:
                        break;
                }
            }
            publishFadeRichTooltip(anchor, text);
        },
        [sessionRef, t, tVars, timeValueFormatCtx],
    );

    /** 淡变富 Tooltip 的锚点（内核容器；AppTooltip 自己跟随指针）。 */
    const fadeTooltipAnchor = React.useCallback(
        (): HTMLElement | null =>
            typeof document === "undefined"
                ? null
                : (document.querySelector("[data-hs-fade-tooltip-anchor]") as HTMLElement | null),
        [],
    );

    /**
     * 淡变 Tooltip 的**唯一发布点**（悬停与两种拖拽共用）。
     *
     * 悬停只给当前值；拖拽传入 `delta`（当前值 − 按下时的值）后，长度与曲率两行
     * 各追加 `[±增量]`，并随指针**逐帧刷新** —— 富内容经事件总线直接更新气泡，
     * 不依赖元素属性变化被观察到，因此拖拽中的读数实时跟着走。
     */
    const publishFadeSideInfo = React.useCallback(
        (anchor: HTMLElement | null, side: FadeSideInfo) => {
            publishFadeRichTooltip(
                anchor,
                buildSingleFadeInfoContent({
                    ...side,
                    formatCtx: timeValueFormatCtx,
                    t: t as unknown as FadeLabelLookup,
                }),
            );
        },
        [t, timeValueFormatCtx],
    );

    /** 交叉点抓手（双列）的发布：两侧各自带自己的增量。 */
    const publishFadeGripInfo = React.useCallback(
        (
            anchor: HTMLElement | null,
            earlier: Omit<FadeSideInfo, "isOut">,
            later: Omit<FadeSideInfo, "isOut">,
        ) => {
            publishFadeRichTooltip(
                anchor,
                buildCrossfadeGripInfoContent({
                    earlier,
                    later,
                    formatCtx: timeValueFormatCtx,
                    t: t as unknown as FadeLabelLookup,
                }),
            );
        },
        [t, timeValueFormatCtx],
    );

    /** 内核淡变角预览：只改对应一侧的淡变长度（另一侧保持不变）。 */
    const handleKernelFadePreview = React.useCallback(
        (args: {
            clipId: string;
            side: "in" | "out";
            fadeSec: number;
            deltaSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
            curveEnv: {
                clientY: number;
                envTopClientY: number;
                bodyHeightPx: number;
                pointerSec: number;
            };
        }) => {
            if (kernelFadeOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const clip = session.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                // 不展开编组（旧实现 supportsGroupExpansion 排除 fade）。
                const participants = resolveKernelEditParticipants({
                    anchorClipId: clip.id,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds: session.tracks.map((track) => track.id),
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    expandGroups: false,
                });
                const baseById = new Map<
                    string,
                    {
                        fadeInSec: number;
                        fadeOutSec: number;
                        lengthSec: number;
                        fadeInDir: number;
                        fadeOutDir: number;
                        fadeInShape: number;
                        fadeOutShape: number;
                    }
                >();
                const autoBaseById = new Map<
                    string,
                    { autoFadeInSec: number; autoFadeOutSec: number }
                >();
                for (const participant of participants) {
                    const item = session.clips.find(
                        (candidate) => candidate.id === participant.clipId,
                    );
                    if (item === undefined) continue;
                    autoBaseById.set(participant.clipId, {
                        autoFadeInSec: Number(item.autoFadeInSec) || 0,
                        autoFadeOutSec: Number(item.autoFadeOutSec) || 0,
                    });
                    baseById.set(participant.clipId, {
                        fadeInSec: Number(item.fadeInSec) || 0,
                        fadeOutSec: Number(item.fadeOutSec) || 0,
                        lengthSec: Math.max(0, Number(item.lengthSec) || 0),
                        fadeInDir: Number(item.fadeInDir) || 0,
                        fadeOutDir: Number(item.fadeOutDir) || 0,
                        fadeInShape: Number(item.fadeInShape) || 0,
                        fadeOutShape: Number(item.fadeOutShape) || 0,
                    });
                }
                // 自动交叉淡化：受影响集合 = 参与者；可调整侧 = 本次拖拽的那一侧
                // （拖淡入只允许自动调整 fadeIn）。
                const xfadeClipIds = participants.map((participant) => participant.clipId);
                const editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }> = {};
                for (const id of xfadeClipIds) {
                    editSides[id] =
                        args.side === "in"
                            ? { fadeIn: true, fadeOut: false }
                            : { fadeIn: false, fadeOut: true };
                }
                kernelFadeOriginRef.current = {
                    clipId: clip.id,
                    fadeInSec: clip.fadeInSec,
                    fadeOutSec: clip.fadeOutSec,
                    participants,
                    baseById,
                    autoBaseById,
                    xfadeClipIds,
                    initialCrossfadeSides: computeInitialCrossfadeSides(
                        session.clips,
                        xfadeClipIds,
                    ),
                    editSides,
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelFadeOriginRef.current;
            if (origin === null) return;

            // ── 曲率拖拽：`modifier.fadeCurvatureDrag`（默认 Alt）按住时改**曲率** ──
            // 与旧实现 `useEditDrag` 的 fade 分支同一套：把指针投影到曲线族上取最近
            // 点，解出该侧的新 `dir`。按帧判定修饰键 → 长度/曲率可以无缝互切。
            //
            // 曲率只作用于**锚点 clip 的该侧**（旧实现明确："曲率只作用于当前 clip
            // 的该侧：指针 Y 必须映射到该 clip 自己的 gain=1 基线"——各行 body 几何
            // 不同，无法跨 clip 共用同一指针 Y）。
            if (isModifierActive(fadeCurvatureKb, args.modifiers)) {
                // 修饰键按下了 = 用户要的是**曲率**。宿主轴版本读不出来时做不了曲率，
                // 此时**不能**退化成改长度 —— 那是另一个编辑，用户会以为自己在调曲率
                // 却把淡变长度改掉了（静默改错数据比什么都不做更坏）。
                if (!canEditFadeCurvature()) return;
                const clip = sessionRef.current.clips.find((item) => item.id === args.clipId);
                if (clip === undefined) return;
                const clipStart = Number(clip.startSec) || 0;
                const clipLen = Math.max(0, Number(clip.lengthSec) || 0);
                const widthSec =
                    args.side === "in"
                        ? effectiveFadeSec(clip.fadeInSec, clip.autoFadeInSec)
                        : effectiveFadeSec(clip.fadeOutSec, clip.autoFadeOutSec);
                const leftSec = args.side === "in" ? clipStart : clipStart + clipLen - widthSec;
                const pt = resolveCurvePointer(
                    args.curveEnv,
                    { leftSec, widthSec },
                    args.curveEnv.pointerSec,
                    args.curveEnv.clientY,
                );
                if (pt === null) return;
                const shape = resolveCurvatureEditBase(
                    (args.side === "in" ? clip.fadeInShape : clip.fadeOutShape) ?? 0,
                ).shape;
                const baseDir = (args.side === "in" ? clip.fadeInDir : clip.fadeOutDir) ?? 0;
                // 【为什么投影要用宿主那套曲线】画布在新轴宿主上画的是 `hostFadeGainForAxes`
                // 的混合曲线，不是 `fadeGainSigned`。求解器若还按后者算，指针与画出来的
                // 包络就是两条不同的曲线 —— 表现为"拖了不跟手"。
                const curveMode = args.side === "in" ? "in" : "out";
                const continuous = clip.hostFades?.curve_mode === "continuous";
                const hostS = continuous
                    ? curveMode === "out"
                        ? clip.hostFades!.out_s
                        : clip.hostFades!.in_s
                    : null;
                const aspectYOverX =
                    args.curveEnv.bodyHeightPx / Math.max(1, widthSec * pxPerSecRef.current);
                // 【continuous 宿主为什么必须二维求解】它的曲线由 `(curvature, S)` 两轴
                // 共同决定，而 `solveNearestCurveDir` 的签名只能解一维 —— 在 S 族曲线上
                // 拖动时它会把 S 钉死成常数、只动 curvature，解出的点不在曲线族上（"拖了
                // 不跟手 / 跳变"）。二维版解出的两个分量**都写**，连续拖动才跟手。
                const baseS = continuous
                    ? Number((curveMode === "out" ? clip.fadeOutS : clip.fadeInS) ?? hostS) || 0
                    : 0;
                const solved = continuous
                    ? solveNearestCurveAxes({
                          mode: curveMode,
                          curvature: Number(baseDir) || 0,
                          s: baseS,
                          pointerX01: pt.t,
                          pointerY01: pt.gain,
                          aspectYOverX,
                          gainAt: (t, c, s) => hostFadeGainForAxes(curveMode, c, s, t),
                      })
                    : null;
                const nextDir = solved
                    ? solved.curvature
                    : solveNearestCurveDir({
                          shape,
                          dir: baseDir,
                          mode: args.side,
                          pointerX01: pt.t,
                          pointerY01: pt.gain,
                          aspectYOverX,
                          ...(hostS === null
                              ? {}
                              : {
                                    gainAt: (t: number, dir: number) =>
                                        hostFadeGainForAxes(curveMode, dir, hostS, t),
                                }),
                      }).dir;
                dispatch(
                    args.side === "in"
                        ? setClipFades({
                              clipId: args.clipId,
                              fadeInDir: nextDir,
                              ...(solved ? { fadeInS: solved.s } : {}),
                          })
                        : setClipFades({
                              clipId: args.clipId,
                              fadeOutDir: nextDir,
                              ...(solved ? { fadeOutS: solved.s } : {}),
                          }),
                );
                // 曲率拖拽的实时浮标：曲率行带 `[±增量]`，长度行只给当前值。
                // 增量基准必须取**按下时**的 dir（`origin.baseById`）—— `baseDir`
                // 上面那个变量是本帧从 Redux 读的"上一帧值"，用它会让增量逐帧归零。
                const curveBase = origin.baseById.get(args.clipId);
                const curveIsOut = args.side === "out";
                publishFadeSideInfo(fadeTooltipAnchor(), {
                    isOut: curveIsOut,
                    shape,
                    dir: nextDir,
                    lengthSec: widthSec,
                    hostFades: clip.hostFades,
                    delta: {
                        dir:
                            nextDir -
                            ((curveIsOut ? curveBase?.fadeOutDir : curveBase?.fadeInDir) ?? 0),
                    },
                });
                return;
            }

            // 多选批量：**同一个淡变长度值**应用到全部参与者（各自按自身长度钳制），
            // 与旧实现 `applyBulkFadeValue` 同源（不是"同一增量"）。
            const target = args.side === "in" ? "fadeInSec" : "fadeOutSec";
            const updates = applyBulkFadeValue({
                clipIds: origin.participants.map((participant) => participant.clipId),
                clipsById: new Map(
                    [...origin.baseById].map(([clipId, base]) => [
                        clipId,
                        { lengthSec: base.lengthSec },
                    ]),
                ),
                target,
                nextValue: args.fadeSec,
            });
            batch(() => {
                for (const update of updates) dispatch(setClipFades(update));
            });
            // 手动拖拽淡变 = 用户手动 fade：**该侧的自动交叉淡化必须清零**。
            //
            // 旧实现 `useEditDrag` 的 fade 分支正是这样做的（拖拽期
            // `setClipAutoFades({autoFade*: 0})` + 节流 remote 写 0），而**不是**
            // 去跑 `previewAutoCrossfade`。原因是绘制端按「自动 > 0 时自动赢」取值：
            // 若让自动值在拖拽中继续按重叠量重算，用户刚拖出来的手动长度会被立即
            // 覆盖回自动值——表现为「拖这条包络线完全不动」。
            batch(() => {
                for (const participant of origin.participants) {
                    dispatch(
                        args.side === "in"
                            ? setClipAutoFades({ clipId: participant.clipId, autoFadeInSec: 0 })
                            : setClipAutoFades({ clipId: participant.clipId, autoFadeOutSec: 0 }),
                    );
                }
            });
            // 长度拖拽的实时浮标：长度行带 `[±位移]`，曲率行只给当前值。
            //
            // 【位移基准取"按下时用户看到的长度"】绘制端按「自动 > 0 时自动赢」取值，
            // 因此按下前显示的可能是自动交叉淡化长度而非手动字段 —— 拿 `fadeInSec`
            // 当基准会让"从自动值拖走"那一下报出与实际不符的巨大位移。
            const lenBase = origin.baseById.get(args.clipId);
            const lenAutoBase = origin.autoBaseById.get(args.clipId);
            const lenIsOut = args.side === "out";
            const nextLength = Math.min(Math.max(args.fadeSec, 0), lenBase?.lengthSec ?? 0);
            const baseLength = effectiveFadeSec(
                lenIsOut ? lenBase?.fadeOutSec : lenBase?.fadeInSec,
                lenIsOut ? lenAutoBase?.autoFadeOutSec : lenAutoBase?.autoFadeInSec,
            );
            publishFadeSideInfo(fadeTooltipAnchor(), {
                isOut: lenIsOut,
                shape: (lenIsOut ? lenBase?.fadeOutShape : lenBase?.fadeInShape) ?? 0,
                dir: (lenIsOut ? lenBase?.fadeOutDir : lenBase?.fadeInDir) ?? 0,
                lengthSec: nextLength,
                // 宿主读数按 clip id 现取：长度拖拽期间它不变，不必进手势快照。
                hostFades: sessionRef.current.clips.find((item) => item.id === args.clipId)
                    ?.hostFades,
                delta: { lengthSec: nextLength - baseLength },
            });
        },
        [
            beginKernelGestureInteraction,
            dispatch,
            fadeCurvatureKb,
            // 曲率纵横比换算读实时 pxPerSec：创建时快照会在缩放后把曲率解算
            // 权重算错；ref 读取既新鲜又不让回调随缩放重建。
            pxPerSecRef,
            multiSelectedClipIds,
            sessionRef,
            fadeTooltipAnchor,
            publishFadeSideInfo,
        ],
    );

    /** 内核淡变角收尾：提交或回滚（取消时两侧一起还原）。 */
    const handleKernelFadeCommit = React.useCallback(
        (args: { clipId: string; side: "in" | "out"; fadeSec: number; cancelled: boolean }) => {
            const origin = kernelFadeOriginRef.current;
            kernelFadeOriginRef.current = null;
            if (origin === null) return;
            if (args.cancelled) {
                // 取消：全部参与者一起还原（多选淡变只还原锚点会让其余 clip 留在半途）。
                // **包括自动交叉淡化**：预览期把它清零了（手动值必须赢），取消就必须
                // 放回去，否则按一次 Esc 会让那条自动交叉淡化永久消失。
                batch(() => {
                    for (const [clipId, base] of origin.baseById) {
                        dispatch(
                            setClipFades({
                                clipId,
                                fadeInSec: base.fadeInSec,
                                fadeOutSec: base.fadeOutSec,
                                fadeInDir: base.fadeInDir,
                                fadeOutDir: base.fadeOutDir,
                                fadeInShape: base.fadeInShape,
                                fadeOutShape: base.fadeOutShape,
                            }),
                        );
                    }
                    for (const [clipId, auto] of origin.autoBaseById) {
                        dispatch(
                            setClipAutoFades({
                                clipId,
                                autoFadeInSec: auto.autoFadeInSec,
                                autoFadeOutSec: auto.autoFadeOutSec,
                            }),
                        );
                    }
                });
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 提交值取 Redux 当前值（预览已写入按各自长度钳制后的结果）。
            //
            // 只提交**手动长度确实被本手势改动**的参与者，并把该侧自动交叉淡化
            // 一起写 0（自动 → 手动的转换）——与旧实现 `useEditDrag` 的 fade 分支
            // 同一规则（它按 `lengthEdited` 过滤后逐 clip 清 auto）。
            const session = sessionRef.current;
            const updates: Array<{
                clipId: string;
                fadeInSec?: number;
                fadeOutSec?: number;
                autoFadeInSec?: number;
                autoFadeOutSec?: number;
                fadeInDir?: number;
                fadeOutDir?: number;
                fadeInS?: number;
                fadeOutS?: number;
            }> = [];
            for (const participant of origin.participants) {
                const clip = session.clips.find((item) => item.id === participant.clipId);
                if (clip === undefined) continue;
                const base = origin.baseById.get(participant.clipId);
                // S 只对 continuous 宿主可写（`validate_fade_axes` 会在旧轴宿主上拒绝
                // 任何 S 字段）——所以只在 continuous 时把它随曲率一起送出。
                const continuous = clip.hostFades?.curve_mode === "continuous";
                if (args.side === "in") {
                    const manual = Number(clip.fadeInSec) || 0;
                    const lengthEdited = Math.abs(manual - (base?.fadeInSec ?? 0)) > 1e-9;
                    const nextDir = Number(clip.fadeInDir) || 0;
                    const dirEdited = Math.abs(nextDir - (base?.fadeInDir ?? 0)) > 1e-9;
                    // 长度没改、曲率也没改 → 本次手势对该 clip 无写入。
                    if (!lengthEdited && !dirEdited) continue;
                    updates.push({
                        clipId: participant.clipId,
                        // 长度只在真的被改动时写（纯曲率拖拽必须保留原有长度与自动值）。
                        ...(lengthEdited ? { fadeInSec: manual, autoFadeInSec: 0 } : {}),
                        // 曲率/形状的落盘：缺了它们，bulk 回灌会把拖拽期的修改丢掉
                        //（旧实现同一处理——它总是带上 dir / shape）。
                        fadeInDir: nextDir,
                        // continuous 宿主：曲线是二维的，S 必须随曲率一起写回，
                        // 否则拖拽期解出的 S 会在下一次回灌时被宿主的旧值冲掉。
                        ...(continuous ? { fadeInS: Number(clip.fadeInS) || 0 } : {}),
                    });
                } else {
                    const manual = Number(clip.fadeOutSec) || 0;
                    const lengthEdited = Math.abs(manual - (base?.fadeOutSec ?? 0)) > 1e-9;
                    const nextDir = Number(clip.fadeOutDir) || 0;
                    const dirEdited = Math.abs(nextDir - (base?.fadeOutDir ?? 0)) > 1e-9;
                    if (!lengthEdited && !dirEdited) continue;
                    updates.push({
                        clipId: participant.clipId,
                        ...(lengthEdited ? { fadeOutSec: manual, autoFadeOutSec: 0 } : {}),
                        fadeOutDir: nextDir,
                        ...(continuous ? { fadeOutS: Number(clip.fadeOutS) || 0 } : {}),
                    });
                }
            }
            if (updates.length === 0) {
                endKernelGestureInteraction();
                return;
            }
            // 淡变手势**不做** `applyAutoCrossfade` 收尾：手动 fade 拖拽的结果就是
            // 手动值 + 该侧自动清零（旧实现同样不在这里重算自动交叉淡化——重算会
            // 立刻把手动值覆盖回自动值，等于拖拽无效）。
            void dispatch(setClipsStateBulkRemote({ updates }))
                .unwrap()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /** 内核框选：拖动前的选择快照（合并与回滚的基准）。 */
    return {
        handleKernelClipHover,
        handleKernelFadePreview,
        handleKernelFadeCommit,
        fadeTooltipAnchor,
        publishFadeGripInfo,
        publishFadeSideInfo,
    };
}
