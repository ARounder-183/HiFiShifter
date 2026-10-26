// 时间轴内核增益拖拽、就地编辑与交叉淡化手柄。拆自 TimelinePanel。
import React from "react";
import { clipItemGain } from "../clipGain";
import { canEditFadeCurvature } from "../../../../services/hostCapabilities";
import { isModifierActive } from "../../../../features/keybindings/keybindingsSlice";
import type { Keybinding } from "../../../../features/keybindings/types";
import {
    resolveCurvatureEditBase,
    resolveCurvePointer,
    solveNearestCurveAxes,
    solveNearestCurveDir,
} from "../reaperFade";
import { hostFadeGainForAxes } from "../hostFadeDisplay";
import { publishFadeRichTooltip } from "../fadeTooltipText";
import { effectiveFadeSec } from "../kernel/interaction/fadeTargets";
import {
    setClipsStateBulkRemote,
    setClipFades,
    setClipGain,
    moveClipStart,
    checkpointHistory,
    setClipLength,
    setClipSourceRange,
    setClipAutoFades,
} from "../../../../features/session/sessionSlice";
import type { TimelineSessionSlice } from "./useTimelineState";
import type { HostFadeMetadata } from "../../../../types/api";
import type { FadeInfoDelta } from "../fadeTooltipText";

type FadeSideInfo = {
    isOut: boolean;
    shape: number;
    dir: number;
    lengthSec: number;
    delta?: FadeInfoDelta;
    hostFades?: HostFadeMetadata;
};
import { batch } from "react-redux";
import { formatEditNumber, formatGainDbValue, gainToDb } from "../math";
import { parsePlaybackRateInput } from "../runtime/timelineCanvasStyle";
import { SNAP_HIGHLIGHT_GROUP, clearSnapHighlights } from "../../../../utils/snapHighlight";
import { resolveKernelEditParticipants } from "../hooks/kernelEditSet";
import {
    applyDetachedAutoCrossfadeClears,
    computeInitialCrossfadeSides,
} from "../hooks/autoCrossfade";
import { computeEffectiveSnap } from "../../../../utils/timelineSnapping";
import { applyBulkGainDeltaDb } from "../hooks/bulkClipEdit";
import {
    GAIN_MAX_DB,
    GAIN_MIN_DB,
    clampGainDb,
    gainDbFromKnobTravel,
    gainToDbSafe,
} from "../../../../utils/gainDb";
import { advanceFineAxisDrag, type FineAxisDragState } from "../../../../utils/fineAxisDrag";
import { readSlipClip } from "../hooks/slipWindow";
import { computeCrossfadeGrip, type CrossfadeGripClipBase } from "../hooks/crossfadeGripWindow";
import { activeClipTakeName } from "../../../../features/session/sessionTypes";
import type { useTimelineState } from "../hooks/useTimelineState";
import type { useI18n } from "../../../../i18n/I18nProvider";
import type { AppDispatch } from "../../../../app/store";

type Deps = {
    state: ReturnType<typeof useTimelineState>;
    s: TimelineSessionSlice;
    dispatch: AppDispatch;
    multiSelectedClipIds: string[];
    paramFineAdjustKb: Keybinding;
    t: ReturnType<typeof useI18n>["t"];
    commitTrackLaneGain: (clipId: string, gain: number) => void;
    commitTrackLaneRename: (clipId: string, name: string) => void;
    commitTrackLaneRate: (clipId: string, update: { rate: number }) => void;
    publishFadeGripInfo: (
        anchor: HTMLElement | null,
        earlier: Omit<FadeSideInfo, "isOut">,
        later: Omit<FadeSideInfo, "isOut">,
    ) => void;
    fadeTooltipAnchor: () => HTMLElement | null;
    beginKernelGestureInteraction: () => void;
    endKernelGestureInteraction: () => void;
};

export function useTimelineKernelGainCrossfade({
    state,
    s,
    dispatch,
    multiSelectedClipIds,
    paramFineAdjustKb,
    t,
    commitTrackLaneGain,
    commitTrackLaneRename,
    commitTrackLaneRate,
    publishFadeGripInfo,
    fadeTooltipAnchor,
    beginKernelGestureInteraction,
    endKernelGestureInteraction,
}: Deps) {
    const {
        sessionRef,
        fadeCurvatureKb,
        pxPerSecRef,
        noSnapKb,
        snapTimelineDetailed,
        crossfadeGripKb,
    } = state;
    const kernelGainOriginRef = React.useRef<{
        clipId: string;
        clipIds: string[];
        baseGainById: Map<string, number>;
        fineAxis: FineAxisDragState;
    } | null>(null);

    /**
     * 内核音量旋钮预览：竖直位移 → dB → 逐 clip 钳制后写乐观值。
     *
     * 位移先经 `gainDbFromKnobTravel` 换算（归一化行程，全程 96px，与值域大小无关），
     * 精细调整修饰键经 `advanceFineAxisDrag` 减速，钳制与多选平移复用
     * `applyBulkGainDeltaDb`。
     */
    const handleKernelGainDragPreview = React.useCallback(
        (args: {
            clipId: string;
            deltaYPx: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
        }) => {
            if (kernelGainOriginRef.current?.clipId !== args.clipId) {
                const session = sessionRef.current;
                const participants = resolveKernelEditParticipants({
                    anchorClipId: args.clipId,
                    multiSelectedClipIds,
                    clips: session.clips,
                    trackIds: session.tracks.map((track) => track.id),
                    ignoreGrouping: session.ignoreGrouping,
                    disabledGroupIds: session.disabledGroupIds,
                    // 增益拖拽不展开编组（旧实现 supportsGroupExpansion 排除 gain）。
                    expandGroups: false,
                });
                if (participants.length === 0) return;
                kernelGainOriginRef.current = {
                    clipId: args.clipId,
                    clipIds: participants.map((participant) => participant.clipId),
                    baseGainById: new Map(
                        participants.map((participant) => {
                            const clip = session.clips.find(
                                (item) => item.id === participant.clipId,
                            );
                            // 旋钮编辑 item 层（见 clipGain.ts）：基准值必须取自同一层，
                            // 否则第一次拖动会从 take 增益"跳"到 item 增益。
                            return [participant.clipId, clipItemGain(clip)] as const;
                        }),
                    ),
                    // 轴向状态以「相对按下点的位移」为 raw：状态内部只比较增量，
                    // 因此不需要绝对 clientY。
                    fineAxis: { raw: 0, adjusted: 0, fineActive: false },
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelGainOriginRef.current;
            if (origin === null) return;
            const adjustedY = advanceFineAxisDrag(
                origin.fineAxis,
                args.deltaYPx,
                isModifierActive(paramFineAdjustKb, args.modifiers),
            );
            // 【映射：先让**被抓住的那个 clip** 跟手，其余按同一个 dB 增量平移】
            //
            // 旧实现是 `deltaDb = 位移 × CLIP_GAIN_DRAG_DB_PER_PX`（固定 0.25 dB/px）。
            // 值域从 ±12 扩到 (−60, +24.1) 之后，固定 dB/px 会让拖 145px 就走完全程 ——
            // 用户明确要求"范围扩大不得让旋钮一动就大幅调整"。
            //
            // 现在位移先换算成**归一化行程**上的位置（见 `gainDbFromKnobTravel`），
            // 全程恒为 96px（与旧实现逐像素一致）；再把"锚点 clip 的新 dB − 旧 dB"
            // 作为增量应用给整个多选集合 —— 这样多选时各 clip 的相对关系仍然保留。
            const anchorBaseGain = origin.baseGainById.get(args.clipId) ?? 1;
            const anchorBaseDb = gainToDbSafe(anchorBaseGain);
            // 向上拖 = 变大 ⇒ 屏幕位移（向下为正）要取反。
            const anchorNextDb = gainDbFromKnobTravel(anchorBaseDb, -adjustedY);
            const updates = applyBulkGainDeltaDb({
                clipIds: origin.clipIds,
                clipsById: new Map(
                    [...origin.baseGainById].map(([clipId, gain]) => [clipId, { gain }]),
                ),
                deltaDb: anchorNextDb - anchorBaseDb,
                minDb: GAIN_MIN_DB,
                maxDb: GAIN_MAX_DB,
            });
            batch(() => {
                for (const update of updates) dispatch(setClipGain(update));
            });
            // 拖动中的实时浮标（旧实现 `ClipHeader` 的 `gainTooltip` 拖动变体）：
            // 普通悬停只显示当前值，拖动时还要显示**本次拖动的增量**。内容逐帧
            // 更新——浮标的锚点仍是内核容器，AppTooltip 自己跟随指针。
            const anchor =
                typeof document === "undefined"
                    ? null
                    : (document.querySelector(
                          "[data-hs-fade-tooltip-anchor]",
                      ) as HTMLElement | null);
            if (anchor !== null) {
                const baseGain = origin.baseGainById.get(args.clipId);
                // 浮标读的也是 item 层（与旋钮、header 标签同一口径）。
                const currentGain =
                    clipItemGain(
                        sessionRef.current.clips.find((item) => item.id === args.clipId),
                    ) ??
                    baseGain ??
                    1;
                const clamped = clampGainDb(gainToDbSafe(currentGain));
                publishFadeRichTooltip(
                    anchor,
                    baseGain === undefined
                        ? t("gain_value_tooltip").replace("{gain}", formatGainDbValue(clamped))
                        : t("gain_value_tooltip_drag")
                              .replace("{gain}", formatGainDbValue(clamped))
                              .replace("{delta}", formatGainDbValue(clamped - gainToDb(baseGain))),
                );
            }
        },
        [
            beginKernelGestureInteraction,
            dispatch,
            multiSelectedClipIds,
            paramFineAdjustKb,
            sessionRef,
            t,
        ],
    );

    /**
     * 内核音量旋钮收尾：提交或回滚。
     *
     * 特殊说明：未越起手阈值的单击（`changed = false`）**不写后端**——旧实现同样
     * 只在真正拖动后才落库（单击不产生 undo 步）。
     */
    const handleKernelGainDragCommit = React.useCallback(
        (args: { clipId: string; changed: boolean; cancelled: boolean }) => {
            const origin = kernelGainOriginRef.current;
            kernelGainOriginRef.current = null;
            // 未越起手阈值的单击（`changed = false`）没有远程写入：交互锁直接释放。
            if (origin === null || !args.changed) {
                endKernelGestureInteraction();
                return;
            }
            if (args.cancelled) {
                batch(() => {
                    for (const [clipId, gain] of origin.baseGainById) {
                        dispatch(setClipGain({ clipId, gain }));
                    }
                });
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 提交值取 Redux 当前值（预览已写入钳制后的结果）——用基准 + 位移重算
            // 会绕开钳制，表现为"预览到 ±12dB 上限、落库却超出"。
            const session = sessionRef.current;
            void dispatch(
                setClipsStateBulkRemote({
                    updates: origin.clipIds.map((clipId) => {
                        // 旋钮写 item 层（REAPER 的 D_VOL）；take 层由 take 面板负责。
                        const clip = session.clips.find((item) => item.id === clipId);
                        return { clipId, itemGain: clipItemGain(clip) };
                    }),
                }),
            )
                .unwrap()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /** 内核双击旋钮 → 恢复 0 dB（复用旧实现的增益提交入口，含乐观更新与落库）。 */
    const handleKernelGainReset = React.useCallback(
        (clipId: string) => {
            commitTrackLaneGain(clipId, 0);
        },
        [commitTrackLaneGain],
    );

    /**
     * 内核态行内编辑状态（重命名 / 增益 / 速率）。
     *
     * 值的格式化与解析都留在面板（领域知识：增益是 dB、速率有 `x` / `%` 前缀）；
     * 内核视图只负责把输入框定位到 clip header 上。
     */
    const [kernelInlineEdit, setKernelInlineEdit] = React.useState<{
        clipId: string;
        field: "name" | "gain" | "rate";
        initialValue: string;
        inputMode: "text" | "decimal";
    } | null>(null);

    /** 内核双击名称区 → 进入重命名。 */
    const handleKernelRenameClipStart = React.useCallback(
        (clipId: string) => {
            const clip = sessionRef.current.clips.find((item) => item.id === clipId);
            if (clip === undefined) return;
            setKernelInlineEdit({
                clipId,
                field: "name",
                // 初值必须是**活动 Take 的名字**（`activeClipTakeName`），不是容器
                // clip 的 `name`。
                //
                // 提交方（`commitTrackLaneRename` → `renameClipTakeRemote`）在多 Take
                // clip 上写的是**当前 Take 的名字**：预填容器名会让"双击 + 直接回车"
                // 把 Take 名**改成容器名**，而用户什么都没输入。旧实现 `ClipHeader`
                // 的 `editTakeName` 同源。
                initialValue: activeClipTakeName(clip),
                inputMode: "text",
            });
        },
        [sessionRef],
    );

    /**
     * 内核单击增益 / 速率标签 → 进入行内编辑。
     *
     * 初值取法与旧实现 `ClipHeader` 完全一致：速率用 `formatEditNumber(playbackRate)`、
     * 增益用钳制到 ±12dB 后的 `formatEditNumber`（编辑态必须保留精度，不能用展示级取整）。
     */
    const handleKernelBadgeEditStart = React.useCallback(
        (clipId: string, field: "gain" | "rate") => {
            const clip = sessionRef.current.clips.find((item) => item.id === clipId);
            if (clip === undefined) return;
            const initialValue =
                field === "rate"
                    ? formatEditNumber(clip.playbackRate)
                    : formatEditNumber(clampGainDb(gainToDbSafe(clip.gain)));
            setKernelInlineEdit({ clipId, field, initialValue, inputMode: "decimal" });
        },
        [sessionRef],
    );

    /**
     * 行内编辑的提交 / 取消（引用随编辑目标变化）。
     *
     * 解析失败一律按「取消」处理：旧实现同样在解析失败时静默放弃，而不是写入
     * 一个兜底值——后者会让用户的一次误输入变成一次真实的状态变更。
     */
    const kernelInlineEditProp = React.useMemo(() => {
        if (kernelInlineEdit === null) return null;
        const { clipId, field, initialValue, inputMode } = kernelInlineEdit;
        return {
            clipId,
            field,
            initialValue,
            inputMode,
            onCommit: (raw: string): void => {
                setKernelInlineEdit(null);
                if (field === "name") {
                    const trimmed = raw.trim();
                    // 空输入 = 保持原名（旧实现 `ClipHeader` 的
                    // `finalName = trimmed.length > 0 ? trimmed : editTakeName`）。
                    // 直接 return 也等价——但必须先确认**预填值就是提交目标**
                    // （多 Take clip 的预填是活动 Take 名，见
                    // `handleKernelRenameClipStart`），否则"回车不变名"会变成
                    // "回车把 Take 名改成容器名"。
                    if (trimmed.length === 0 || trimmed === initialValue) return;
                    commitTrackLaneRename(clipId, trimmed);
                    return;
                }
                if (field === "rate") {
                    const parsed = parsePlaybackRateInput(raw);
                    if (parsed == null) return;
                    commitTrackLaneRate(clipId, { rate: parsed });
                    return;
                }
                const parsed = Number.parseFloat(raw);
                if (!Number.isFinite(parsed)) return;
                commitTrackLaneGain(clipId, parsed);
            },
            onCancel: (): void => setKernelInlineEdit(null),
        };
    }, [kernelInlineEdit, commitTrackLaneRename, commitTrackLaneRate, commitTrackLaneGain]);

    /** 内核交叉点抓手：拖拽起点的两侧几何（换算位移与取消回滚）。 */
    const kernelCrossfadeOriginRef = React.useRef<{
        /** 两侧按下时的完整几何快照（含源窗口 / 循环 / 倒放 / 媒体时长）。 */
        earlier: CrossfadeGripClipBase;
        later: CrossfadeGripClipBase;
        /** 按下时的重叠长度（反向模式的淡变缩放基准）与两侧生效淡变。 */
        baseOverlapSec: number;
        earlierFadeOutSec: number;
        laterFadeInSec: number;
        earlierFadeOutAuto: boolean;
        laterFadeInAuto: boolean;
        /**
         * 曲率拖拽（`modifier.fadeCurvatureDrag` 按住）的两侧快照。
         *
         * 与旧实现 `useEditDrag` 的 `crossfadeCurveSides` 同源：区域以**秒**冻结
         * （拖拽中长度不变），形状沿用该侧当前值，`baseDir` 逐帧被求解器覆写
         * ——这样连续拖动时每一帧都从**上一帧的解**出发，手感连续且不会来回跳。
         */
        curveSides: {
            a: {
                clipId: string;
                leftSec: number;
                widthSec: number;
                shape: number;
                baseDir: number;
                /** 新轴宿主的 S 分量（`null` = 不是新轴宿主）：曲率投影要按同一条曲线算。 */
                hostS: number | null;
                /** 二维求解的 S 起点（continuous 宿主才用）。 */
                baseS: number;
            };
            b: {
                clipId: string;
                leftSec: number;
                widthSec: number;
                shape: number;
                baseDir: number;
                hostS: number | null;
                baseS: number;
            };
        };
        /**
         * 本次手势作用的两个 clip（自动交叉淡化重算 / 清理的作用集合）。
         *
         * 与旧实现 `useEditDrag` 的 `crossfadeClipIds` 同源：收尾要按**落库后的重叠
         * 关系**重算自动交叉淡化，或清理"已脱离重叠"的自动值。
         */
        xfadeClipIds: string[];
        /**
         * 按下时每侧的**重叠关系**（`computeInitialCrossfadeSides` 的结果）。
         *
         * 分开后靠它区分"这一侧的 auto 是被本次编辑弄没的"（→清零，手动 fade 恢复
         * 显示）与"本来就没有"。
         */
        initialCrossfadeSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
        /**
         * 本次手势真正触碰的侧：交叉点抓手动的恰是前块的**淡出**侧与后块的
         * **淡入**侧（旧实现 `useEditDrag` 的 `editSides` 同一口径）。
         */
        editSides: Record<string, { fadeIn: boolean; fadeOut: boolean }>;
        /** 各 clip 按下时的全部可回滚字段（取消路径用）。 */
        baseById: Map<
            string,
            {
                startSec: number;
                lengthSec: number;
                sourceStartSec: number;
                sourceEndSec: number;
                fadeInSec: number;
                fadeOutSec: number;
                autoFadeInSec: number;
                autoFadeOutSec: number;
                /**
                 * 形状 / 曲率的按下时快照。
                 *
                 * 【为什么放进快照】抓手 Tooltip 在拖拽期要显示两侧的**当前值**与
                 * **位移量**。当前曲率由求解器逐帧给出，但增量基准必须是**按下时**的
                 * 值 —— `curveSides.*.baseDir` 被求解器逐帧覆写以保持求解连续，不能
                 * 当基准；从 Redux 现读又会拿到上一帧的值（增量逐帧归零）。
                 */
                fadeInShape: number;
                fadeOutShape: number;
                fadeInDir: number;
                fadeOutDir: number;
            }
        >;
    } | null>(null);

    /**
     * 内核交叉点抓手预览。
     *
     * 几何换算整体交给共享纯函数 `computeCrossfadeGrip`（与旧实现
     * `useEditDrag` 的 `crossfade_edges` 分支同一套语义）：
     * - **同向**（默认）：两侧边缘同向平移，重叠不变；
     * - **反向**（按住 `modifier.crossfadeGrip`，默认 Ctrl/⌘）：两侧相向移动，
     *   重叠变化，两侧淡变按「新重叠 / 原重叠」比例缩放。
     *
     * 每一侧都要按「循环 / 倒放 / 非 Loop 派生窗口」改写**源窗口**——只改长度会让
     * 波形与音频对不上（旧实现的抓手拖拽同样维护源窗口）。
     */
    const handleKernelCrossfadeGripPreview = React.useCallback(
        (args: {
            earlierClipId: string;
            laterClipId: string;
            deltaSec: number;
            modifiers: { ctrlKey: boolean; shiftKey: boolean; altKey: boolean; metaKey: boolean };
            curveEnv: {
                clientY: number;
                envTopClientY: number;
                bodyHeightPx: number;
                pointerSec: number;
            };
        }) => {
            const clips = sessionRef.current.clips;
            const earlier = clips.find((item) => item.id === args.earlierClipId);
            const later = clips.find((item) => item.id === args.laterClipId);
            if (earlier === undefined || later === undefined) return;
            if (kernelCrossfadeOriginRef.current?.earlier.id !== args.earlierClipId) {
                const toBase = (clip: typeof earlier): CrossfadeGripClipBase => {
                    const view = readSlipClip(clip);
                    return {
                        id: clip.id,
                        startSec: Number(clip.startSec) || 0,
                        lengthSec: view.lengthSec,
                        sourceStartSec: view.sourceStartSec,
                        sourceEndSec: view.sourceEndSec,
                        playbackRate: view.playbackRate,
                        loopEnabled: view.loopEnabled,
                        reversed: view.reversed,
                        mediaDurationSec: view.contentDurSec ?? 0,
                    };
                };
                const baseById = new Map<
                    string,
                    {
                        startSec: number;
                        lengthSec: number;
                        sourceStartSec: number;
                        sourceEndSec: number;
                        fadeInSec: number;
                        fadeOutSec: number;
                        autoFadeInSec: number;
                        autoFadeOutSec: number;
                        /** 形状 / 曲率的按下时快照：抓手 Tooltip 的"当前值"与"增量基准"都要它们。 */
                        fadeInShape: number;
                        fadeOutShape: number;
                        fadeInDir: number;
                        fadeOutDir: number;
                    }
                >();
                for (const clip of [earlier, later]) {
                    baseById.set(clip.id, {
                        startSec: Number(clip.startSec) || 0,
                        lengthSec: Math.max(0, Number(clip.lengthSec) || 0),
                        sourceStartSec: Number(clip.sourceStartSec ?? 0) || 0,
                        sourceEndSec: Number(clip.sourceEndSec ?? 0) || 0,
                        fadeInSec: Number(clip.fadeInSec) || 0,
                        fadeOutSec: Number(clip.fadeOutSec) || 0,
                        autoFadeInSec: Number(clip.autoFadeInSec) || 0,
                        autoFadeOutSec: Number(clip.autoFadeOutSec) || 0,
                        fadeInShape: Number(clip.fadeInShape) || 0,
                        fadeOutShape: Number(clip.fadeOutShape) || 0,
                        fadeInDir: Number(clip.fadeInDir) || 0,
                        fadeOutDir: Number(clip.fadeOutDir) || 0,
                    });
                }
                kernelCrossfadeOriginRef.current = {
                    earlier: toBase(earlier),
                    later: toBase(later),
                    baseOverlapSec: Math.max(
                        0,
                        (Number(earlier.startSec) || 0) +
                            (Number(earlier.lengthSec) || 0) -
                            (Number(later.startSec) || 0),
                    ),
                    // 缩放基准取**生效**淡变（自动交叉淡化 > 0 时它赢）——用户看到的
                    // 是那条包络线，按它缩放才与画面一致。
                    earlierFadeOutSec: effectiveFadeSec(earlier.fadeOutSec, earlier.autoFadeOutSec),
                    laterFadeInSec: effectiveFadeSec(later.fadeInSec, later.autoFadeInSec),
                    earlierFadeOutAuto: Number(earlier.autoFadeOutSec ?? 0) > 0,
                    laterFadeInAuto: Number(later.autoFadeInSec ?? 0) > 0,
                    // 曲率拖拽的两侧快照：宽度取**生效**淡变并各自钳到自身长度
                    // （旧实现 `crossfadeCurveSides` 同一取法）；前块的淡出区右端贴
                    // 它的右缘，后块的淡入区左端贴它的起点。
                    curveSides: (() => {
                        const widthA = Math.max(
                            0,
                            Math.min(
                                effectiveFadeSec(earlier.fadeOutSec, earlier.autoFadeOutSec),
                                Math.max(0, Number(earlier.lengthSec) || 0),
                            ),
                        );
                        const widthB = Math.max(
                            0,
                            Math.min(
                                effectiveFadeSec(later.fadeInSec, later.autoFadeInSec),
                                Math.max(0, Number(later.lengthSec) || 0),
                            ),
                        );
                        return {
                            a: {
                                clipId: earlier.id,
                                leftSec:
                                    (Number(earlier.startSec) || 0) +
                                    (Number(earlier.lengthSec) || 0) -
                                    widthA,
                                widthSec: widthA,
                                shape: resolveCurvatureEditBase(Number(earlier.fadeOutShape) || 0)
                                    .shape,
                                baseDir: Number(earlier.fadeOutDir) || 0,
                                // 新轴宿主的 S 分量：曲率拖拽的投影必须与画布同一条曲线。
                                hostS:
                                    earlier.hostFades?.curve_mode === "continuous"
                                        ? earlier.hostFades.out_s
                                        : null,
                                // 二维求解的 S 起点（continuous 宿主才用）。
                                baseS:
                                    earlier.hostFades?.curve_mode === "continuous"
                                        ? Number(earlier.fadeOutS ?? earlier.hostFades.out_s) || 0
                                        : 0,
                            },
                            b: {
                                clipId: later.id,
                                leftSec: Number(later.startSec) || 0,
                                widthSec: widthB,
                                shape: resolveCurvatureEditBase(Number(later.fadeInShape) || 0)
                                    .shape,
                                baseDir: Number(later.fadeInDir) || 0,
                                hostS:
                                    later.hostFades?.curve_mode === "continuous"
                                        ? later.hostFades.in_s
                                        : null,
                                baseS:
                                    later.hostFades?.curve_mode === "continuous"
                                        ? Number(later.fadeInS ?? later.hostFades.in_s) || 0
                                        : 0,
                            },
                        };
                    })(),
                    xfadeClipIds: [earlier.id, later.id],
                    // 自动交叉淡化（收尾重算 / 清理）所需的两份关系快照：按下时的
                    // 重叠关系 + 本次手势触碰的侧（旧实现 `useEditDrag` 同源）。
                    initialCrossfadeSides: computeInitialCrossfadeSides(clips, [
                        earlier.id,
                        later.id,
                    ]),
                    editSides: {
                        [earlier.id]: { fadeIn: false, fadeOut: true },
                        [later.id]: { fadeIn: true, fadeOut: false },
                    },
                    baseById,
                };
                // 首个真实位移帧 = 手势开始：上交互锁（见 helper 说明）。
                beginKernelGestureInteraction();
            }
            const origin = kernelCrossfadeOriginRef.current;
            if (origin === null) return;

            // ── 曲率拖拽：`modifier.fadeCurvatureDrag`（默认 Alt）按住时改**曲率** ──
            // 与旧实现 `useEditDrag` 的 `crossfade_edges` Alt 分支同一套：交叉点上的
            // 两条包络线**各自**解一条"经过指针点"的新曲率，边缘位置与长度都完全不动。
            // 求解从上一帧的解出发（`baseDir` 逐帧覆写），连续拖动才平滑。
            if (isModifierActive(fadeCurvatureKb, args.modifiers)) {
                // 同单侧分支：修饰键要的是曲率，做不了就不做，不退化成移动几何。
                if (!canEditFadeCurvature()) return;
                const sides = origin.curveSides;
                const ptA = resolveCurvePointer(
                    args.curveEnv,
                    { leftSec: sides.a.leftSec, widthSec: sides.a.widthSec },
                    args.curveEnv.pointerSec,
                    args.curveEnv.clientY,
                );
                const ptB = resolveCurvePointer(
                    args.curveEnv,
                    { leftSec: sides.b.leftSec, widthSec: sides.b.widthSec },
                    args.curveEnv.pointerSec,
                    args.curveEnv.clientY,
                );
                if (ptA === null || ptB === null) return;
                // 取成本地常量：闭包里读 `sides.a.hostS` 会丢掉 `!== null` 的收窄。
                const hostSA = sides.a.hostS;
                const hostSB = sides.b.hostS;
                const aspectA =
                    args.curveEnv.bodyHeightPx /
                    Math.max(1, sides.a.widthSec * pxPerSecRef.current);
                const aspectB =
                    args.curveEnv.bodyHeightPx /
                    Math.max(1, sides.b.widthSec * pxPerSecRef.current);
                // continuous 宿主：两轴二维求解（见单侧拖拽处的说明）。
                const solvedA =
                    hostSA === null
                        ? null
                        : solveNearestCurveAxes({
                              mode: "out",
                              curvature: sides.a.baseDir,
                              s: sides.a.baseS,
                              pointerX01: ptA.t,
                              pointerY01: ptA.gain,
                              aspectYOverX: aspectA,
                              gainAt: (t, c, s) => hostFadeGainForAxes("out", c, s, t),
                          });
                const solvedB =
                    hostSB === null
                        ? null
                        : solveNearestCurveAxes({
                              mode: "in",
                              curvature: sides.b.baseDir,
                              s: sides.b.baseS,
                              pointerX01: ptB.t,
                              pointerY01: ptB.gain,
                              aspectYOverX: aspectB,
                              gainAt: (t, c, s) => hostFadeGainForAxes("in", c, s, t),
                          });
                const dirA = solvedA
                    ? solvedA.curvature
                    : solveNearestCurveDir({
                          shape: sides.a.shape,
                          dir: sides.a.baseDir,
                          mode: "out",
                          pointerX01: ptA.t,
                          pointerY01: ptA.gain,
                          aspectYOverX: aspectA,
                          // 与画布同一条曲线（见单侧拖拽处的说明）。
                          ...(hostSA === null
                              ? {}
                              : {
                                    gainAt: (t: number, dir: number) =>
                                        hostFadeGainForAxes("out", dir, hostSA, t),
                                }),
                      }).dir;
                const dirB = solvedB
                    ? solvedB.curvature
                    : solveNearestCurveDir({
                          shape: sides.b.shape,
                          dir: sides.b.baseDir,
                          mode: "in",
                          pointerX01: ptB.t,
                          pointerY01: ptB.gain,
                          aspectYOverX: aspectB,
                          ...(hostSB === null
                              ? {}
                              : {
                                    gainAt: (t: number, dir: number) =>
                                        hostFadeGainForAxes("in", dir, hostSB, t),
                                }),
                      }).dir;
                sides.a.baseDir = dirA;
                sides.b.baseDir = dirB;
                // 逐帧覆写 S 起点：二维求解从上一帧的解出发，连续拖动才平滑。
                if (solvedA) sides.a.baseS = solvedA.s;
                if (solvedB) sides.b.baseS = solvedB.s;
                batch(() => {
                    dispatch(
                        setClipFades({
                            clipId: sides.a.clipId,
                            fadeOutDir: dirA,
                            ...(solvedA ? { fadeOutS: solvedA.s } : {}),
                        }),
                    );
                    dispatch(
                        setClipFades({
                            clipId: sides.b.clipId,
                            fadeInDir: dirB,
                            ...(solvedB ? { fadeInS: solvedB.s } : {}),
                        }),
                    );
                });
                // 曲率拖拽的实时浮标（双列）：曲率行各带 `[±增量]`，长度行只给当前值。
                // 增量基准取按下时的 dir（`origin.baseById`）—— `sides.*.baseDir` 被
                // 本分支逐帧覆写以保持求解连续，不能当基准。
                const curveBaseA = origin.baseById.get(earlier.id);
                const curveBaseB = origin.baseById.get(later.id);
                publishFadeGripInfo(
                    fadeTooltipAnchor(),
                    {
                        shape: curveBaseA?.fadeOutShape ?? 0,
                        dir: dirA,
                        lengthSec: origin.earlierFadeOutSec,
                        hostFades: earlier.hostFades,
                        delta: { dir: dirA - (curveBaseA?.fadeOutDir ?? 0) },
                    },
                    {
                        shape: curveBaseB?.fadeInShape ?? 0,
                        dir: dirB,
                        lengthSec: origin.laterFadeInSec,
                        hostFades: later.hostFades,
                        delta: { dir: dirB - (curveBaseB?.fadeInDir ?? 0) },
                    },
                );
                return;
            }

            // ── 吸附（`snapTimelineDetailed`，与 clip 拖拽同一套多源候选）──
            //
            // 旧实现把**指针时间**吸附后再换算位移（`const rawDelta = beat - drag.basePointerSec`，
            // 其中 `beat` 是吸附后的指针时间），因此交叉点会落在网格 / 其它 clip 边缘上。
            // 内核只给几何位移，吸附规则不在内核里重写——这里用同一个函数补回来。
            //
            // `pointerSec`（当前指针内容秒）与 `deltaSec` 都来自内核的同一次测量，
            // 两者相减即按下时的指针时间（每帧恒定），无需在快照里另存一份。
            //
            // 免吸附修饰键（默认 Shift）临时取反吸附总开关；`highlight` 必须传，
            // 否则吸附生效但没有视觉反馈（与 clip 拖拽的注释同一理由）。
            const rawPointerSec = args.curveEnv.pointerSec;
            const basePointerSec = rawPointerSec - args.deltaSec;
            const snapActive = computeEffectiveSnap(
                s.snapEnabled,
                isModifierActive(noSnapKb, args.modifiers),
            );
            // 【吸附目标：**冻结**的两侧边界 + 其它稳定候选】
            //
            // 抓手一次拖动两侧 clip，而用户真正想对齐的正是这两个 clip 的边界：
            // 前块的**终止位置**与后块的**起始位置**。它们不能作为**实时**候选——
            // 那两个 clip 自己随拖动移动，实时候选会让吸附追着自己的尾巴跑
            // （实测反向模式：目标随指针漂移，落点比目标偏 0.14s）。因此：
            // 1. 把这两个 clip 放进 `excludeClipIds`，排除它们的实时边界；
            // 2. 用 `extraCandidates` 给出它们在**按下时**的冻结边界（拖拽起点几何，
            //    来自手势快照 `origin`，全程不变）。
            // 网格、其它 clip 边缘等其余候选本来就是稳定的，照常参与。
            // 优先级沿用引擎给 clip 边缘的取值（clipStart 20 / clipEnd 21）。
            const frozenCandidates = [
                {
                    sec: origin.earlier.startSec + origin.earlier.lengthSec,
                    kind: "clipEnd" as const,
                    priority: 21,
                    clipId: earlier.id,
                    trackId: earlier.trackId,
                },
                {
                    sec: origin.later.startSec,
                    kind: "clipStart" as const,
                    priority: 20,
                    clipId: later.id,
                    trackId: later.trackId,
                },
            ];
            const snappedPointerSec = snapActive
                ? snapTimelineDetailed(rawPointerSec, "clip", {
                      originSec: basePointerSec,
                      // 交叉点必然在**同一条轨道**内，吸附候选也就取该轨道
                      // （旧实现的 `anchorTrackId` 同为被拖 clip 的轨道）。
                      anchorTrackId: earlier.trackId,
                      excludeClipIds: new Set([earlier.id, later.id]),
                      extraCandidates: frozenCandidates,
                      highlight: {
                          sources: [{ trackId: earlier.trackId, clipId: earlier.id }],
                      },
                  }).sec
                : rawPointerSec;
            if (!snapActive) clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            const deltaSec = snappedPointerSec - basePointerSec;
            const result = computeCrossfadeGrip({
                earlier: origin.earlier,
                later: origin.later,
                deltaSec,
                // 反向模式：与旧实现同源（`modifier.crossfadeGrip`）。
                opposite: isModifierActive(crossfadeGripKb, args.modifiers),
                baseOverlapSec: origin.baseOverlapSec,
                earlierFadeOutSec: origin.earlierFadeOutSec,
                laterFadeInSec: origin.laterFadeInSec,
                earlierFadeOutAuto: origin.earlierFadeOutAuto,
                laterFadeInAuto: origin.laterFadeInAuto,
            });
            if (result === null) return;
            batch(() => {
                dispatch(
                    setClipLength({
                        clipId: result.earlier.clipId,
                        lengthSec: result.earlier.lengthSec,
                    }),
                );
                if (result.earlier.sourceStartSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.earlier.clipId,
                            sourceStartSec: result.earlier.sourceStartSec,
                        }),
                    );
                }
                if (result.earlier.sourceEndSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.earlier.clipId,
                            sourceEndSec: result.earlier.sourceEndSec,
                        }),
                    );
                }
                dispatch(
                    moveClipStart({ clipId: result.later.clipId, startSec: result.later.startSec }),
                );
                dispatch(
                    setClipLength({
                        clipId: result.later.clipId,
                        lengthSec: result.later.lengthSec,
                    }),
                );
                if (result.later.sourceStartSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.later.clipId,
                            sourceStartSec: result.later.sourceStartSec,
                        }),
                    );
                }
                if (result.later.sourceEndSec !== undefined) {
                    dispatch(
                        setClipSourceRange({
                            clipId: result.later.clipId,
                            sourceEndSec: result.later.sourceEndSec,
                        }),
                    );
                }
                // 反向模式：两侧淡变按新重叠比例缩放。
                //
                // 【必须分流到两个 reducer】自动交叉淡化与手动 fade 是**分离存储**的
                // 两套字段：`computeCrossfadeGrip` 已经按"这一侧是不是 auto"分流给出
                // 了目标字段（auto 侧给 autoFade*、手动侧给 fade*），这里必须按同一
                // 口径派发——全都塞给 `setClipFades` 会让 auto 字段被静默丢弃，
                // 结果是"反向拖动后 auto 淡变不再等于重叠长度"（旧实现同样是分流的）。
                for (const fade of result.fades) {
                    if (fade.fadeInSec !== undefined || fade.fadeOutSec !== undefined) {
                        dispatch(setClipFades(fade));
                    }
                    if (fade.autoFadeInSec !== undefined || fade.autoFadeOutSec !== undefined) {
                        dispatch(setClipAutoFades(fade));
                    }
                }
            });
            // 抓手拖拽的实时浮标（双列）：两侧各带自己的长度位移。
            //
            // 默认模式下重叠不变、两侧淡变长度**不动**（只移动 clip 边缘），此时位移
            // 为 0 ⇒ 不出现方括号，读数仍随边缘移动实时刷新；反向模式按比例缩放，
            // 两侧位移各不相同，因此各自从自己的按下时生效长度算起。
            const gripBaseA = origin.baseById.get(earlier.id);
            const gripBaseB = origin.baseById.get(later.id);
            const fadeOf = (clipId: string) => result.fades.find((fade) => fade.clipId === clipId);
            const fadeA = fadeOf(earlier.id);
            const fadeB = fadeOf(later.id);
            const lengthA = fadeA?.fadeOutSec ?? fadeA?.autoFadeOutSec ?? origin.earlierFadeOutSec;
            const lengthB = fadeB?.fadeInSec ?? fadeB?.autoFadeInSec ?? origin.laterFadeInSec;
            publishFadeGripInfo(
                fadeTooltipAnchor(),
                {
                    shape: gripBaseA?.fadeOutShape ?? 0,
                    dir: gripBaseA?.fadeOutDir ?? 0,
                    lengthSec: lengthA,
                    hostFades: earlier.hostFades,
                    delta: { lengthSec: lengthA - origin.earlierFadeOutSec },
                },
                {
                    shape: gripBaseB?.fadeInShape ?? 0,
                    dir: gripBaseB?.fadeInDir ?? 0,
                    lengthSec: lengthB,
                    hostFades: later.hostFades,
                    delta: { lengthSec: lengthB - origin.laterFadeInSec },
                },
            );
        },
        // 吸附总开关是设置态，但交叉抓手预览在拖拽中实时读取它：若不进依赖，
        // 回调只在挂载时创建一次，之后切换吸附开关再拖抓手仍按挂载时的快照
        // 吸附/免吸附，与 clip 拖拽 / trim 的实时口径不一致（同源处理见
        // handleKernelDragPreview / handleKernelTrimPreview 的依赖表）。
        [
            beginKernelGestureInteraction,
            crossfadeGripKb,
            dispatch,
            // 曲率分支用它把归一化 t 换算回屏幕距离权重。
            fadeCurvatureKb,
            noSnapKb,
            // 曲率纵横比换算读实时 pxPerSec（缩放后创建时快照会算错）。
            pxPerSecRef,
            sessionRef,
            s.snapEnabled,
            snapTimelineDetailed,
            fadeTooltipAnchor,
            publishFadeGripInfo,
        ],
    );

    /**
     * 内核交叉点抓手收尾：取消则回滚两侧，否则提交。
     *
     * 与拖拽 / trim 同源——**提交值取 Redux 当前值**（预览已写入钳制后的结果），
     * 用 origin + 位移重算会绕开钳制，表现为「松手后跳回越界位置」。
     */
    const handleKernelCrossfadeGripCommit = React.useCallback(
        (args: {
            earlierClipId: string;
            laterClipId: string;
            deltaSec: number;
            cancelled: boolean;
        }) => {
            const origin = kernelCrossfadeOriginRef.current;
            kernelCrossfadeOriginRef.current = null;
            if (origin === null) return;
            // 手势结束：清掉吸附高亮。
            //
            // 【为什么必须显式清】预览通过 `snapTimelineDetailed({ highlight })` 发布
            // 高亮，而**只有再次带 `highlight` 调用**才会清除它 —— 松手后不再有预览帧，
            // 没人来清。其它吸附手势的收尾各自负责：clip 拖拽 / 裁切在提交里
            // `clearSnapHighlights`，吸附偏移走 `endSnapGesture()` 的深度归零兜底；
            // 交叉点抓手此前两者都没有，于是拖拽中亮起的吸附竖线**一直留在画面上**
            //（用户报告："按住交叉淡化反向模式拖抓手，松手后高亮线不消失"）。
            clearSnapHighlights(SNAP_HIGHLIGHT_GROUP);
            if (args.cancelled) {
                // 取消：两侧**全部字段**一起还原——预览改过源窗口与淡变（反向模式），
                // 只还原起点 / 长度会让 Redux 停在半途、与后端分叉。
                batch(() => {
                    for (const [clipId, base] of origin.baseById) {
                        dispatch(moveClipStart({ clipId, startSec: base.startSec }));
                        dispatch(setClipLength({ clipId, lengthSec: base.lengthSec }));
                        dispatch(
                            setClipSourceRange({
                                clipId,
                                sourceStartSec: base.sourceStartSec,
                                sourceEndSec: base.sourceEndSec,
                            }),
                        );
                        dispatch(
                            setClipFades({
                                clipId,
                                fadeInSec: base.fadeInSec,
                                fadeOutSec: base.fadeOutSec,
                            }),
                        );
                        dispatch(
                            setClipAutoFades({
                                clipId,
                                autoFadeInSec: base.autoFadeInSec,
                                autoFadeOutSec: base.autoFadeOutSec,
                            }),
                        );
                    }
                });
                endKernelGestureInteraction();
                return;
            }
            // 提交值取 Redux 当前值（预览已写入钳制后的结果，含曲率 / 淡变 / auto）。
            const clips = sessionRef.current.clips;
            const earlier = clips.find((item) => item.id === origin.earlier.id);
            const later = clips.find((item) => item.id === origin.later.id);
            if (earlier === undefined || later === undefined) {
                endKernelGestureInteraction();
                return;
            }
            dispatch(checkpointHistory());
            // 源窗口必须一起提交：只写长度会让后端按旧源区间重新解释内容
            // （波形与音频都对不上）——与裁切提交同一约束。
            //
            // 【淡变长度 / 曲率 / 自动交叉淡化也必须一起提交】两处由来：
            // - 曲率拖拽（`modifier.fadeCurvatureDrag`）在拖拽期只改 Redux 的
            //   `fadeInDir / fadeOutDir`；只提交几何会让批量 fulfilled 的整份时间线
            //   回灌把拖拽期的曲率**丢弃**（表现为"松开鼠标后曲率没改对"）。
            // - 反向模式按比例缩放两侧淡变（写 auto 或手动字段），同样只在 Redux 里。
            //
            // 【为什么**不**提交 `fadeInShape`/`fadeOutShape`】交叉点拖拽改的是**几何**
            // （重叠与两侧长度），不是形状。而 ≥7.81 的宿主上"补丁里带 shape"会让后端
            // 走"写 `(curvature, S)` 预设对并丢弃补丁里的 dir"那条分支
            // （`host_edit.rs` 的 `continue`）—— 于是**纯几何的握把拖拽会把两端曲率
            // 重置成预设默认值**（用户看到的"拖了一下交叉点，曲线形状自己变了"）。
            // 形状的修改只走显式路径（形状行 / Ctrl+点击循环），不搭几何提交的车。
            const persist = dispatch(
                setClipsStateBulkRemote({
                    updates: [
                        {
                            clipId: earlier.id,
                            startSec: earlier.startSec,
                            lengthSec: earlier.lengthSec,
                            sourceStartSec: earlier.sourceStartSec,
                            sourceEndSec: earlier.sourceEndSec,
                            fadeInSec: Number(earlier.fadeInSec) || 0,
                            fadeOutSec: Number(earlier.fadeOutSec) || 0,
                            autoFadeInSec: Number(earlier.autoFadeInSec) || 0,
                            autoFadeOutSec: Number(earlier.autoFadeOutSec) || 0,
                            fadeInDir: Number(earlier.fadeInDir) || 0,
                            fadeOutDir: Number(earlier.fadeOutDir) || 0,
                            // continuous 宿主：S 是曲线的一部分，随曲率一起提交（旧轴宿主
                            // 不写 —— `validate_fade_axes` 会拒绝任何 S 字段）。
                            ...(earlier.hostFades?.curve_mode === "continuous"
                                ? {
                                      fadeInS: Number(earlier.fadeInS) || 0,
                                      fadeOutS: Number(earlier.fadeOutS) || 0,
                                  }
                                : {}),
                        },
                        {
                            clipId: later.id,
                            startSec: later.startSec,
                            lengthSec: later.lengthSec,
                            sourceStartSec: later.sourceStartSec,
                            sourceEndSec: later.sourceEndSec,
                            fadeInSec: Number(later.fadeInSec) || 0,
                            fadeOutSec: Number(later.fadeOutSec) || 0,
                            autoFadeInSec: Number(later.autoFadeInSec) || 0,
                            autoFadeOutSec: Number(later.autoFadeOutSec) || 0,
                            fadeInDir: Number(later.fadeInDir) || 0,
                            fadeOutDir: Number(later.fadeOutDir) || 0,
                            ...(later.hostFades?.curve_mode === "continuous"
                                ? {
                                      fadeInS: Number(later.fadeInS) || 0,
                                      fadeOutS: Number(later.fadeOutS) || 0,
                                  }
                                : {}),
                        },
                    ],
                }),
            ).unwrap();
            void (async () => {
                try {
                    await persist;
                } finally {
                    // 自动交叉淡化收尾：**无条件**清理"已脱离重叠"的自动值。
                    //
                    // 【为什么只清理、不重算（与旧实现逐条一致）】交叉点抓手与淡变
                    // 长度是同一件事的两种表现：反向模式在**拖拽中**就按「新重叠 /
                    // 原重叠」把两侧淡变缩放了（`computeCrossfadeGrip` 的 fades，auto
                    // 侧写 auto、手动侧写手动），随整份 patch 落盘；因此收尾只需要把
                    // "因这次拖动而不再重叠"的 auto 清零——不清会让历史 auto 值永远
                    // 盖住应恢复的手动 fade（REAPER 导入的工程尤其明显）。
                    //
                    // 旧实现 `useEditDrag` 也是这个口径：`shouldApplyAutoCrossfade`
                    // 只列 trim/stretch（不含 `crossfade_edges`），交叉点拖拽落到
                    // `runWithOptionalAutoCrossfade` 的 else 分支 = 只做 detached clears。
                    // 开关关闭时同样必须清（"即使开关关闭也要清"是该函数的既有契约）。
                    const latest = sessionRef.current;
                    await applyDetachedAutoCrossfadeClears(
                        latest,
                        origin.xfadeClipIds,
                        dispatch,
                        origin.initialCrossfadeSides,
                        origin.editSides,
                    );
                }
            })()
                .catch(() => undefined)
                .finally(endKernelGestureInteraction);
        },
        [dispatch, endKernelGestureInteraction, sessionRef],
    );

    /** 内核交互回调集合；插件保留选择/查看/参数动作，不编辑宿主clip几何。 */
    return {
        handleKernelGainDragPreview,
        handleKernelGainDragCommit,
        handleKernelGainReset,
        handleKernelRenameClipStart,
        handleKernelBadgeEditStart,
        kernelInlineEditProp,
        handleKernelCrossfadeGripPreview,
        handleKernelCrossfadeGripCommit,
    };
}
