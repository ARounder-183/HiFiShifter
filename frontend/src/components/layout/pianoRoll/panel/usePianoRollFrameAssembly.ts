/**
 * 参数编辑器的**帧装配**：坐标投影、波形 peaks、live 覆盖、曲线图层与唯一的绘制入口。
 *
 * 【为什么是独立的一个 hook】这一段原本是面板模型里最长的一整块（2241-2726），而它
 * 描述的其实只有一件事：**一帧画面由什么组成**。轴投影 → 可见窗口 → per-clip 峰值
 * → 绘制中的 live 覆盖 → 曲线 / 选区 / 剪贴板图层描述符 → `drawRef.current`。
 * 这些量彼此之间全是"下一层读上一层"，与模型里其余的功能（工具栏、MIDI 导入、
 * 编辑命令分发）没有交集；拆出来之后，"一帧怎么组装"第一次可以整块读。
 *
 * 【所有权不变】所有状态仍在面板（`useLiveParamEditing` / `useClipsPeaksForPianoRoll`
 * 的调用点、各 ref、`drawRef` 的写入时机全部留在原来的相对位置）。本 hook 只是把
 * 这些语句搬进了一个子 hook，因此 hook 顺序、effect 注册顺序、依赖数组与拆分前
 * **逐条相同** —— 这也是调用点必须固定在模型原处的原因。
 *
 * 【与模型的分工】模型负责"哪些数据存在"，这里负责"这一帧画什么"。模型把轴投影、
 * 图层描述符构造函数与绘制入口原样透出给视图层（`sceneBuilders` / `drawFrame`
 * 仍从模型返回值取用），因此视图侧的契约没有变化。
 */
import { useCallback, useEffect, useMemo, useRef, type RefObject } from "react";
import type { TimelineAxis } from "../../renderKernel/timelineAxis.js";
import {
    createTimelineAxis,
    viewportEndSec,
    viewportStartSec,
} from "../../renderKernel/timelineAxis.js";
import { createTickAxis } from "../../timeline/runtime/tickAxis.js";
import { shiftPitchValue } from "../../timeline/clipPitchDrag";
import type { MutablePianoRollKernelData } from "../kernel/host/pianoRollKernelData";
import type {
    PianoRollCurveLayer,
    PianoRollSelectionBandSpec,
} from "../kernel/host/pianoRollKernelData";
import type { PianoRollKernelHost } from "../kernel/host/pianoRollKernelHost";
import type { ParamClipboardData } from "../paramClipboardMapping";
import { clampParamWriteValue } from "../paramRanges";
import type { FrameRange, ParamSelection } from "../paramSelection";
import type { DetectedPitchCurve, ReferencePitchOverlay } from "../render";
import { buildReferencePitchStrokeColor } from "../referenceRootTracks";
import type { ParamViewSegment, ValueViewport } from "../types";
import { useClipsPeaksForPianoRoll } from "../useClipsPeaksForPianoRoll";
import { useLiveParamEditing } from "../useLiveParamEditing";
import type { LiveEditFlag } from "../liveEditFlag";
import { drawPianoRollFrame } from "./drawFrame";
import type { ModelDeps } from "./modelDeps";
import {
    buildCurveLayers as buildCurveLayersImpl,
    buildSelectionBandSpec as buildSelectionBandSpecImpl,
    secSpanClipRect as secSpanClipRectImpl,
    selectionClipRect as selectionClipRectImpl,
} from "./sceneBuilders";

/**
 * 本 hook 的输入。
 *
 * 能从 `ModelDeps`（模型字段的声明类型表）取的字段一律从那里取：那份表的存在
 * 就是为了让拆出去的模块声明"我需要模型里的哪几个值"而不引用模型自身（否则类型
 * 会绕成环，见该文件的说明）。模型**没有对外暴露**的中间值（缩放 / 滚动位置 /
 * 轨道组 / 响度取数序号等）在这里就地声明。
 */
type PianoRollFrameAssemblyInputs = {
    // —— 会话 / 主题 / i18n ——
    s: ModelDeps["s"];
    t: ModelDeps["t"];
    themeMode: ModelDeps["themeMode"];
    fontFamily: ModelDeps["fontFamily"];
    // —— 视口（渲染期真值）——
    pxPerSec: number;
    pxPerBeat: number;
    scrollLeft: number;
    viewSize: ModelDeps["viewSize"];
    // —— 当前参数与门禁 ——
    editParam: ModelDeps["editParam"];
    pitchEnabled: ModelDeps["pitchEnabled"];
    pitchHardDisableReason: ModelDeps["pitchHardDisableReason"];
    childPitchHardDisableReason: ModelDeps["childPitchHardDisableReason"];
    effectiveProjectScale: ModelDeps["effectiveProjectScale"];
    // —— 数据窗口 ——
    trackClips: ModelDeps["s"]["clips"];
    groupTrackIds: Set<string>;
    rootTrackId: ModelDeps["rootTrackId"];
    rootTrack: ModelDeps["s"]["tracks"][number] | null;
    visibleReferenceRootTrackIds: string[];
    visibleSecondaryParamIds: ModelDeps["visibleSecondaryParamIds"];
    hoveredReferenceRootTrackId: string | null;
    snapToggleHeld: boolean;
    // —— 取数 hook（`usePianoRollData`）的切片 ——
    paramView: ModelDeps["paramView"];
    setParamView: ModelDeps["setParamView"];
    secondaryParamViews: ModelDeps["secondaryParamViews"];
    referencePitchViews: Record<string, ParamViewSegment>;
    bumpRefreshToken: ModelDeps["bumpRefreshToken"];
    notifyLiveEditEnded: () => void;
    // —— 响度快照（`useLoudnessCurves`）的切片 ——
    loudnessAnalysisPending: ModelDeps["loudnessAnalysisPending"];
    loudnessSnapshotFetchSeq: number;
    getLatestLoudnessFetchSeq: () => number;
    flushLoudnessPending: () => void;
    // —— 选区 / 剪贴板 / 值域视口 ——
    selectionRef: RefObject<ParamSelection | null>;
    clipboardRef: RefObject<ParamClipboardData | null>;
    paramViewsRef: RefObject<Record<string, ValueViewport>>;
    pitchViewRef: RefObject<ValueViewport>;
    paramMorphOverlays: ModelDeps["paramMorphOverlays"];
    // —— 滚动 / 播放头 ref 镜像 ——
    pxPerSecRef: RefObject<number>;
    scrollLeftRef: RefObject<number>;
    viewSizeRef: RefObject<{ w: number; h: number }>;
    visualPlayheadSecRef: RefObject<number>;
    // —— 画布 / 内核 ——
    canvasRef: RefObject<HTMLCanvasElement | null>;
    axisCanvasRef: RefObject<HTMLCanvasElement | null>;
    hostRef: RefObject<PianoRollKernelHost | null>;
    kernelDataRef: RefObject<MutablePianoRollKernelData>;
    drawRef: RefObject<() => void>;
    // —— live 编辑标志 ——
    liveEditActiveRef: LiveEditFlag;
    liveEditEndHandlerRef: RefObject<(() => void) | null>;
    // —— 投影与标脏 ——
    valueToY: ModelDeps["valueToY"];
    invalidate: ModelDeps["invalidate"];
};

/**
 * 帧装配。
 *
 * 调用点必须固定在面板模型里原来的位置：`useClipsPeaksForPianoRoll` 与
 * `useLiveParamEditing` 在这里被调用，它们自身的 effect 与状态槽顺序取决于本
 * hook 的调用位置；`drawRef.current` 的写入也发生在渲染期（与拆分前同一时机）。
 */
export function usePianoRollFrameAssembly({
    s,
    t,
    themeMode,
    fontFamily,
    pxPerSec,
    pxPerBeat,
    scrollLeft,
    viewSize,
    editParam,
    pitchEnabled,
    pitchHardDisableReason,
    childPitchHardDisableReason,
    effectiveProjectScale,
    trackClips,
    groupTrackIds,
    rootTrackId,
    rootTrack,
    visibleReferenceRootTrackIds,
    visibleSecondaryParamIds,
    hoveredReferenceRootTrackId,
    snapToggleHeld,
    paramView,
    setParamView,
    secondaryParamViews,
    referencePitchViews,
    bumpRefreshToken,
    notifyLiveEditEnded,
    loudnessAnalysisPending,
    loudnessSnapshotFetchSeq,
    getLatestLoudnessFetchSeq,
    flushLoudnessPending,
    selectionRef,
    clipboardRef,
    paramViewsRef,
    pitchViewRef,
    paramMorphOverlays,
    pxPerSecRef,
    scrollLeftRef,
    viewSizeRef,
    visualPlayheadSecRef,
    canvasRef,
    axisCanvasRef,
    hostRef,
    kernelDataRef,
    drawRef,
    liveEditActiveRef,
    liveEditEndHandlerRef,
    valueToY,
    invalidate,
}: PianoRollFrameAssemblyInputs) {
    /**
     * 参数编辑器的统一坐标投影（渲染期）。
     *
     * 用 React state 的 scrollLeft / pxPerSec 构造：本面板所有**渲染期**的时间↔
     * 像素换算都走它。滚动热路径另用 ref 构造 `drawAxis`（见 drawRef），因为
     * 滚动时 ref 同步更新而 state 滞后一帧。
     *
     * 注意：`scrollLeft` 是**绘制坐标**（由 timelineViewportNativeToState 转换
     * 得到），与 axis 的契约一致；DOM 原生坐标只在量测边界转换。
     */
    const prAxis = useMemo(
        () =>
            createTimelineAxis({
                pxPerSec,
                scrollLeftPx: scrollLeft,
                viewportWidthPx: viewSize.w,
                dpr: window.devicePixelRatio || 1,
            }),
        [pxPerSec, scrollLeft, viewSize.w],
    );

    /**
     * **刻度生成**专用轴（与 `prAxis` 分开）。
     *
     * 【为什么要分开】`prAxis` 是绘制投影，消费方（画布 / 波形 / 曲线 / 命中测试）
     * 要的是真实视口；而刻度生成要的是"窗口必须覆盖真实视口"的**保守**窗口 ——
     * 这里传入的 `scrollLeft` 是 256px 量化提交的 React state，最多滞后内核 255px。
     * 生成窗口按 `createTickAxis` 的约定多留一个量化步长，覆盖才成立。此前本面板
     * 直接拿 `prAxis`（既无量化也无宽度补偿）生成刻度，缩放后窗口可能盖不住视口
     * —— 用户看到的就是"标尺文字整片消失"。
     */
    const prTickAxis = useMemo(
        () =>
            createTickAxis({
                pxPerSec,
                scrollLeftPx: scrollLeft,
                viewportWidthPx: viewSize.w,
                dpr: window.devicePixelRatio || 1,
            }),
        [pxPerSec, scrollLeft, viewSize.w],
    );

    // 可见区域的 sec 范围：仅用于数据窗口选择（clip peaks 取窗），
    // 像素换算一律经 prAxis，不得再由这里除回秒。
    const visibleStartSec = viewportStartSec(prAxis);
    const visibleEndSec = viewportEndSec(prAxis);

    // Per-clip 波形 peaks（替代原来的 mix 波形）
    const clipPeaks = useClipsPeaksForPianoRoll({
        clips: trackClips,
        visibleStartSec,
        visibleEndSec,
    });
    // Data and viewport changes should always trigger a canvas redraw.
    // usePianoRollData() may call invalidate() before these refs update,
    // so we schedule a follow-up redraw after React commits state.
    // clipPeaks 已经通过 useMemo 稳定化，只在数据真正变化时才产生新引用。
    useEffect(() => {
        invalidate();
    }, [clipPeaks, paramView, secondaryParamViews, pxPerBeat, viewSize.w, viewSize.h, invalidate]);

    useEffect(() => {
        invalidate();
    }, [editParam, visibleSecondaryParamIds, themeMode, invalidate]);

    // 检测音高曲线更新时触发重绘（必须在 detectedPitchCurves 声明之后 ?
    // useEffect 已移 ?detectedPitchCurves useMemo 定义之后，见下方 ?

    const paramViewRef = useRef<import("../types").ParamViewSegment | null>(null);
    useEffect(() => {
        paramViewRef.current = paramView;
    }, [paramView]);

    const {
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        clearCommittedLiveEditOverride,
        markLiveEditCommitted,
        commitStroke: commitStrokeBase,
    } = useLiveParamEditing({
        rootTrackId,
        editParam,
        pitchEnabled,
        paramView,
        setParamView,
        bumpRefreshToken,
        invalidate,
    });

    /**
     * 提交时记下的"已发出的最大取数序号"（见 `useLoudnessCurves` 的
     * `getLatestFetchSeq` 与下方收尾 effect）。
     */
    const committedSettleSeqRef = useRef(0);

    /**
     * **参数写入成功**后调用：让波形在快照追上之前继续显示刚提交的值。
     *
     * 【做两件事】① 记下当前的取数序号水位（只有序号更大的快照才可能是提交之后
     * 取的）；② 把 live 覆盖层标记为"已提交"—— **保留**其值而不是立刻撤下，
     * 避免波形的幅度因子退回旧快照（"松手闪回旧波形"）。
     *
     * 【谁调用】面板的 `commitStroke` 包装层，以及 `usePianoRollInteractions` 里
     * 四条**自己发起回写**的提交路径（选区拖拽 / 右键拖拽 / 直线拖拽 / morph
     * 应用）—— 它们不经过 `commitStroke`，此前都是立刻把覆盖层置空，正是闪屏的
     * 来源。失败路径仍走硬清除（见各自的 catch）。
     */
    const onParamCommitSucceeded = useCallback(() => {
        committedSettleSeqRef.current = getLatestLoudnessFetchSeq();
        markLiveEditCommitted();
    }, [getLatestLoudnessFetchSeq, markLiveEditCommitted]);

    /**
     * **提交之后取得**的响度快照到位 ⇒ 撤下"已提交"的 live 覆盖层。
     *
     * 【为什么要等】提交成功后覆盖层不立刻撤下（见 `LiveEditOverride.committed`）：
     * 它的值就是刚提交的曲线，而整工程快照还要走一趟 IPC。若此时撤下，波形的幅度
     * 因子退回**旧快照**，于是"先跳回旧波形、再恢复新波形"（用户报告的松手闪屏）。
     *
     * 【为什么用取数序号而不是"快照换了引用"】提交发生时可能有一次**提交之前就
     * 已发出**的在飞取数（例如原声基线分析完成触发的刷新），它的数据里没有本次
     * 编辑。只看"引用变了"会在这份陈旧快照到达时提前收尾、照样闪一下。比较取数
     * 序号则可以排除它：那种请求的序号 ≤ 提交时的最大值。
     *
     * 【为什么不会挡住撤销】撤销/重做经 `paramsEpoch`、切轨经 `rootTrackId`，
     * 都会重新取数并带来更大的序号，因此覆盖层的存活期最多到"下一个提交后的快照
     * 到达"，不会长期遮挡后续状态。
     */
    useEffect(() => {
        if (loudnessSnapshotFetchSeq > committedSettleSeqRef.current) {
            clearCommittedLiveEditOverride();
        }
    }, [loudnessSnapshotFetchSeq, clearCommittedLiveEditOverride]);

    // Clip 音高拖拽（修饰键 + 波形垂直拖拽）的实时预览桥：拖拽侧以节流
    // 后端预览写入修改 pitch 参数线；这里把同一音分偏移实时应用到本机
    // live 覆盖层，画布立即渲染拖拽结果 —— 与选择工具在选区内垂直拖拽的
    // 实时编辑同一渲染路径。仅当前参数为"音高"时生效（其余参数无法展示
    // 音高曲线；选区设定仍由 selectClipParamRange 完成）。
    // 提交事件把最终偏移就地写入 paramView 并清除覆盖层，与拖拽侧的
    // epoch 重取数无缝衔接（同键同值，无闪跳）。
    useEffect(() => {
        function readDragWindow(e: Event): {
            cents: number;
            startFrame: number;
            frameCount: number;
        } | null {
            const detail = (
                e as CustomEvent<{
                    cents?: number;
                    startFrame?: number;
                    frameCount?: number;
                }>
            ).detail;
            if (!detail) return null;
            const cents = Number(detail.cents ?? 0);
            const startFrame = Math.max(0, Math.floor(Number(detail.startFrame ?? 0)));
            const frameCount = Math.max(1, Math.floor(Number(detail.frameCount ?? 1)));
            if (!Number.isFinite(cents) || !Number.isFinite(startFrame)) return null;
            return { cents, startFrame, frameCount };
        }
        function applyPitchDragPreview(e: Event) {
            const drag = readDragWindow(e);
            if (!drag || !paramView || editParam !== "pitch" || !pitchEnabled) return;
            ensureLiveEditBase(paramView);
            const override = liveEditOverrideRef.current;
            if (!override || override.key !== paramView.key) return;
            const deltaSemitones = drag.cents / 100;
            const windowEndFrame = drag.startFrame + drag.frameCount;
            // 以 paramView 的原始帧为基准重复推导（预览事件幂等，不叠加），
            // 再走**统一的 live 写入入口**：它负责值域钳制（与后端同构）、版本号
            // 推进与区间记账。此前这里直接改数组 —— 既不钳制（音高拖到 127 以上
            // 时预览值与后端存下的值不一致），也不推进版本号（主画布签名与波形
            // 几何缓存都看不到变化）。
            const dense = new Array<number>(drag.frameCount);
            for (let i = 0; i < override.edit.length; i += 1) {
                const frame = paramView.startFrame + i * paramView.stride;
                if (frame < drag.startFrame || frame >= windowEndFrame) continue;
                dense[frame - drag.startFrame] = shiftPitchValue(
                    paramView.edit[i] ?? 0,
                    deltaSemitones,
                );
            }
            // 步长 > 1 时 dense 会有空洞：写入侧 `dense[j] ?? edit[i]` 会保留原值。
            applyDenseToLiveEdit(
                paramView,
                drag.startFrame,
                dense,
                drag.startFrame,
                windowEndFrame - 1,
                "draw",
            );
            invalidate();
        }
        function commitPitchDragPreview(e: Event) {
            const drag = readDragWindow(e);
            if (!drag || !paramView || editParam !== "pitch" || !pitchEnabled) return;
            const deltaSemitones = drag.cents / 100;
            const windowEndFrame = drag.startFrame + drag.frameCount;
            const nextEdit = paramView.edit.slice();
            for (let i = 0; i < nextEdit.length; i += 1) {
                const frame = paramView.startFrame + i * paramView.stride;
                if (frame < drag.startFrame || frame >= windowEndFrame) continue;
                // 与后端写入值域同构地钳制，保证"本地曲线"与"随后取回的曲线"一致。
                nextEdit[i] = clampParamWriteValue(
                    "pitch",
                    shiftPitchValue(paramView.edit[i] ?? 0, deltaSemitones),
                );
            }
            setParamView({ ...paramView, edit: nextEdit });
            liveEditOverrideRef.current = null;
            invalidate();
        }
        window.addEventListener("hifi:pitchDragPreview", applyPitchDragPreview);
        window.addEventListener("hifi:pitchDragCommit", commitPitchDragPreview);
        return () => {
            window.removeEventListener("hifi:pitchDragPreview", applyPitchDragPreview);
            window.removeEventListener("hifi:pitchDragCommit", commitPitchDragPreview);
        };
    }, [
        paramView,
        editParam,
        pitchEnabled,
        ensureLiveEditBase,
        liveEditOverrideRef,
        setParamView,
        invalidate,
        applyDenseToLiveEdit,
    ]);

    /**
     * 笔画结束时补触发所有被推迟的刷新。
     *
     * 由 `liveEditActiveRef` 的 **true→false 边沿**自动调用（见其定义），因此
     * pointerup / pointercancel / 各工具自己的中止分支都会走到这里，不需要逐个
     * 调用点记得补取。
     */
    liveEditEndHandlerRef.current = () => {
        // 曲线：用最新视口重取（含 pitch_orig_updated 的强制取数）。
        notifyLiveEditEnded();
        // 响度快照：把笔画期间被推迟的那一份落地，避免等下一次 IPC 往返。
        flushLoudnessPending();
    };

    // 包装 commitStroke：提交笔画后清除 liveEditActive 状态 —— 该写入会经边沿回调
    // 自动补触发上面两项刷新，因此这里不再显式调用。
    const commitStroke: typeof commitStrokeBase = useCallback(
        async (points, mode) => {
            // 收尾分工：**水位**记在这里（必须在发起写入之前 —— 提交内部会 bump
            // 刷新令牌、立刻发起取数）；**标记已提交**由 commitStrokeBase 在成功
            // 路径完成（失败则硬清除覆盖层）。两者合起来即 onParamCommitSucceeded
            // 的语义，见其说明。
            committedSettleSeqRef.current = getLatestLoudnessFetchSeq();
            await commitStrokeBase(points, mode);
            // 置回 false 即触发边沿回调（补取曲线 + 落地响度快照）。
            liveEditActiveRef.current = false;
        },
        [commitStrokeBase, getLatestLoudnessFetchSeq, liveEditActiveRef],
    );

    // 从 store 中的 clipPitchCurves 转换为 DetectedPitchCurve[] 供 drawPianoRoll 使用。
    // 仅在 pitch 模式下且轨道 Compose 开启时显示，其他情况下传空数组以避免不必要的计算。
    const detectedPitchCurves = useMemo((): DetectedPitchCurve[] => {
        if (editParam !== "pitch") return [];
        if (!rootTrack?.composeEnabled) return [];
        return Object.entries(s.clipPitchCurves)
            .filter(([clipId]) => {
                // 只保留属于当前轨道组内的 clip，显示 root 及所有子轨道的 detected curve
                const clip = s.clips.find((cl) => cl.id === clipId);
                return clip && groupTrackIds.has(clip.trackId) && !clip.muted;
            })
            .map(([, c]) => ({
                curveStartSec: c.curveStartSec,
                midiCurve: c.midiCurve,
                framePeriodMs: c.framePeriodMs,
            }));
    }, [editParam, rootTrack, s.clipPitchCurves, s.clips, groupTrackIds]);

    const referencePitchOverlays = useMemo((): ReferencePitchOverlay[] => {
        if (editParam !== "pitch") return [];
        return visibleReferenceRootTrackIds
            .map((trackId) => {
                const paramViewForTrack = referencePitchViews[trackId];
                if (!paramViewForTrack) return null;
                const totalPoints = Math.max(
                    paramViewForTrack.orig.length,
                    paramViewForTrack.edit.length,
                );
                if (totalPoints < 2) return null;
                const track = s.tracks.find((item) => item.id === trackId);
                return {
                    rootTrackId: trackId,
                    strokeColor: buildReferencePitchStrokeColor(
                        track?.color ?? null,
                        hoveredReferenceRootTrackId === trackId,
                    ),
                    highlighted: hoveredReferenceRootTrackId === trackId,
                    paramView: paramViewForTrack,
                };
            })
            .filter((item): item is ReferencePitchOverlay => item != null);
    }, [
        editParam,
        hoveredReferenceRootTrackId,
        referencePitchViews,
        s.tracks,
        visibleReferenceRootTrackIds,
    ]);

    // 检测音高曲线更新时触发重绘
    useEffect(() => {
        invalidate();
    }, [detectedPitchCurves, invalidate]);

    useEffect(() => {
        invalidate();
    }, [invalidate, referencePitchOverlays]);

    // Ensure pitch-snap related changes immediately redraw
    useEffect(() => {
        invalidate();
    }, [
        s.pitchSnapEnabled,
        s.pitchSnapUnit,
        effectiveProjectScale,
        s.scaleHighlightMode,
        s.tempoMap,
        snapToggleHeld,
        invalidate,
    ]);

    // 剪贴板预览开关变化时立即重绘
    useEffect(() => {
        invalidate();
    }, [s.showClipboardPreview, invalidate]);

    // 【已删除：scaleSegments 帧间缓存】
    //
    // 它存在的唯一目的是给 `drawPianoRoll` 的 `scaleSegments` 入参供数（Tempo Map
    // 分段音阶高亮）。该入参随 Canvas2D 音高网格分支一起删除（分支因 `skipGrid`
    // 恒为 true 而不可达），缓存随之失去唯一读者。
    //
    // 【分段音阶高亮的去向】GL 网格几何是视口坐标、不含时间轴，无法表达"不同时间段
    // 用不同音阶"，因此**分段高亮未迁移**；现在只支持单一工程音阶。见
    // `buildGridSpec` 的说明。`s.tempoMap` 仍被自动吸附与标尺使用，未失去读者。

    /**
     * 组装曲线图层描述符（阶段 3：曲线 GL）。
     *
     * 流程：按**绘制顺序**产出 7 类图层 —— 参考线 → 检测曲线 → 副参数 → 原始曲线
     * → 编辑曲线 → 选区高亮 → 剪贴板预览（后者在上）。顺序与 `render.ts` 的
     * Canvas2D 调用顺序**逐项对应**，因为 GL 靠数组顺序决定层叠。
     *
     * 特殊说明 1：每条图层自带 `valueToY`。副参数的值域与主参数不同（音分 / 度数 /
     * 共振峰），必须用各自的投影，否则副参数曲线会被画到错误高度——而且错得"合理"
     * （仍在视口内），极难归因。
     *
     * 特殊说明 2：虚线图案一律经 `getFixedDashPattern`（按 dpr 量化），与 Canvas2D
     * 路径同一个函数，杜绝疏密分叉。
     *
     * 特殊说明 3：`values` 传引用而不是副本——采样值不随滚动变化，每帧复制的成本
     * 在长曲线上很可观。
     *
     * 特殊说明 4：两个需要裁剪的图层（选区高亮 / 剪贴板预览）把选区矩形同时作为
     * `clipRect`（GL 侧走 scissor）与坐标边界。
     *
     * @param axis 当前投影（每帧由 drawRef 构造）。
     * @returns 曲线图层列表（按绘制顺序）；GL 关闭时不会被调用。
     */
    function buildCurveLayers(axis: TimelineAxis): PianoRollCurveLayer[] {
        return buildCurveLayersImpl(
            {
                clipboardRef,
                detectedPitchCurves,
                editParam,
                liveEditOverrideRef,
                paramView,
                pitchEnabled,
                referencePitchOverlays,
                secSpanClipRect,
                secondaryParamViews,
                selectionClipRect,
                selectionRef,
                themeMode,
                valueToY,
                viewSizeRef,
                visibleSecondaryParamIds,
            },
            axis,
        );
    }

    /**
     * 把选区（拍单位）组装为 GL 场景层的选区块镜像。
     *
     * 【为什么在这里换算帧 → 秒】宿主只认统一投影（`TimelineAxis` 以秒为单位），
     * 而选区数据是帧。帧 → 秒只依赖工程级帧周期（与 BPM 无关）；放到宿主侧就是
     * 第三份口径。因此本函数只做"业务单位 → 宿主的秒"，不改任何几何。
     *
     * 【选区带 == 被圈住的采样点】选区边界是**切点**（两帧中间，见
     * `paramSelection.snapCut`）：第 k 帧的采样点画在 `framesToTime(k)`，它的领地
     * 是左右各半帧，因此带画在 `[startFrame - 0.5, startFrame + frameCount - 0.5]`。
     * 于是"框住第 2、3 帧"的两条边界正好落在第 1/2 帧之间与第 3/4 帧之间。
     *
     * 【为什么空选区返回 null 而不是空数组】两者对宿主是同一件事（清空实例），
     * 但 null 让"没有选区"与"有选区但都在视口外"在调试时仍然可区分。
     *
     * @param selection 选区（多区间，帧）；null / 空表示无选区。
     * @returns 选区块镜像；无选区时 null。
     */
    function buildSelectionBandSpec(
        selection: ParamSelection | null,
    ): PianoRollSelectionBandSpec | null {
        return buildSelectionBandSpecImpl({ paramView, themeMode }, selection);
    }

    /**
     * 把秒区间换算为视口坐标的裁剪矩形。
     *
     * @param axis 当前投影。
     * @param startSec 区间起点（工程秒）。
     * @param endSec 区间终点（工程秒）。
     * @returns 裁剪矩形；宽度为 0 时返回 null。
     */
    function secSpanClipRect(
        axis: TimelineAxis,
        startSec: number,
        endSec: number,
    ): { x: number; y: number; w: number; h: number } | null {
        return secSpanClipRectImpl({ viewSizeRef }, axis, startSec, endSec);
    }

    /**
     * 把单段选区（帧）换算为视口坐标的裁剪矩形。
     *
     * 与选区带同口径：边界取**切点**（两帧中间）。
     *
     * @param axis 当前投影。
     * @param range 单段选区（半开帧区间）。
     * @returns 裁剪矩形；宽度为 0 时返回 null。
     */
    function selectionClipRect(
        axis: TimelineAxis,
        range: FrameRange,
    ): { x: number; y: number; w: number; h: number } | null {
        return selectionClipRectImpl({ paramView, secSpanClipRect }, axis, range);
    }

    // Keep draw function always up-to-date (invalidate() is stable and calls drawRef.current()).
    drawRef.current = () =>
        drawPianoRollFrame({
            axisCanvasRef,
            buildCurveLayers,
            buildSelectionBandSpec,
            canvasRef,
            childPitchHardDisableReason,
            clipboardRef,
            detectedPitchCurves,
            editParam,
            fontFamily,
            hostRef,
            kernelDataRef,
            liveEditOverrideRef,
            loudnessAnalysisPending,
            paramMorphOverlays,
            paramView,
            paramViewsRef,
            pitchEnabled,
            pitchHardDisableReason,
            pitchViewRef,
            pxPerSecRef,
            referencePitchOverlays,
            s,
            scrollLeftRef,
            secondaryParamViews,
            selectionRef,
            t,
            themeMode,
            valueToY,
            viewSize,
            viewSizeRef,
            visibleSecondaryParamIds,
            visualPlayheadSecRef,
        });
    return {
        prAxis,
        prTickAxis,
        clipPeaks,
        paramViewRef,
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        commitStroke,
        onParamCommitSucceeded,
        detectedPitchCurves,
        referencePitchOverlays,
        buildCurveLayers,
        buildSelectionBandSpec,
        secSpanClipRect,
        selectionClipRect,
    };
}
