import { useCallback, useEffect, useRef, useState } from "react";

import type { ParamFramesPayload } from "../../../types/api";
import { paramsApi } from "../../../services/api";
import { clamp } from "../timeline";

import type { ParamName, ParamViewSegment } from "./types";
import { isDynParam } from "./paramRanges";
import { framesToTime, timeToFrame } from "./utils";
const paramFramePeriodCache = new Map<string, number>();

/**
 * 「参数曲线取数中」提示的静默窗口（毫秒）。
 *
 * 取数由水平缩放 / 滚动 / 编辑提交驱动，IPC 往返通常 1~5 ms。短于此窗口的取数
 * 完全不让状态栏闪烁 —— 那是正常交互的一部分，不是用户需要知道的"加载"。
 * 只有真慢到会被察觉的取数才点亮提示。
 */
const LOADING_INDICATOR_DELAY_MS = 150;

export function usePianoRollData(args: {
    editParam: ParamName;
    secondaryParamIds: ParamName[];
    referenceRootTrackIds: string[];
    pitchEnabled: boolean;
    paramsEpoch: number;
    rootTrackId: string | null;
    selectedTrackId: string | null;
    scrollLeft: number;
    pxPerSec: number;
    viewWidth: number;
    viewSizeRef: React.MutableRefObject<{ w: number; h: number }>;
    scrollLeftRef: React.MutableRefObject<number>;
    pxPerSecRef: React.MutableRefObject<number>;
    invalidate: () => void;
    /** 外部通知当前是否正在进行 live 编辑（pointer down 期间为 true）。
     *  为 true 时，pitch_orig_updated 触发的曲线刷新会被推迟到 pointer-up 后执行。 */
    liveEditActiveRef?: React.MutableRefObject<boolean>;
}) {
    const {
        editParam,
        secondaryParamIds,
        referenceRootTrackIds,
        pitchEnabled,
        paramsEpoch,
        rootTrackId,
        selectedTrackId,
        scrollLeft,
        pxPerSec,
        viewWidth,
        viewSizeRef,
        scrollLeftRef,
        pxPerSecRef,
        invalidate,
        liveEditActiveRef: externalLiveEditActiveRef,
    } = args;

    // 内部 fallback：若外部未传入 liveEditActiveRef，则使用内部 ref（始终为 false）。
    const internalLiveEditActiveRef = useRef(false);
    const liveEditActiveRef = externalLiveEditActiveRef ?? internalLiveEditActiveRef;

    // pitch_orig_updated 到达时若正在编辑，将刷新推迟到 pointer-up 后执行。
    const pendingPitchUpdatedRefreshRef = useRef(false);
    /**
     * 笔画进行中被推迟的取数（见取数 effect 里的说明）。
     *
     * 与 `pendingPitchUpdatedRefreshRef` 是同一契约的两个来源：那个记的是
     * "后端分析结果到了"，这个记的是"视口/令牌变了"。两者都在
     * `notifyLiveEditEnded()` 里统一补触发。
     */
    const pendingFetchWhileEditingRef = useRef(false);
    const [paramView, setParamView] = useState<ParamViewSegment | null>(null);
    // 副参数曲线（与 edit 区分，用于叠加显示）
    const [secondaryParamViews, setSecondaryParamViews] = useState<
        Partial<Record<ParamName, ParamViewSegment>>
    >({});
    const [referencePitchViews, setReferencePitchViews] = useState<
        Record<string, ParamViewSegment>
    >({});
    const secondaryFetchReqIdRef = useRef(0);
    const referenceFetchReqIdRef = useRef(0);

    const [pitchEditUserModified, setPitchEditUserModified] = useState<boolean | null>(null);
    const [pitchEditBackendAvailable, setPitchEditBackendAvailable] = useState<boolean | null>(
        null,
    );

    const [isRefreshing, setIsRefreshing] = useState(false);

    /**
     * 参数曲线取数中（供状态栏提示）。
     *
     * ★ 刻意**不**直接由计数器派生。取数极其频繁（水平缩放、滚动、每一笔编辑提交
     * 都会触发），而 IPC 往返通常只有 1~5 ms —— 若每次取数都翻转一次这个状态，
     * 用户看到的就是"任何操作都闪一下加载中"，且每次翻转都会惊动订阅者。
     *
     * 因此改为**延迟显示**：取数开始后静默 `LOADING_INDICATOR_DELAY_MS`，窗口内完成
     * 则连一次 setState 都不发生（见 `isLoadingRef` 的等值短路）；只有真的慢到用户
     * 会察觉的取数才点亮。这才是进度提示该有的语义。
     */
    const [isLoading, setIsLoading] = useState(false);
    /** 在途取数计数。只是"要不要显示"的判定依据，不该触发渲染，故用 ref。 */
    const loadingCountRef = useRef(0);
    /** 延迟显示定时器。 */
    const loadingDelayRef = useRef<number | null>(null);
    /** `isLoading` 的镜像，用于等值短路 —— 让"快速取数"真正做到零 setState。 */
    const isLoadingRef = useRef(false);

    function applyLoadingState(next: boolean) {
        if (isLoadingRef.current === next) return;
        isLoadingRef.current = next;
        setIsLoading(next);
    }

    function beginLoading() {
        loadingCountRef.current += 1;
        if (loadingCountRef.current !== 1 || loadingDelayRef.current != null) return;
        loadingDelayRef.current = window.setTimeout(() => {
            loadingDelayRef.current = null;
            // 到点时若已无在途取数，说明是"刚清零又被取消"的竞态，不点亮。
            if (loadingCountRef.current > 0) applyLoadingState(true);
        }, LOADING_INDICATOR_DELAY_MS);
    }

    function endLoading() {
        loadingCountRef.current = Math.max(0, loadingCountRef.current - 1);
        if (loadingCountRef.current > 0) return;
        if (loadingDelayRef.current != null) {
            window.clearTimeout(loadingDelayRef.current);
            loadingDelayRef.current = null;
        }
        applyLoadingState(false);
    }

    // 卸载时清掉悬挂的延迟定时器（否则会在已卸载组件上 setState）。
    useEffect(
        () => () => {
            if (loadingDelayRef.current != null) {
                window.clearTimeout(loadingDelayRef.current);
                loadingDelayRef.current = null;
            }
        },
        [],
    );

    const fpRetryRef = useRef<Set<string>>(new Set());

    const paramViewRef = useRef<ParamViewSegment | null>(null);
    useEffect(() => {
        paramViewRef.current = paramView;
    }, [paramView]);

    const fetchDebounceRef = useRef<number | null>(null);
    const fetchReqIdRef = useRef(0);

    /**
     * 当前参数（渲染期同步刷新）。
     *
     * 【为什么需要它】取数响应是在**闭包里**回来的，而参数可能在请求在途时被切换：
     * 请求 id（`fetchReqIdRef`）只能拦住"被更新的请求取代"的响应，拦不住"参数已切
     * 换、新请求还没发出"的那一个（参数切换的去抖窗口内正是如此）。拿它比对请求
     * 发起时的参数，就能保证**只有当前参数的曲线才允许落进 state**。
     */
    const currentParamRef = useRef<ParamName>(editParam);
    currentParamRef.current = editParam;

    /** 上一次取数的「轨道|参数」作用域（用于识别参数/轨道切换并立即取数）。 */
    const lastFetchScopeRef = useRef<string | null>(null);
    const [refreshToken, setRefreshToken] = useState(0);

    const [forceParamFetchToken, setForceParamFetchToken] = useState(0);
    const lastAppliedForceParamFetchTokenRef = useRef(0);

    // Force parameter refresh when the session state changes meaningfully (undo/redo/timeline edits).
    // 同时清除旧曲线数据，避免旧数据在新数据到达前短暂显示（也修复初次导入后曲线不显示的问题）。
    //
    // 【参数 / 轨道切换为什么也必须清空】左轴（刻度 / 琴键 / 标签 / 值域）是从
    // `editParam` 与参数描述符**同步**推导的：切换后当帧就变成新参数的样子；而曲线
    // 数据要等取数回来（还带一段去抖）。不清空的话，这段窗口里画面上就是"**新参数的
    // 标尺 + 旧参数的曲线**"—— 旧参数的值被新参数的值域投影，画出一条完全不相干的
    // 线，然后才被新数据替换，用户看到的就是"切参数时曲线闪一下"。
    //
    // 清空让这段窗口里**没有曲线**（而不是错误的曲线）：曲线数据、副参数叠加、
    // 参考音高、以及音高编辑状态徽标的可用性标记全部属于"上一个参数"，一并丢弃。
    useEffect(() => {
        if (!rootTrackId) return;
        setParamView(null);
        setSecondaryParamViews({});
        setReferencePitchViews({});
        setPitchEditUserModified(null);
        setPitchEditBackendAvailable(null);
        setForceParamFetchToken((x) => x + 1);
    }, [paramsEpoch, rootTrackId, editParam]);

    // 监听 pitch_orig_updated 事件，触发曲线刷新。
    // 注意：分析进度状态（started/progress）由全局 PitchAnalysisProvider 统一管理。
    // 此处只负责在分析完成后刷新 PianoRoll 曲线数据。
    useEffect(() => {
        let disposed = false;
        let unlistenUpdated: null | (() => void) = null;

        async function setup() {
            if (editParam !== "pitch") return;
            if (!pitchEnabled) return;
            if (!rootTrackId) return;
            try {
                const mod = await import("@tauri-apps/api/event");

                type PitchOrigUpdatedPayload = { rootTrackId?: string };

                unlistenUpdated = await mod.listen<PitchOrigUpdatedPayload>(
                    "pitch_orig_updated",
                    (event) => {
                        if (disposed) return;
                        const payload = event.payload ?? {};
                        if (payload?.rootTrackId && payload.rootTrackId !== rootTrackId) return;

                        // 若用户正在绘制曲线（pointer down），推迟曲线刷新到 pointer-up 之后，
                        // 避免后端分析结果覆盖用户正在绘制的内容（liveEditOverride 机制）。
                        if (liveEditActiveRef.current) {
                            pendingPitchUpdatedRefreshRef.current = true;
                        } else {
                            setForceParamFetchToken((x) => x + 1);
                            setRefreshToken((x) => x + 1);
                        }
                    },
                );
                // await 期间 effect 可能已被清理（快速切换参数/轨道/卸载）：
                // 此时必须立刻注销，否则监听器泄漏并存活整个应用生命周期。
                if (disposed) {
                    unlistenUpdated();
                    unlistenUpdated = null;
                }
            } catch {
                // Safe no-op: browser/pywebview builds won't have the Tauri API.
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlistenUpdated) unlistenUpdated();
        };
    }, [editParam, pitchEnabled, rootTrackId, liveEditActiveRef]);

    // 监听 dyn_orig_updated 事件（动态的原声电平基线分析完成），刷新曲线。
    //
    // 与 pitch_orig_updated 同构：后端把基线写进 `dyn_orig` 后推送。
    // 【对所有参数监听】波形的「可听结果」映射在**任何**参数面板都需要 dyn
    // 基线与 volume 曲线（useLoudnessCurves），因此不能只在 dyn 面板监听 ——
    // 否则切到音量/音高等面板时，分析完成事件无人消费、波形一直用旧基线。
    // 绘制中仍遵守"推迟到 pointer-up"的既有保护。
    useEffect(() => {
        let disposed = false;
        let unlisten: null | (() => void) = null;

        async function setup() {
            if (!rootTrackId) return;
            try {
                const mod = await import("@tauri-apps/api/event");
                type DynOrigUpdatedPayload = { rootTrackId?: string };
                unlisten = await mod.listen<DynOrigUpdatedPayload>("dyn_orig_updated", (event) => {
                    if (disposed) return;
                    const payload = event.payload ?? {};
                    if (payload?.rootTrackId && payload.rootTrackId !== rootTrackId) return;
                    if (liveEditActiveRef.current) {
                        pendingPitchUpdatedRefreshRef.current = true;
                    } else {
                        setForceParamFetchToken((x) => x + 1);
                        setRefreshToken((x) => x + 1);
                    }
                });
                if (disposed) {
                    unlisten();
                    unlisten = null;
                }
            } catch {
                // Safe no-op: browser/pywebview builds won't have the Tauri API.
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlisten) unlisten();
        };
    }, [rootTrackId, liveEditActiveRef]);

    useEffect(() => {
        if (editParam !== "pitch") return;
        if (pitchEnabled) return;
        setParamView(null);
        setPitchEditUserModified(null);
        setPitchEditBackendAvailable(null);
        setReferencePitchViews({});
    }, [editParam, pitchEnabled]);

    // editParam 切换时，清除副参数缓存。
    useEffect(() => {
        setSecondaryParamViews({});
    }, [editParam, rootTrackId]);

    useEffect(() => {
        setSecondaryParamViews((prev) => {
            const next: Partial<Record<ParamName, ParamViewSegment>> = {};
            for (const paramId of secondaryParamIds) {
                if (prev[paramId]) {
                    next[paramId] = prev[paramId];
                }
            }
            return next;
        });
    }, [secondaryParamIds]);

    useEffect(() => {
        setReferencePitchViews((prev) => {
            const next: Record<string, ParamViewSegment> = {};
            for (const trackId of referenceRootTrackIds) {
                if (prev[trackId]) {
                    next[trackId] = prev[trackId];
                }
            }
            return next;
        });
    }, [referenceRootTrackIds]);

    function computeVisibleRequest() {
        const debug =
            typeof window !== "undefined" &&
            window.localStorage?.getItem("hifishifter.debugPianoRoll") === "1";

        const trackId = rootTrackId;
        if (!trackId) {
            if (debug) {
                console.debug("[PianoRollData] no rootTrackId; skip fetch");
            }
            return null;
        }

        const { w } = viewSizeRef.current;
        const sl = scrollLeftRef.current;
        // 可见窗口的秒范围：`pxPerSec` 是"秒 ↔ 像素"的唯一系数（轴的原生单位就是秒）。
        // 此前绕道 `scrollLeft / pxPerBeat × secPerBeat`，两个量都与 BPM 有关、相乘
        // 才抵消 —— 结果就是"改 BPM 触发整条曲线重取"。
        const pps = Math.max(1e-9, pxPerSecRef.current);
        const startSec = sl / pps;
        const durSec = w / pps;

        const visibleStartSec = startSec;
        const visibleDurSec = Math.max(1e-6, durSec);
        const visibleEndSec = visibleStartSec + visibleDurSec;

        const quantStepSec = 0.02;
        const q = (x: number) => {
            const step = Math.max(1e-6, quantStepSec);
            return Math.round(x / step) * step;
        };

        const fpKey = `${trackId}|${editParam}`;
        const cachedFp = paramFramePeriodCache.get(fpKey);
        const pvForFp = paramViewRef.current;
        const pvFp =
            pvForFp && pvForFp.key.startsWith(`${trackId}|${editParam}|`)
                ? pvForFp.framePeriodMs
                : null;
        const fpMs = Number(cachedFp ?? pvFp ?? 5) || 5;

        const paramCoversVisible = (() => {
            const pv = paramViewRef.current;
            if (!pv) return false;
            // Check version to invalidate old cache with wrong coordinate calculations
            if (!pv.key.startsWith(`v2|${trackId}|${editParam}|`)) return false;
            const fp = Math.max(1e-6, pv.framePeriodMs);
            const step = Math.max(1, Math.floor(pv.stride));
            const startSecPv = framesToTime(pv.startFrame, fp);
            const endFramePv = pv.startFrame + (pv.orig.length - 1) * step;
            const endSecPv = framesToTime(endFramePv, fp);
            return startSecPv <= visibleStartSec && endSecPv >= visibleEndSec;
        })();

        const paramMarginSec = visibleDurSec;
        const covParamStartSec = Math.max(0, visibleStartSec - paramMarginSec);
        const covParamDurSec = visibleDurSec + 2 * paramMarginSec;
        const paramStartSecQ = Math.max(0, q(covParamStartSec));
        const paramDurSecQ = Math.max(quantStepSec, q(covParamDurSec));

        // DEBUG: Log data request parameters（复用函数入口已读取的 debug 开关）
        const debugEnabled = debug;

        if (debugEnabled) {
            console.log("[usePianoRollData] Request params:", {
                trackId,
                editParam,
                visibleStartSec,
                visibleDurSec,
                visibleEndSec,
                paramMarginSec,
                covParamStartSec,
                covParamDurSec,
                paramStartSecQ,
                paramDurSecQ,
                framePeriodMs: fpMs,
            });
        }

        // CRITICAL FIX: Use unquantized time for precise frame calculation
        // Quantization is only for cache alignment, not coordinate calculation
        const startFrame = Math.max(0, timeToFrame(covParamStartSec, fpMs));
        // Request full-resolution curve by default.
        // With fp=5ms, even tens of seconds are only a few thousand samples.
        const viewFrames = clamp(
            Math.max(
                1,
                // Use unquantized duration for frame count calculation
                timeToFrame(covParamStartSec + covParamDurSec, fpMs) - startFrame + 1,
            ),
            1,
            200_000,
        );
        const stride = 1;
        const frameCount = viewFrames;
        // Version 2: Fixed coordinate calculation to use unquantized time
        const paramKey = `v2|${trackId}|${editParam}|${startFrame}|${frameCount}|${stride}`;

        const secondaryRequests = secondaryParamIds.map((secondaryParam) => {
            const secondaryFpKey = `${trackId}|${secondaryParam}`;
            const secondaryCachedFp = paramFramePeriodCache.get(secondaryFpKey);
            const secondaryFpMs = Number(secondaryCachedFp ?? 5) || 5;
            const secondaryStartFrame = Math.max(0, timeToFrame(covParamStartSec, secondaryFpMs));
            const secondaryFrameCount = clamp(
                Math.max(
                    1,
                    timeToFrame(covParamStartSec + covParamDurSec, secondaryFpMs) -
                        secondaryStartFrame +
                        1,
                ),
                1,
                200_000,
            );
            const secondaryParamKey = `v2|${trackId}|${secondaryParam}|${secondaryStartFrame}|${secondaryFrameCount}|${stride}`;
            return {
                secondaryParam,
                secondaryFpKey,
                secondaryFpMs,
                secondaryStartFrame,
                secondaryFrameCount,
                secondaryParamKey,
            };
        });

        const referenceRequests =
            editParam === "pitch"
                ? referenceRootTrackIds
                      .filter(
                          (referenceTrackId) => referenceTrackId && referenceTrackId !== trackId,
                      )
                      .map((referenceTrackId) => {
                          const referenceFpKey = `${referenceTrackId}|pitch`;
                          const referenceCachedFp = paramFramePeriodCache.get(referenceFpKey);
                          const referenceFpMs = Number(referenceCachedFp ?? 5) || 5;
                          const referenceStartFrame = Math.max(
                              0,
                              timeToFrame(covParamStartSec, referenceFpMs),
                          );
                          const referenceFrameCount = clamp(
                              Math.max(
                                  1,
                                  timeToFrame(covParamStartSec + covParamDurSec, referenceFpMs) -
                                      referenceStartFrame +
                                      1,
                              ),
                              1,
                              200_000,
                          );
                          const referenceParamKey = `v2|${referenceTrackId}|pitch|${referenceStartFrame}|${referenceFrameCount}|${stride}`;
                          return {
                              referenceTrackId,
                              referenceFpKey,
                              referenceFpMs,
                              referenceStartFrame,
                              referenceFrameCount,
                              referenceParamKey,
                          };
                      })
                : [];

        return {
            debug,
            trackId,
            /** 本次请求要取的参数（响应落地前用来比对"是否已被切换走"）。 */
            param: editParam,
            paramCoversVisible,
            paramKey,
            startFrame,
            frameCount,
            stride,
            fpMs,
            fpKey,
            forceParamFetchToken,
            secondaryRequests,
            referenceRequests,
        };
    }

    async function refreshVisible() {
        const req = computeVisibleRequest();
        if (!req) return;

        const {
            debug,
            trackId,
            paramCoversVisible,
            paramKey,
            startFrame,
            frameCount,
            stride,
            fpMs,
            fpKey,
            forceParamFetchToken: localForceParamFetchToken,
        } = req;

        const reqId = ++fetchReqIdRef.current;

        if (editParam === "pitch" && !pitchEnabled) {
            // Skip pitch fetch when disabled; waveform still updates.
            return;
        }

        const forceParam = localForceParamFetchToken !== lastAppliedForceParamFetchTokenRef.current;
        const shouldFetchParam = !paramCoversVisible || forceParam;

        if (editParam !== "pitch" || req.referenceRequests.length === 0) {
            setReferencePitchViews({});
        } else {
            void (async () => {
                const referenceReqId = ++referenceFetchReqIdRef.current;
                try {
                    const responses = await Promise.all(
                        req.referenceRequests.map(async (referenceReq) => {
                            const res = await paramsApi.getParamFrames(
                                referenceReq.referenceTrackId,
                                "pitch",
                                referenceReq.referenceStartFrame,
                                referenceReq.referenceFrameCount,
                                stride,
                            );
                            if (!res?.ok) return null;
                            const payload = res as ParamFramesPayload;
                            const fpRes =
                                Number(payload.frame_period_ms ?? referenceReq.referenceFpMs) ||
                                referenceReq.referenceFpMs;
                            paramFramePeriodCache.set(referenceReq.referenceFpKey, fpRes);
                            return [
                                referenceReq.referenceTrackId,
                                {
                                    key: referenceReq.referenceParamKey,
                                    framePeriodMs: fpRes,
                                    startFrame:
                                        Number(
                                            payload.start_frame ?? referenceReq.referenceStartFrame,
                                        ) || referenceReq.referenceStartFrame,
                                    stride,
                                    referenceKind: payload.reference_kind ?? "source_curve",
                                    orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                                    edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                                } as ParamViewSegment,
                            ] as const;
                        }),
                    );
                    if (referenceFetchReqIdRef.current !== referenceReqId) return;
                    // 参数已被切换：这份参考音高属于上一个参数的画面，丢弃。
                    if (req.param !== currentParamRef.current) return;
                    const next: Record<string, ParamViewSegment> = {};
                    for (const entry of responses) {
                        if (!entry) continue;
                        next[entry[0]] = entry[1];
                    }
                    setReferencePitchViews(next);
                    invalidate();
                } catch {
                    // ignore
                }
            })();
        }

        // 副参数异步加载（独立请求，不影响主参数刷新逻辑）。
        void (async () => {
            if (req.secondaryRequests.length === 0) return;
            const secReqId = ++secondaryFetchReqIdRef.current;
            try {
                const responses = await Promise.all(
                    req.secondaryRequests.map(async (secondaryReq) => {
                        const secondaryPitchEnabled =
                            secondaryReq.secondaryParam !== "pitch" ||
                            pitchEnabled ||
                            editParam === "pitch";
                        if (!secondaryPitchEnabled && secondaryReq.secondaryParam === "pitch") {
                            return null;
                        }
                        const res = await paramsApi.getParamFrames(
                            req.trackId,
                            secondaryReq.secondaryParam,
                            secondaryReq.secondaryStartFrame,
                            secondaryReq.secondaryFrameCount,
                            stride,
                        );
                        if (!res?.ok) return null;
                        const payload = res as ParamFramesPayload;
                        const fpRes =
                            Number(payload.frame_period_ms ?? secondaryReq.secondaryFpMs) ||
                            secondaryReq.secondaryFpMs;
                        paramFramePeriodCache.set(secondaryReq.secondaryFpKey, fpRes);
                        return [
                            secondaryReq.secondaryParam,
                            {
                                key: secondaryReq.secondaryParamKey,
                                framePeriodMs: fpRes,
                                startFrame:
                                    Number(
                                        payload.start_frame ?? secondaryReq.secondaryStartFrame,
                                    ) || secondaryReq.secondaryStartFrame,
                                stride,
                                referenceKind: payload.reference_kind ?? "source_curve",
                                orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                                edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                            } as ParamViewSegment,
                        ] as const;
                    }),
                );
                if (secondaryFetchReqIdRef.current !== secReqId) return;
                // 参数已被切换：副参数叠加同样属于上一个参数的画面，丢弃
                // （否则它会以旧参数的值、新参数的值域画出来）。
                if (req.param !== currentParamRef.current) return;
                setSecondaryParamViews((prev) => {
                    const next = { ...prev };
                    for (const secondaryReq of req.secondaryRequests) {
                        delete next[secondaryReq.secondaryParam];
                    }
                    for (const entry of responses) {
                        if (!entry) continue;
                        next[entry[0]] = entry[1];
                    }
                    return next;
                });
                invalidate();
            } catch {
                // ignore
            }
        })();

        if (shouldFetchParam) {
            void (async () => {
                // 复用外层已读取的 debug 开关，避免每次取数都同步访问 localStorage
                const debugEnabled = debug;
                beginLoading();
                try {
                    if (debugEnabled) {
                        console.log("[usePianoRollData] Fetching param frames:", {
                            trackId,
                            editParam,
                            startFrame,
                            frameCount,
                            stride,
                            startTimeSec: framesToTime(startFrame, fpMs),
                            endTimeSec: framesToTime(startFrame + frameCount - 1, fpMs),
                        });
                    }

                    const res = await paramsApi.getParamFrames(
                        trackId,
                        editParam,
                        startFrame,
                        frameCount,
                        stride,
                    );
                    if (fetchReqIdRef.current !== reqId) return;
                    // 【参数切换的兜底】请求 id 只拦得住"被新请求取代"的响应；参数
                    // 切换后新请求还没发出（去抖窗口）时，旧参数的响应仍是最新的那
                    // 一个 —— 若放它落地，清空过的 state 会被旧曲线重新填满，闪动
                    // 依旧。因此这里再比一次参数。
                    if (req.param !== currentParamRef.current) return;
                    if (!res?.ok) {
                        if (debug) {
                            console.debug("[PianoRollData] paramFrames not ok", {
                                trackId,
                                editParam,
                                paramKey,
                                startFrame,
                                frameCount,
                                stride,
                                res,
                            });
                        }
                        return;
                    }

                    const payload = res as ParamFramesPayload;

                    if (editParam === "pitch") {
                        const userModified = payload.pitch_edit_user_modified;
                        setPitchEditUserModified(
                            typeof userModified === "boolean" ? userModified : null,
                        );

                        const backendAvail = payload.pitch_edit_backend_available;
                        setPitchEditBackendAvailable(
                            typeof backendAvail === "boolean" ? backendAvail : null,
                        );
                    }
                    const fpRes = Number(payload.frame_period_ms ?? fpMs) || fpMs;
                    paramFramePeriodCache.set(fpKey, fpRes);

                    const receivedStartFrame =
                        Number(payload.start_frame ?? startFrame) || startFrame;
                    const receivedOrigLen = (payload.orig ?? []).length;
                    const receivedEditLen = (payload.edit ?? []).length;

                    if (debugEnabled) {
                        console.log("[usePianoRollData] Received param data:", {
                            trackId,
                            editParam,
                            requestedStartFrame: startFrame,
                            requestedFrameCount: frameCount,
                            receivedStartFrame,
                            receivedOrigLen,
                            receivedEditLen,
                            framePeriodMs: fpRes,
                            receivedStartSec: framesToTime(receivedStartFrame, fpRes),
                            receivedEndSec: framesToTime(
                                receivedStartFrame + receivedEditLen - 1,
                                fpRes,
                            ),
                            receivedDurSec: framesToTime(receivedEditLen - 1, fpRes),
                        });
                    }

                    setParamView({
                        key: paramKey,
                        framePeriodMs: fpRes,
                        startFrame: receivedStartFrame,
                        stride,
                        referenceKind: payload.reference_kind ?? "source_curve",
                        orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                        edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                    });
                    lastAppliedForceParamFetchTokenRef.current = localForceParamFetchToken;
                    invalidate();

                    if (Math.abs(fpRes - fpMs) > 1e-3) {
                        const retryKey = `${fpKey}|${fpMs}`;
                        if (!fpRetryRef.current.has(retryKey)) {
                            fpRetryRef.current.add(retryKey);
                            void Promise.resolve().then(() => refreshVisible());
                        }
                    }
                } catch {
                    // ignore
                } finally {
                    endLoading();
                }
            })();
        }
    }

    async function refreshNow() {
        const req = computeVisibleRequest();
        if (!req) return;

        const { debug, trackId, paramKey, startFrame, frameCount, stride, fpMs, fpKey } = req;

        setIsRefreshing(true);
        const reqId = ++fetchReqIdRef.current;
        const referenceReqId = ++referenceFetchReqIdRef.current;
        const shouldFetchParam = !(editParam === "pitch" && !pitchEnabled);
        try {
            beginLoading();
            const [paramRes, secondaryResults, referenceResults] = await Promise.all([
                shouldFetchParam
                    ? paramsApi.getParamFrames(
                          trackId,
                          editParam,
                          startFrame,
                          frameCount,
                          stride,
                          true,
                          isDynParam(editParam),
                      )
                    : Promise.resolve(null),
                Promise.all(
                    req.secondaryRequests.map(async (secondaryReq) => {
                        const secondaryPitchEnabled =
                            secondaryReq.secondaryParam !== "pitch" ||
                            pitchEnabled ||
                            editParam === "pitch";
                        if (!secondaryPitchEnabled && secondaryReq.secondaryParam === "pitch") {
                            return null;
                        }
                        const res = await paramsApi.getParamFrames(
                            trackId,
                            secondaryReq.secondaryParam,
                            secondaryReq.secondaryStartFrame,
                            secondaryReq.secondaryFrameCount,
                            stride,
                        );
                        if (!res?.ok) return null;
                        const secPayload = res as ParamFramesPayload;
                        const secFpRes =
                            Number(secPayload.frame_period_ms ?? secondaryReq.secondaryFpMs) ||
                            secondaryReq.secondaryFpMs;
                        paramFramePeriodCache.set(secondaryReq.secondaryFpKey, secFpRes);
                        return [
                            secondaryReq.secondaryParam,
                            {
                                key: secondaryReq.secondaryParamKey,
                                framePeriodMs: secFpRes,
                                startFrame:
                                    Number(
                                        secPayload.start_frame ?? secondaryReq.secondaryStartFrame,
                                    ) || secondaryReq.secondaryStartFrame,
                                stride,
                                referenceKind: secPayload.reference_kind ?? "source_curve",
                                orig: (secPayload.orig ?? []).map((v) => Number(v) || 0),
                                edit: (secPayload.edit ?? []).map((v) => Number(v) || 0),
                            } as ParamViewSegment,
                        ] as const;
                    }),
                ),
                editParam === "pitch"
                    ? Promise.all(
                          req.referenceRequests.map(async (referenceReq) => {
                              const res = await paramsApi.getParamFrames(
                                  referenceReq.referenceTrackId,
                                  "pitch",
                                  referenceReq.referenceStartFrame,
                                  referenceReq.referenceFrameCount,
                                  stride,
                              );
                              if (!res?.ok) return null;
                              const payload = res as ParamFramesPayload;
                              const fpRes =
                                  Number(payload.frame_period_ms ?? referenceReq.referenceFpMs) ||
                                  referenceReq.referenceFpMs;
                              paramFramePeriodCache.set(referenceReq.referenceFpKey, fpRes);
                              return [
                                  referenceReq.referenceTrackId,
                                  {
                                      key: referenceReq.referenceParamKey,
                                      framePeriodMs: fpRes,
                                      startFrame:
                                          Number(
                                              payload.start_frame ??
                                                  referenceReq.referenceStartFrame,
                                          ) || referenceReq.referenceStartFrame,
                                      stride,
                                      referenceKind: payload.reference_kind ?? "source_curve",
                                      orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                                      edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                                  } as ParamViewSegment,
                              ] as const;
                          }),
                      )
                    : Promise.resolve([]),
            ]);

            if (fetchReqIdRef.current !== reqId) return;
            // 参数已被切换：这份曲线属于上一个参数，不得落进 state（同 refreshVisible）。
            if (req.param !== currentParamRef.current) return;

            if (shouldFetchParam && paramRes?.ok) {
                const payload = paramRes as ParamFramesPayload;

                if (editParam === "pitch") {
                    const userModified = payload.pitch_edit_user_modified;
                    setPitchEditUserModified(
                        typeof userModified === "boolean" ? userModified : null,
                    );

                    const backendAvail = payload.pitch_edit_backend_available;
                    setPitchEditBackendAvailable(
                        typeof backendAvail === "boolean" ? backendAvail : null,
                    );
                }
                const fpRes = Number(payload.frame_period_ms ?? fpMs) || fpMs;
                paramFramePeriodCache.set(fpKey, fpRes);
                setParamView({
                    key: paramKey,
                    framePeriodMs: fpRes,
                    startFrame: Number(payload.start_frame ?? startFrame) || startFrame,
                    stride,
                    referenceKind: payload.reference_kind ?? "source_curve",
                    orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                    edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                    // dyn 的「未画」位图随同一路取数返回（见 ParamViewSegment
                    // 的字段说明）；非 dyn 显式不携带。
                    editSentinel: isDynParam(editParam)
                        ? (payload.edit_sentinel ?? undefined)
                        : undefined,
                });

                if (Math.abs(fpRes - fpMs) > 1e-3) {
                    const retryKey = `${fpKey}|${fpMs}`;
                    if (!fpRetryRef.current.has(retryKey)) {
                        fpRetryRef.current.add(retryKey);
                        void Promise.resolve().then(() => refreshNow());
                    }
                }
            } else if (shouldFetchParam && debug) {
                console.debug("[PianoRollData] refreshNow paramFrames not ok", {
                    trackId,
                    editParam,
                    paramKey,
                    paramRes,
                });
            }

            setSecondaryParamViews((prev) => {
                const next = { ...prev };
                for (const secondaryReq of req.secondaryRequests) {
                    delete next[secondaryReq.secondaryParam];
                }
                for (const entry of secondaryResults) {
                    if (!entry) continue;
                    next[entry[0]] = entry[1];
                }
                return next;
            });
            if (referenceFetchReqIdRef.current === referenceReqId) {
                const nextReferenceViews: Record<string, ParamViewSegment> = {};
                for (const entry of referenceResults) {
                    if (!entry) continue;
                    nextReferenceViews[entry[0]] = entry[1];
                }
                setReferencePitchViews(nextReferenceViews);
            }
        } finally {
            setIsRefreshing(false);
            endLoading();
            invalidate();
        }
    }

    async function refreshSecondaryNow() {
        const req = computeVisibleRequest();
        if (!req) return;

        if (req.secondaryRequests.length === 0) {
            setSecondaryParamViews({});
        }

        const secReqId = ++secondaryFetchReqIdRef.current;
        const referenceReqId = ++referenceFetchReqIdRef.current;
        beginLoading();
        try {
            const [responses, referenceResponses] = await Promise.all([
                Promise.all(
                    req.secondaryRequests.map(async (secondaryReq) => {
                        const secondaryPitchEnabled =
                            secondaryReq.secondaryParam !== "pitch" ||
                            pitchEnabled ||
                            editParam === "pitch";
                        if (!secondaryPitchEnabled && secondaryReq.secondaryParam === "pitch") {
                            return null;
                        }
                        const res = await paramsApi.getParamFrames(
                            req.trackId,
                            secondaryReq.secondaryParam,
                            secondaryReq.secondaryStartFrame,
                            secondaryReq.secondaryFrameCount,
                            req.stride,
                            true,
                            isDynParam(secondaryReq.secondaryParam),
                        );
                        if (!res?.ok) return null;

                        const payload = res as ParamFramesPayload;
                        const fpRes =
                            Number(payload.frame_period_ms ?? secondaryReq.secondaryFpMs) ||
                            secondaryReq.secondaryFpMs;
                        paramFramePeriodCache.set(secondaryReq.secondaryFpKey, fpRes);
                        return [
                            secondaryReq.secondaryParam,
                            {
                                key: secondaryReq.secondaryParamKey,
                                framePeriodMs: fpRes,
                                startFrame:
                                    Number(
                                        payload.start_frame ?? secondaryReq.secondaryStartFrame,
                                    ) || secondaryReq.secondaryStartFrame,
                                stride: req.stride,
                                referenceKind: payload.reference_kind ?? "source_curve",
                                orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                                edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                                editSentinel: isDynParam(secondaryReq.secondaryParam)
                                    ? (payload.edit_sentinel ?? undefined)
                                    : undefined,
                            } as ParamViewSegment,
                        ] as const;
                    }),
                ),
                editParam === "pitch"
                    ? Promise.all(
                          req.referenceRequests.map(async (referenceReq) => {
                              const res = await paramsApi.getParamFrames(
                                  referenceReq.referenceTrackId,
                                  "pitch",
                                  referenceReq.referenceStartFrame,
                                  referenceReq.referenceFrameCount,
                                  req.stride,
                              );
                              if (!res?.ok) return null;
                              const payload = res as ParamFramesPayload;
                              const fpRes =
                                  Number(payload.frame_period_ms ?? referenceReq.referenceFpMs) ||
                                  referenceReq.referenceFpMs;
                              paramFramePeriodCache.set(referenceReq.referenceFpKey, fpRes);
                              return [
                                  referenceReq.referenceTrackId,
                                  {
                                      key: referenceReq.referenceParamKey,
                                      framePeriodMs: fpRes,
                                      startFrame:
                                          Number(
                                              payload.start_frame ??
                                                  referenceReq.referenceStartFrame,
                                          ) || referenceReq.referenceStartFrame,
                                      stride: req.stride,
                                      referenceKind: payload.reference_kind ?? "source_curve",
                                      orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                                      edit: (payload.edit ?? []).map((v) => Number(v) || 0),
                                  } as ParamViewSegment,
                              ] as const;
                          }),
                      )
                    : Promise.resolve([]),
            ]);
            if (secondaryFetchReqIdRef.current === secReqId) {
                setSecondaryParamViews((prev) => {
                    const next = { ...prev };
                    for (const secondaryReq of req.secondaryRequests) {
                        delete next[secondaryReq.secondaryParam];
                    }
                    for (const entry of responses) {
                        if (!entry) continue;
                        next[entry[0]] = entry[1];
                    }
                    return next;
                });
            }
            if (referenceFetchReqIdRef.current === referenceReqId) {
                const nextReferenceViews: Record<string, ParamViewSegment> = {};
                for (const entry of referenceResponses) {
                    if (!entry) continue;
                    nextReferenceViews[entry[0]] = entry[1];
                }
                setReferencePitchViews(nextReferenceViews);
            }
            invalidate();
        } finally {
            endLoading();
        }
    }

    useEffect(() => {
        if (!rootTrackId) return;

        if (fetchDebounceRef.current != null) {
            window.clearTimeout(fetchDebounceRef.current);
            fetchDebounceRef.current = null;
        }

        // 【笔画进行中推迟取数】绘制期间换入新曲线，会让进行中的笔画以新数据重画
        // （表现为笔画被打断 / 跳变），并与 `liveEditOverride` 的实时预览互相覆盖。
        // 与 `pitch_orig_updated` / `dyn_orig_updated` 的处理同一契约：只记 pending，
        // 由 pointer-up 的 `notifyLiveEditEnded()` 统一补触发。
        //
        // 推迟而不是取消：pointer-up 后那次提交本身也会 bump 刷新令牌，因此这里的
        // pending 只是兜住"笔画期间发生、且提交路径没覆盖到"的变化（如缩放、滚动）。
        if (liveEditActiveRef.current) {
            pendingFetchWhileEditingRef.current = true;
            return;
        }

        // 【参数 / 轨道切换不走去抖】去抖是为滚动、缩放这类**连续输入**准备的；
        // 切换参数是离散动作，吃 75ms 去抖只会白白拉长"标尺已换、曲线未到"的空窗
        // （旧实现里那段时间还显示着旧参数的曲线）。切换时立即取数，把空窗压到
        // 一个 IPC 往返（约一帧）。
        const fetchScope = `${rootTrackId}|${editParam}`;
        if (lastFetchScopeRef.current !== fetchScope) {
            lastFetchScopeRef.current = fetchScope;
            void refreshVisible();
            return;
        }
        fetchDebounceRef.current = window.setTimeout(() => {
            fetchDebounceRef.current = null;
            void refreshVisible();
        }, 75);

        return () => {
            if (fetchDebounceRef.current != null) {
                window.clearTimeout(fetchDebounceRef.current);
                fetchDebounceRef.current = null;
            }
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [
        rootTrackId,
        selectedTrackId,
        editParam,
        secondaryParamIds,
        referenceRootTrackIds,
        pitchEnabled,
        scrollLeft,
        pxPerSec,
        viewWidth,
        refreshToken,
        forceParamFetchToken,
    ]);

    /**
     * 由外部（PianoRollPanel）在 pointer-up 时调用，通知 live 编辑已经结束。
     *
     * 集中补触发两类被推迟的刷新：
     * - `pitch_orig_updated` / `dyn_orig_updated`（后端分析结果到位）；
     * - 笔画期间被推迟的取数（视口 / 令牌变化，见取数 effect）。
     *
     * 两者都走"bump 令牌"这一条路，让取数 effect 统一按最新视口重算。
     */
    function notifyLiveEditEnded() {
        const pitchUpdated = pendingPitchUpdatedRefreshRef.current;
        const fetchDeferred = pendingFetchWhileEditingRef.current;
        pendingPitchUpdatedRefreshRef.current = false;
        pendingFetchWhileEditingRef.current = false;
        if (!pitchUpdated && !fetchDeferred) return;
        // `pitch_orig_updated` 必须**强制**取数：后端数据真的变了，覆盖检查
        // （`paramCoversVisible`）会误判"视口已覆盖"而跳过。
        if (pitchUpdated) setForceParamFetchToken((x) => x + 1);
        setRefreshToken((x) => x + 1);
    }

    // 引用必须跨渲染稳定：它被下游 commitStroke / onCanvasPointerDown 等
    // 长依赖数组的 useCallback 链消费，内联箭头函数会让整条 memo 体系
    // 每次渲染都失效重建。
    const bumpRefreshToken = useCallback(() => {
        setRefreshToken((x) => x + 1);
        // Also bump forceParamFetchToken so refreshVisible() bypasses the
        // "paramCoversVisible" cache check and actually re-fetches data
        // from the backend after edit operations.
        setForceParamFetchToken((x) => x + 1);
    }, []);

    return {
        paramView,
        setParamView,
        secondaryParamViews,
        referencePitchViews,
        bumpRefreshToken,
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isRefreshing,
        isLoading,
        pitchEditUserModified,
        pitchEditBackendAvailable,
        refreshToken,
    };
}
