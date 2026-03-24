import { useEffect, useRef, useState } from "react";

import type { ParamFramesPayload } from "../../../types/api";
import { paramsApi } from "../../../services/api";
import { clamp } from "../timeline";

import type { ParamName, ParamViewSegment } from "./types";
import { framesToTime, timeToFrame } from "./utils";
const paramFramePeriodCache = new Map<string, number>();

export function usePianoRollData(args: {
    editParam: ParamName;
    secondaryParamIds: ParamName[];
    pitchEnabled: boolean;
    paramsEpoch: number;
    rootTrackId: string | null;
    selectedTrackId: string | null;
    secPerBeat: number;
    scrollLeft: number;
    pxPerBeat: number;
    viewWidth: number;
    viewSizeRef: React.MutableRefObject<{ w: number; h: number }>;
    scrollLeftRef: React.MutableRefObject<number>;
    pxPerBeatRef: React.MutableRefObject<number>;
    invalidate: () => void;
    /** 澶栭儴閫氱煡褰撳墠鏄惁姝ｅ湪杩涜 live 缂栬緫锛坧ointer down 鏈熼棿锟?true锛夛拷?
     *  锟?true 鏃讹紝pitch_orig_updated 瑙﹀彂鐨勬洸绾垮埛鏂颁細琚帹杩熷埌 pointer-up 鍚庢墽琛岋拷?*/
    liveEditActiveRef?: React.MutableRefObject<boolean>;
}) {
    const {
        editParam,
        secondaryParamIds,
        pitchEnabled,
        paramsEpoch,
        rootTrackId,
        selectedTrackId,
        secPerBeat,
        scrollLeft,
        pxPerBeat,
        viewWidth,
        viewSizeRef,
        scrollLeftRef,
        pxPerBeatRef,
        invalidate,
        liveEditActiveRef: externalLiveEditActiveRef,
    } = args;

    // 鍐呴儴 fallback锛氳嫢澶栭儴鏈紶锟?liveEditActiveRef锛屽垯浣跨敤鍐呴儴 ref锛堝缁堜负 false锛夛拷?
    const internalLiveEditActiveRef = useRef(false);
    const liveEditActiveRef =
        externalLiveEditActiveRef ?? internalLiveEditActiveRef;

    // 锟?pitch_orig_updated 鍒拌揪鏃惰嫢姝ｅ湪缂栬緫锛屽皢鍒锋柊鎺ㄨ繜锟?pointer-up 鍚庢墽琛岋拷?
    const pendingPitchUpdatedRefreshRef = useRef(false);
    const [paramView, setParamView] = useState<ParamViewSegment | null>(null);
    // 鍓弬鏁版洸绾匡紙锟?edit锛岀敤浜庡彔鍔犳樉绀猴級
    const [secondaryParamViews, setSecondaryParamViews] = useState<
        Partial<Record<ParamName, ParamViewSegment>>
    >({});
    const secondaryFetchReqIdRef = useRef(0);

    const [pitchEditUserModified, setPitchEditUserModified] = useState<
        boolean | null
    >(null);
    const [pitchEditBackendAvailable, setPitchEditBackendAvailable] = useState<
        boolean | null
    >(null);

    const [isRefreshing, setIsRefreshing] = useState(false);
    const [loadingCount, setLoadingCount] = useState(0);
    const isLoading = loadingCount > 0;

    function beginLoading() {
        setLoadingCount((c) => c + 1);
    }
    function endLoading() {
        setLoadingCount((c) => Math.max(0, c - 1));
    }

    const fpRetryRef = useRef<Set<string>>(new Set());

    const paramViewRef = useRef<ParamViewSegment | null>(null);
    useEffect(() => {
        paramViewRef.current = paramView;
    }, [paramView]);

    const fetchDebounceRef = useRef<number | null>(null);
    const fetchReqIdRef = useRef(0);
    const [refreshToken, setRefreshToken] = useState(0);

    const [forceParamFetchToken, setForceParamFetchToken] = useState(0);
    const lastAppliedForceParamFetchTokenRef = useRef(0);

    // Force parameter refresh when the session state changes meaningfully (undo/redo/timeline edits).
    // 鍚屾椂娓呴櫎鏃ф洸绾挎暟鎹紝閬垮厤鏃ф暟鎹湪鏂版暟鎹埌杈惧墠鐭殏鏄剧ず锛堜篃淇鍒濇瀵煎叆鍚庢洸绾夸笉鏄剧ず鐨勯棶棰橈級锟?
    useEffect(() => {
        if (!rootTrackId) return;
        setParamView(null);
        setSecondaryParamViews({});
        setForceParamFetchToken((x) => x + 1);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [paramsEpoch, rootTrackId]);

    // 鐩戝惉 pitch_orig_updated 浜嬩欢锛岃Е鍙戞洸绾垮埛鏂帮拷?
    // 娉ㄦ剰锛氬垎鏋愯繘搴︾姸鎬侊紙started/progress锛夌敱鍏ㄥ眬 PitchAnalysisProvider 缁熶竴绠＄悊锟?
    // 姝ゅ鍙礋璐ｅ湪鍒嗘瀽瀹屾垚鍚庡埛锟?PianoRoll 鏇茬嚎鏁版嵁锟?
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
                        if (
                            payload?.rootTrackId &&
                            payload.rootTrackId !== rootTrackId
                        )
                            return;

                        // 鑻ョ敤鎴锋鍦ㄧ粯鍒舵洸绾匡紙pointer down锛夛紝鎺ㄨ繜鏇茬嚎鍒锋柊锟?pointer-up 鍚庯紝
                        // 閬垮厤鍚庣鍒嗘瀽缁撴灉瑕嗙洊鐢ㄦ埛姝ｅ湪缁樺埗锟?liveEditOverride 鍐呭锟?
                        if (liveEditActiveRef.current) {
                            pendingPitchUpdatedRefreshRef.current = true;
                        } else {
                            setForceParamFetchToken((x) => x + 1);
                            setRefreshToken((x) => x + 1);
                        }
                    },
                );
            } catch {
                // Safe no-op: browser/pywebview builds won't have the Tauri API.
            }
        }

        void setup();

        return () => {
            disposed = true;
            if (unlistenUpdated) unlistenUpdated();
        };
    }, [editParam, pitchEnabled, rootTrackId]);

    useEffect(() => {
        if (editParam !== "pitch") return;
        if (pitchEnabled) return;
        setParamView(null);
        setPitchEditUserModified(null);
        setPitchEditBackendAvailable(null);
    }, [editParam, pitchEnabled]);

    // 锟?editParam 鍒囨崲鏃讹紝娓呴櫎鍓弬鏁扮紦锟?
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

    function computeVisibleRequest() {
        const debug =
            typeof window !== "undefined" &&
            window.localStorage?.getItem("hifishifter.debugPianoRoll") === "1";

        const trackId = rootTrackId;
        if (!trackId) {
            if (debug) {
                // eslint-disable-next-line no-console
                console.debug("[PianoRollData] no rootTrackId; skip fetch");
            }
            return null;
        }

        const { w } = viewSizeRef.current;
        const sl = scrollLeftRef.current;
        const ppb = pxPerBeatRef.current;
        const startBeat = sl / Math.max(1e-9, ppb);
        const durBeats = w / Math.max(1e-9, ppb);
        const startSec = startBeat * secPerBeat;
        const durSec = durBeats * secPerBeat;

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

        // DEBUG: Log data request parameters
        const debugEnabled =
            typeof window !== "undefined" &&
            window.localStorage?.getItem("hifishifter.debugPianoRoll") === "1";

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
                timeToFrame(covParamStartSec + covParamDurSec, fpMs) -
                    startFrame +
                    1,
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
            const secondaryStartFrame = Math.max(
                0,
                timeToFrame(covParamStartSec, secondaryFpMs),
            );
            const secondaryFrameCount = clamp(
                Math.max(
                    1,
                    timeToFrame(
                        covParamStartSec + covParamDurSec,
                        secondaryFpMs,
                    ) -
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

        return {
            debug,
            trackId,
            paramCoversVisible,
            paramKey,
            startFrame,
            frameCount,
            stride,
            fpMs,
            fpKey,
            forceParamFetchToken,
            secondaryRequests,
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

        const forceParam =
            localForceParamFetchToken !==
            lastAppliedForceParamFetchTokenRef.current;
        const shouldFetchParam = !paramCoversVisible || forceParam;

        // 鍓弬鏁板紓姝ュ姞杞斤紙鐙珛璇锋眰锛屼笉褰卞搷涓诲弬鏁板埛鏂伴€昏緫锟?
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
                        if (
                            !secondaryPitchEnabled &&
                            secondaryReq.secondaryParam === "pitch"
                        ) {
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
                            Number(
                                payload.frame_period_ms ??
                                    secondaryReq.secondaryFpMs,
                            ) || secondaryReq.secondaryFpMs;
                        paramFramePeriodCache.set(
                            secondaryReq.secondaryFpKey,
                            fpRes,
                        );
                        return [
                            secondaryReq.secondaryParam,
                            {
                                key: secondaryReq.secondaryParamKey,
                                framePeriodMs: fpRes,
                                startFrame:
                                    Number(
                                        payload.start_frame ??
                                            secondaryReq.secondaryStartFrame,
                                    ) || secondaryReq.secondaryStartFrame,
                                stride,
                                referenceKind:
                                    payload.reference_kind ?? "source_curve",
                                orig: (payload.orig ?? []).map(
                                    (v) => Number(v) || 0,
                                ),
                                edit: (payload.edit ?? []).map(
                                    (v) => Number(v) || 0,
                                ),
                            } as ParamViewSegment,
                        ] as const;
                    }),
                );
                if (secondaryFetchReqIdRef.current !== secReqId) return;
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
                const debugEnabled =
                    typeof window !== "undefined" &&
                    window.localStorage?.getItem(
                        "hifishifter.debugPianoRoll",
                    ) === "1";
                beginLoading();
                try {
                    if (debugEnabled) {
                        console.log(
                            "[usePianoRollData] Fetching param frames:",
                            {
                                trackId,
                                editParam,
                                startFrame,
                                frameCount,
                                stride,
                                startTimeSec: framesToTime(startFrame, fpMs),
                                endTimeSec: framesToTime(
                                    startFrame + frameCount - 1,
                                    fpMs,
                                ),
                            },
                        );
                    }

                    const res = await paramsApi.getParamFrames(
                        trackId,
                        editParam,
                        startFrame,
                        frameCount,
                        stride,
                    );
                    if (fetchReqIdRef.current !== reqId) return;
                    if (!res?.ok) {
                        if (debug) {
                            // eslint-disable-next-line no-console
                            console.debug(
                                "[PianoRollData] paramFrames not ok",
                                {
                                    trackId,
                                    editParam,
                                    paramKey,
                                    startFrame,
                                    frameCount,
                                    stride,
                                    res,
                                },
                            );
                        }
                        return;
                    }

                    const payload = res as ParamFramesPayload;

                    if (editParam === "pitch") {
                        const userModified = payload.pitch_edit_user_modified;
                        setPitchEditUserModified(
                            typeof userModified === "boolean"
                                ? userModified
                                : null,
                        );

                        const backendAvail =
                            payload.pitch_edit_backend_available;
                        setPitchEditBackendAvailable(
                            typeof backendAvail === "boolean"
                                ? backendAvail
                                : null,
                        );
                    }
                    const fpRes =
                        Number(payload.frame_period_ms ?? fpMs) || fpMs;
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
                            receivedStartSec: framesToTime(
                                receivedStartFrame,
                                fpRes,
                            ),
                            receivedEndSec: framesToTime(
                                receivedStartFrame + receivedEditLen - 1,
                                fpRes,
                            ),
                            receivedDurSec: framesToTime(
                                receivedEditLen - 1,
                                fpRes,
                            ),
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
                    lastAppliedForceParamFetchTokenRef.current =
                        localForceParamFetchToken;
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

        const {
            debug,
            trackId,
            paramKey,
            startFrame,
            frameCount,
            stride,
            fpMs,
            fpKey,
        } = req;

        setIsRefreshing(true);
        const reqId = ++fetchReqIdRef.current;
        const shouldFetchParam = !(editParam === "pitch" && !pitchEnabled);
        try {
            beginLoading();
            const [paramRes, secondaryResults] = await Promise.all([
                shouldFetchParam
                    ? paramsApi.getParamFrames(
                          trackId,
                          editParam,
                          startFrame,
                          frameCount,
                          stride,
                      )
                    : Promise.resolve(null),
                Promise.all(
                    req.secondaryRequests.map(async (secondaryReq) => {
                        const secondaryPitchEnabled =
                            secondaryReq.secondaryParam !== "pitch" ||
                            pitchEnabled ||
                            editParam === "pitch";
                        if (
                            !secondaryPitchEnabled &&
                            secondaryReq.secondaryParam === "pitch"
                        ) {
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
                            Number(
                                secPayload.frame_period_ms ??
                                    secondaryReq.secondaryFpMs,
                            ) || secondaryReq.secondaryFpMs;
                        paramFramePeriodCache.set(
                            secondaryReq.secondaryFpKey,
                            secFpRes,
                        );
                        return [
                            secondaryReq.secondaryParam,
                            {
                                key: secondaryReq.secondaryParamKey,
                                framePeriodMs: secFpRes,
                                startFrame:
                                    Number(
                                        secPayload.start_frame ??
                                            secondaryReq.secondaryStartFrame,
                                    ) || secondaryReq.secondaryStartFrame,
                                stride,
                                referenceKind:
                                    secPayload.reference_kind ?? "source_curve",
                                orig: (secPayload.orig ?? []).map(
                                    (v) => Number(v) || 0,
                                ),
                                edit: (secPayload.edit ?? []).map(
                                    (v) => Number(v) || 0,
                                ),
                            } as ParamViewSegment,
                        ] as const;
                    }),
                ),
            ]);

            if (fetchReqIdRef.current !== reqId) return;

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
                    startFrame:
                        Number(payload.start_frame ?? startFrame) || startFrame,
                    stride,
                    referenceKind: payload.reference_kind ?? "source_curve",
                    orig: (payload.orig ?? []).map((v) => Number(v) || 0),
                    edit: (payload.edit ?? []).map((v) => Number(v) || 0),
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
            invalidate();
            return;
        }

        const secReqId = ++secondaryFetchReqIdRef.current;
        beginLoading();
        try {
            const responses = await Promise.all(
                req.secondaryRequests.map(async (secondaryReq) => {
                    const secondaryPitchEnabled =
                        secondaryReq.secondaryParam !== "pitch" ||
                        pitchEnabled ||
                        editParam === "pitch";
                    if (
                        !secondaryPitchEnabled &&
                        secondaryReq.secondaryParam === "pitch"
                    ) {
                        return null;
                    }
                    const res = await paramsApi.getParamFrames(
                        req.trackId,
                        secondaryReq.secondaryParam,
                        secondaryReq.secondaryStartFrame,
                        secondaryReq.secondaryFrameCount,
                        req.stride,
                    );
                    if (!res?.ok) return null;

                    const payload = res as ParamFramesPayload;
                    const fpRes =
                        Number(
                            payload.frame_period_ms ??
                                secondaryReq.secondaryFpMs,
                        ) || secondaryReq.secondaryFpMs;
                    paramFramePeriodCache.set(
                        secondaryReq.secondaryFpKey,
                        fpRes,
                    );
                    return [
                        secondaryReq.secondaryParam,
                        {
                            key: secondaryReq.secondaryParamKey,
                            framePeriodMs: fpRes,
                            startFrame:
                                Number(
                                    payload.start_frame ??
                                        secondaryReq.secondaryStartFrame,
                                ) || secondaryReq.secondaryStartFrame,
                            stride: req.stride,
                            referenceKind:
                                payload.reference_kind ?? "source_curve",
                            orig: (payload.orig ?? []).map(
                                (v) => Number(v) || 0,
                            ),
                            edit: (payload.edit ?? []).map(
                                (v) => Number(v) || 0,
                            ),
                        } as ParamViewSegment,
                    ] as const;
                }),
            );
            if (secondaryFetchReqIdRef.current !== secReqId) return;
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
        scrollLeft,
        pxPerBeat,
        secPerBeat,
        viewWidth,
        refreshToken,
        forceParamFetchToken,
    ]);

    /**
     * 鐢卞閮紙PianoRollPanel锛夊湪 pointer-up 鏃惰皟鐢紝閫氱煡 live 缂栬緫宸茬粨鏉燂拷?
     * 鑻ユ鍓嶆湁琚帹杩熺殑 pitch_orig_updated 鍒锋柊锛屾鏃剁珛鍗宠Е鍙戯拷?
     */
    function notifyLiveEditEnded() {
        if (pendingPitchUpdatedRefreshRef.current) {
            pendingPitchUpdatedRefreshRef.current = false;
            setForceParamFetchToken((x) => x + 1);
            setRefreshToken((x) => x + 1);
        }
    }

    return {
        paramView,
        setParamView,
        secondaryParamViews,
        bumpRefreshToken: () => {
            setRefreshToken((x) => x + 1);
            // Also bump forceParamFetchToken so refreshVisible() bypasses the
            // "paramCoversVisible" cache check and actually re-fetches data
            // from the backend after edit operations.
            setForceParamFetchToken((x) => x + 1);
        },
        refreshNow,
        refreshSecondaryNow,
        notifyLiveEditEnded,
        isRefreshing,
        isLoading,
        pitchEditUserModified,
        pitchEditBackendAvailable,
    };
}
