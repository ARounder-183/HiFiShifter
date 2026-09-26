/**
 * 参数编辑器波形的「响度自动化快照」数据层。
 *
 * ## 为什么需要这个独立数据源
 *
 * 参数编辑器的波形要画成**可听结果**：`源峰值 × clip增益 × volume(t) × dyn增益(t)`。
 * 这条映射与「当前编辑的是哪个参数」无关（编辑音量、动态、音高……波形都该是
 * 同一个可听结果，见 PianoRollPanel 的 amplitudeMap 说明）—— 因此它不能复用
 * `usePianoRollData` 的 `paramView`（那只覆盖当前参数、且随视口窗口化）。
 *
 * 本 hook 为**当前根轨道组**一次性拉取整条工程的：
 * - `volume` 曲线（用户画的音量包络；无数据帧由后端填默认 1.0）；
 * - `dyn` 曲线（用户目标电平；哨兵已在后端出口解析成原声基线）；
 * - `dyn` 的 `orig`（原声电平基线；分析未就绪时为空数组）。
 *
 * 取数用自适应 stride（全工程点数封顶 {@link MAX_SNAPSHOT_FRAMES}），一次请求
 * 体积可控，且**横向滚动 / 缩放不触发重取** —— 波形形变跟随滚动零延迟。
 *
 * ## 刷新时机
 *
 * - `rootTrackId` / 工程帧数 / 帧周期变化；
 * - `paramsEpoch`（任何 timeline 更新都递增 —— set_param_frames / 移动 clip /
 *   撤销重做都会经过这里）与 `refreshToken`（usePianoRollData 的显式刷新令牌，
 *   其 dyn_orig_updated 监听器对**所有参数**生效 —— 原声电平分析完成时基线
 *   从空到有，波形随之更新）。
 *
 * 附带收益：本 hook 取 `dyn` 帧会触发后端的基线组装/调度（get_param_frames 的
 * dyn 分支），因此音量面板打开时动态基线也会在后台就绪。
 *
 * ## 与其他模块的关系
 * - 消费方：`PianoRollPanel` 用快照 + live 覆盖构造 `makeLoudnessAmplitudeMap`；
 * - 后端：`get_param_frames`（binary 模式，API 层统一解码）。
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { paramsApi } from "../../../services/api";
import type { ParamFramesPayload } from "../../../types/api";
import { isDynParam, VOLUME_PARAM_ID } from "./paramRanges";

/** 快照全工程点数上限：40k 点 ≈ 单曲线 160KB（binary 解码后），超出按 stride 降采样。 */
const MAX_SNAPSHOT_FRAMES = 40_000;

/** 一份响度自动化快照（整条工程，起点恒为帧 0）。 */
export interface LoudnessSnapshot {
    /** 快照第 0 帧对应的绝对帧号（恒 0，保留字段以便与 live 窗口同构对齐）。 */
    startFrame: number;
    /** 帧步长（1 = 全分辨率；长工程自动降采样）。 */
    stride: number;
    /** 帧周期（毫秒）。 */
    framePeriodMs: number;
    /** 逐帧音量包络（无数据帧 = 后端默认 1.0）。 */
    volume: number[];
    /** 逐帧动态目标电平（哨兵已解析为基线；分析未就绪时为退化值 1.0）。 */
    dynTarget: number[];
    /** 逐帧原声电平基线；分析未就绪 / 整组静音时为空数组（动态增益恒 1）。 */
    dynBaseline: number[];
    /**
     * true = volume 恒 1 且动态无基线（增益恒 1）→ 幅度映射与线性直投逐像素
     * 等价。面板据此**不挂映射**，让未使用响度自动化的工程保持既有波形路径。
     */
    identity: boolean;
}

/**
 * 判定快照是否为「恒等」的（波形与线性直投逐像素等价）。
 *
 * 【为什么恒等判定只能看"有没有可能产生非 1 增益"】
 * 这是**可能性**判定，不是**当前是否恰巧为 1**的判定。用户尚未在 dyn 面板
 * 画任何一笔时，后端把哨兵解析成基线、`edit` 恒等于 `orig` —— 此时逐帧比较
 * 会得出"增益全是 1"，但用户**正准备画第一笔**：一旦据此判定恒等、不挂映射，
 * 波形就再也收不到 live 覆盖，实时预览直接失效（且要等提交后快照重取才恢复）。
 * 因此只有"根本没有动态数据可用"（基线为空 ⇒ 增益恒为 1，永远不可能变化）
 * 才算恒等 —— 与"用户还没画"是两件不同的事。
 *
 * 同理，volume 恒 1 也算恒等不是：那是"用户把音量包络整体画在 1.0 上"，
 * 下一笔就能把它拖走。
 */
export function isIdentitySnapshot(snapshot: {
    volume: readonly number[];
    dynBaseline: readonly number[];
}): boolean {
    for (let i = 0; i < snapshot.volume.length; i += 1) {
        const v = snapshot.volume[i];
        if (Number.isFinite(v) && Math.abs(v - 1.0) > 1e-4) return false;
    }
    // 基线非空 ⇒ 响度自动化数据可用（用户随时可以往里画）⇒ 必须挂映射；
    // 空基线 = 增益恒 1，动态这一路永远不可能贡献非 1 增益。
    return snapshot.dynBaseline.length === 0;
}

export function snapshotFromPayloads(
    volumePayload: ParamFramesPayload,
    dynPayload: ParamFramesPayload,
    fallbackFp: number,
): LoudnessSnapshot | null {
    const fp = Number(dynPayload.frame_period_ms ?? fallbackFp) || fallbackFp;
    const dynTarget = (dynPayload.edit ?? []).map((v) => (Number.isFinite(v) ? v : 1.0));
    // 基线**必须保持逐帧对齐**：它是逐帧电平，0（静音）是合法且常见的值，
    // 不能用 filter 剔除 —— 剔除会缩短数组，使后续所有帧整体错位（波形把
    // 一段的增益画到另一段上）。只做有限性净化，长度恒等于目标曲线。
    const dynBaseline = (dynPayload.orig ?? []).map((v) => (Number.isFinite(v) && v > 0 ? v : 0));
    // 基线与目标必须等长（同一请求的两条曲线）；长度异常时按"无基线"处理
    // —— 宁可动态增益恒 1，也不允许错位采样。
    if (dynBaseline.length !== dynTarget.length) dynBaseline.length = 0;
    const next: LoudnessSnapshot = {
        startFrame: 0,
        stride: 1,
        framePeriodMs: fp,
        volume: (volumePayload.edit ?? []).map((v) => (Number.isFinite(v) ? v : 1.0)),
        dynTarget,
        dynBaseline,
        identity: false,
    };
    next.identity = isIdentitySnapshot(next);
    return next;
}

export function useLoudnessCurves(args: {
    rootTrackId: string | null;
    /** 工程总帧数（= ceil(projectSec × 1000 / fp)），决定快照长度。 */
    projectFrames: number;
    framePeriodMs: number;
    /** 外部刷新令牌：任何 timeline 更新都递增（与 usePianoRollData 同源）。 */
    paramsEpoch: number;
    /** 显式刷新令牌（usePianoRollData 的 refreshToken；含 dyn_orig_updated 触发）。 */
    refreshToken: number;
    /**
     * 面板的"笔画进行中"标志（与 `usePianoRollData` 同一个 ref）。
     *
     * 【为什么响度快照也需要它】这份快照换引用会让 `pianoRollAmplitudeMap` 换对象，
     * 波形几何随即**全量重建**（`canReuseGeometry` 失败）。而波形逐帧重建只发生在
     * 编辑 volume / dyn 时（`requestWaveformRepaint` 的参数闸门），因此笔画期间让快照
     * 落地，等于和进行中的笔画抢主线程 —— 这正是"只有这两个参数会被打断"的一环。
     */
    liveEditActiveRef?: React.MutableRefObject<boolean>;
}): {
    snapshot: LoudnessSnapshot | null;
    analysisPending: boolean;
    /**
     * 产出当前快照的那次取数的**序号**（单调递增；每次真正发起取数都会 +1）。
     *
     * 【用途】提交参数后，波形要等"提交之后发起"的这次取数返回才能撤下 live
     * 覆盖层（否则会闪回旧波形）。调用方据此判断"手上这份快照是不是提交之后取的"。
     */
    snapshotFetchSeq: number;
    /**
     * 落地笔画期间被推迟的那一份快照（面板在 pointer-up 调用）。
     *
     * 与 `refreshToken` 的补取分工：补取会**重新发起**一次取数（拿到最新数据），
     * 本函数只是把"已经拿回来但被推迟"的那一份落地，保证笔画结束后画面立刻跟上，
     * 不必等下一次 IPC 往返。
     */
    flushPending: () => void;
    /**
     * 当前**已发出**的最大取数序号（同步读取，不触发渲染）。
     *
     * 【为什么要"已发出的最大值"而不只看快照的序号】提交发生时可能有一次
     * **提交之前就已发出**的在飞取数（例如原声基线分析完成触发的刷新），它的
     * 数据里没有本次编辑。把它当成"提交后的快照"会让覆盖层提前退场、波形闪回。
     * 提交侧只要记下这个最大值，后续只在"序号更大"的快照上收尾即可 ——
     * 因为提交那一次取数必然在此之后发出。
     */
    getLatestFetchSeq: () => number;
} {
    const { rootTrackId, projectFrames, framePeriodMs, paramsEpoch, refreshToken } = args;
    const liveEditActiveRef = args.liveEditActiveRef;

    const [snapshot, setSnapshot] = useState<LoudnessSnapshot | null>(null);
    const [analysisPending, setAnalysisPending] = useState(false);
    const [snapshotFetchSeq, setSnapshotFetchSeq] = useState(0);
    const fetchReqIdRef = useRef(0);
    /**
     * 笔画期间被推迟落地的结果（见 `land`）。
     *
     * 只保留**最后一份**：中途的那些已被更新的结果取代，落地它们没有意义。
     */
    const pendingLandingRef = useRef<{
        snapshot: LoudnessSnapshot | null;
        fetchSeq: number;
        analysisPending: boolean;
    } | null>(null);

    const applyLanding = useCallback(
        (landing: {
            snapshot: LoudnessSnapshot | null;
            fetchSeq: number;
            analysisPending: boolean;
        }) => {
            setSnapshot(landing.snapshot);
            setSnapshotFetchSeq(landing.fetchSeq);
            setAnalysisPending(landing.analysisPending);
        },
        [],
    );

    /**
     * 落地一次取数结果；笔画进行中则只记 pending。
     *
     * 【为什么"发起"照常、"落地"推迟】`committedSettleSeqRef` 依赖"已发出的最大取数
     * 序号"单调递增（判定"这份快照是不是提交之后取的"）。推迟**发起**会让水位错位，
     * 把已修好的"松手闪回旧波形"重新引回来。序号在发起时推进，因此只推迟落地是安全的。
     */
    const land = useCallback(
        (landing: {
            snapshot: LoudnessSnapshot | null;
            fetchSeq: number;
            analysisPending: boolean;
        }) => {
            if (liveEditActiveRef?.current) {
                pendingLandingRef.current = landing;
                return;
            }
            applyLanding(landing);
        },
        [applyLanding, liveEditActiveRef],
    );

    /** 面板在 pointer-up 调用：落地笔画期间被推迟的那一份（若有）。 */
    const flushPending = useCallback(() => {
        const pending = pendingLandingRef.current;
        if (pending === null) return;
        pendingLandingRef.current = null;
        applyLanding(pending);
    }, [applyLanding]);
    // 「在飞合并」：取数已发出时，后续触发只标记 dirty，待本次完成后补一次
    // —— 撤销等离散变更**立即**取数（波形与参数线同批刷新，不再有 250ms
    // 防抖滞后），clip 拖拽等连续 timeline 更新仍被收敛成"在飞 + 一次尾随"。
    const inFlightRef = useRef(false);
    const dirtyRef = useRef(false);
    // 最新输入（尾随补取时读取，保证用的是最新参数而非触发时的旧值）。
    const inputsRef = useRef<{
        rootTrackId: string | null;
        projectFrames: number;
        framePeriodMs: number;
    }>({ rootTrackId: null, projectFrames: 0, framePeriodMs: 5 });

    inputsRef.current = { rootTrackId, projectFrames, framePeriodMs };

    const scheduleFetch = () => {
        if (inFlightRef.current) {
            dirtyRef.current = true;
            return;
        }
        inFlightRef.current = true;
        const reqId = ++fetchReqIdRef.current;
        void (async () => {
            try {
                const {
                    rootTrackId: trackId,
                    projectFrames: frames,
                    framePeriodMs: fp,
                } = inputsRef.current;
                if (!trackId || !(frames > 0)) return;
                const stride = Math.max(1, Math.ceil(frames / MAX_SNAPSHOT_FRAMES));
                const [volumeRes, dynRes] = await Promise.all([
                    paramsApi.getParamFrames(trackId, "volume", 0, frames, stride),
                    paramsApi.getParamFrames(trackId, "dyn", 0, frames, stride),
                ]);
                if (fetchReqIdRef.current !== reqId) return;
                if (!volumeRes?.ok || !dynRes?.ok) {
                    // 失败也走 land：笔画期间置空快照会让波形映射消失、笔画闪断，
                    // 宁可让旧快照多留一会儿。
                    land({ snapshot: null, fetchSeq: reqId, analysisPending: false });
                    return;
                }
                const next = snapshotFromPayloads(
                    volumeRes as ParamFramesPayload,
                    dynRes as ParamFramesPayload,
                    fp,
                );
                if (next) {
                    next.stride = stride;
                    land({
                        snapshot: next,
                        fetchSeq: reqId,
                        analysisPending: (dynRes as ParamFramesPayload).analysis_pending === true,
                    });
                }
            } catch {
                // ignore：保留上一次快照，等待下一次触发
            } finally {
                inFlightRef.current = false;
                if (dirtyRef.current) {
                    dirtyRef.current = false;
                    scheduleFetch();
                }
            }
        })();
    };

    useEffect(() => {
        if (!rootTrackId || !(projectFrames > 0) || !(framePeriodMs > 0)) {
            const reqId = (fetchReqIdRef.current += 1);
            setSnapshot(null);
            setSnapshotFetchSeq(reqId);
            setAnalysisPending(false);
            return;
        }
        scheduleFetch();

        // 输入变化时 effect 会重跑；已挂载的取消标记由请求序号守卫兜底
        // （reqId 不匹配的响应直接丢弃），无需显式取消。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rootTrackId, projectFrames, framePeriodMs, paramsEpoch, refreshToken]);

    /**
     * 同步读取"已发出的最大取数序号"（见返回类型说明）。
     *
     * 用函数而非渲染期快照：提交包装层需要在**调用后端之前**立刻取值，
     * 而那一刻渲染还没发生。
     */
    const getLatestFetchSeq = useCallback(() => fetchReqIdRef.current, []);

    return { snapshot, analysisPending, snapshotFetchSeq, getLatestFetchSeq, flushPending };
}

/**
 * live 覆盖 key 的解析结果。
 *
 * key 形如 `v2|{trackId}|{param}|{startFrame}|{frameCount}|{stride}`；`live.edit`
 * 与发起编辑时的 paramView 窗口对齐，因此 startFrame / stride 必须从 key 里取
 * 出来才能把 live 覆盖与快照曲线对齐采样。
 */
export interface LiveOverrideKeyParts {
    /** 参数 id（key 第 2 段）。 */
    paramId: string;
    /** live 窗口首帧（key 第 3 段）。 */
    startFrame: number;
    /** live 窗口帧步长（key 第 5 段）。 */
    stride: number;
}

/**
 * 解析 live 覆盖的 key。
 *
 * 【为什么要单独导出】调用方（PianoRollPanel 的幅度映射）在几何重建的热路径上
 * 反复取用 live 覆盖 —— 一次重建可达数万次调用。解析必须能**按覆盖对象缓存**
 * （对象身份不变即 key 不变），而缓存的前提是解析本身是一个纯函数。
 */
export function parseLiveOverrideKey(liveKey: string): LiveOverrideKeyParts {
    const parts = liveKey.split("|");
    // [v2, trackId, paramId, startFrame, frameCount, stride]
    return {
        paramId: parts[2] ?? "",
        startFrame: Number(parts[3]) || 0,
        stride: Number(parts[5]) || 1,
    };
}

/**
 * 判定**已解析**的参数 id 是否属于指定参数（volume / dyn）。
 *
 * 【为什么要求传已解析的 id】`liveOverrideMatchesParam` 每次调用都要 split
 * （见其说明），而本函数是热路径可用的无分配变体；参数归属规则（volume 字面量、
 * dyn 的历史别名）仍收口在这里，不会因调用点不同而分叉。
 */
export function liveOverrideParamMatches(paramId: string, param: "volume" | "dyn"): boolean {
    if (param === "volume") return paramId === VOLUME_PARAM_ID;
    return isDynParam(paramId);
}

/**
 * 判断 live 覆盖是否属于指定参数（`live.key` 形如
 * `v2|{trackId}|{param}|{startFrame}|{frameCount}|{stride}`）。
 *
 * 特殊说明：本函数内部会 `split` key 一次。几何重建的热路径请改用
 * {@link parseLiveOverrideKey} + {@link liveOverrideParamMatches} 的缓存形式。
 */
export function liveOverrideMatchesParam(liveKey: string, param: "volume" | "dyn"): boolean {
    return liveOverrideParamMatches(parseLiveOverrideKey(liveKey).paramId, param);
}
