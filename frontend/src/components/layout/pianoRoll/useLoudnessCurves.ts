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

import { useEffect, useRef, useState } from "react";

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

function isIdentitySnapshot(snapshot: { volume: number[]; dynBaseline: number[] }): boolean {
    for (let i = 0; i < snapshot.volume.length; i += 1) {
        const v = snapshot.volume[i];
        if (Number.isFinite(v) && Math.abs(v - 1.0) > 1e-4) return false;
    }
    // 基线非空即可能产生非 1 增益（用户画过目标电平的帧）；空基线 = 增益恒 1。
    return snapshot.dynBaseline.length === 0;
}

function snapshotFromPayloads(
    volumePayload: ParamFramesPayload,
    dynPayload: ParamFramesPayload,
    fallbackFp: number,
): LoudnessSnapshot | null {
    const fp = Number(dynPayload.frame_period_ms ?? fallbackFp) || fallbackFp;
    const dynTarget = (dynPayload.edit ?? []).map((v) => (Number.isFinite(v) ? v : 1.0));
    const dynBaseline = (dynPayload.orig ?? []).filter((v) => Number.isFinite(v) && v > 0);
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
}): { snapshot: LoudnessSnapshot | null; analysisPending: boolean } {
    const { rootTrackId, projectFrames, framePeriodMs, paramsEpoch, refreshToken } = args;

    const [snapshot, setSnapshot] = useState<LoudnessSnapshot | null>(null);
    const [analysisPending, setAnalysisPending] = useState(false);
    const fetchReqIdRef = useRef(0);
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
                    setSnapshot(null);
                    return;
                }
                const next = snapshotFromPayloads(
                    volumeRes as ParamFramesPayload,
                    dynRes as ParamFramesPayload,
                    fp,
                );
                if (next) {
                    next.stride = stride;
                    setSnapshot(next);
                    setAnalysisPending((dynRes as ParamFramesPayload).analysis_pending === true);
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
            fetchReqIdRef.current += 1;
            setSnapshot(null);
            setAnalysisPending(false);
            return;
        }
        scheduleFetch();

        // 输入变化时 effect 会重跑；已挂载的取消标记由请求序号守卫兜底
        // （reqId 不匹配的响应直接丢弃），无需显式取消。
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [rootTrackId, projectFrames, framePeriodMs, paramsEpoch, refreshToken]);

    return { snapshot, analysisPending };

    return { snapshot, analysisPending };
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
