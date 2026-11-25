import { useCallback, useEffect, useRef } from "react";

import { paramsApi } from "../../../services/api";
import { frontendLog } from "../../../services/frontendLog";

import { paramWindowScope } from "./paramWindowKey";
import { clampParamWriteValue } from "./paramRanges";
import { createLiveEditOverride, liveEditCaughtUp } from "./strokeBuffer";
import type { LiveEditOverride, StrokeBuffer } from "./strokeBuffer";
import type { ParamName, ParamViewSegment, StrokeMode, StrokePoint } from "./types";

// 覆盖层类型住在 strokeBuffer（纯逻辑、可单测）；这里转出一次，调用方继续从本模块取。
export type { LiveEditOverride } from "./strokeBuffer";

/**
 * 单次 IPC 允许的最大帧数。
 *
 * 【为什么必须分段】一次 `set_param_frames` 会把 values / present 两份数组整体序列化过
 * IPC 边界。一笔画过整首歌（5 ms 帧周期、3 分钟 ≈ 3.6 万帧）尚可，但用户把视口缩到最小
 * 再画满屏时，绝对帧区间可以是几十万帧 —— 一次性发出去会构造出几十 MB 的 JSON。
 * 65536 是"一次往返够快、序列化又不失控"的经验分界；分段之间用 `checkpoint: false`
 * 合并成**一次撤销**，所以用户按一次 Ctrl+Z 仍然是回到画之前。
 */
export const MAX_FRAMES_PER_WRITE = 65536;

/**
 * 绘制中的参数曲线**覆盖层**（live override）。
 *
 * 【为什么不进 React 状态】它在指针频率上更新，走 state 会让整块面板以指针频率重渲染。
 * 波形面与曲线层都以 ref + 版本号的方式消费它。
 *
 * 【为什么坐标是绝对帧】见 `strokeBuffer.ts` 的模块头：窗口是后端驱动的，把它当坐标系
 * 就必须在笔画期间冻结取数（`liveEditDeferral`）并把边缘滚屏夹在窗口末端 —— 两处补丁
 * 合起来正是"绘制工具拖到水平边缘自动滚屏，滚过一定距离后参数线不再更新"。
 */

/** 把笔画点按时间序去重（相邻同帧只留最后一个值），并规整为整数帧。 */
export function normalizeStrokePoints(points: readonly StrokePoint[]): StrokePoint[] {
    const uniq: StrokePoint[] = [];
    for (const point of points) {
        if (!Number.isFinite(point.frame) || !Number.isFinite(point.value)) continue;
        const frame = Math.max(0, Math.floor(point.frame));
        const last = uniq[uniq.length - 1];
        if (last !== undefined && last.frame === frame) last.value = point.value;
        else uniq.push({ frame, value: point.value });
    }
    return uniq;
}

/**
 * 用点序列回放出一段稠密曲线写进缓冲。
 *
 * 【什么时候用】只提交、不预览的路径（某些工具在 pointer-up 才第一次拿到整条点序列）。
 * 插值规则与旧实现逐值相同：以首点的值铺底，再按相邻点线性插值。
 */
export function replayStrokePoints(
    buffer: StrokeBuffer,
    points: readonly StrokePoint[],
    clamp?: (value: number) => number,
): void {
    if (points.length === 0) return;
    let minF = Number.POSITIVE_INFINITY;
    let maxF = 0;
    for (const point of points) {
        minF = Math.min(minF, point.frame);
        maxF = Math.max(maxF, point.frame);
    }
    const dense = new Array<number>(maxF - minF + 1).fill(points[0].value);
    for (let index = 0; index < points.length - 1; index += 1) {
        const a = points[index];
        const b = points[index + 1];
        const lo = Math.min(a.frame, b.frame);
        const hi = Math.max(a.frame, b.frame);
        const denom = b.frame - a.frame;
        for (let frame = lo; frame <= hi; frame += 1) {
            if (frame < minF || frame > maxF) continue;
            const t = denom === 0 ? 1 : (frame - a.frame) / denom;
            dense[frame - minF] = a.value + (b.value - a.value) * t;
        }
    }
    buffer.write(minF, dense, { clamp });
}

export function useLiveParamEditing(args: {
    rootTrackId: string | null;
    editParam: ParamName;
    pitchEnabled: boolean;

    paramView: ParamViewSegment | null;

    bumpRefreshToken: () => void;
    invalidate: () => void;
}) {
    const { rootTrackId, editParam, pitchEnabled, paramView, bumpRefreshToken, invalidate } = args;

    const liveEditOverrideRef = useRef<LiveEditOverride | null>(null);
    /**
     * 当前参数窗口（**最新**的一份，供提交回写与投影使用）。
     *
     * 【为什么不能用 `paramView` 闭包】`commitStroke` 捕获的是 pointerdown 那一刻的
     * `paramView`；投影必须按**当前**窗口做，否则会把按旧窗口算出的数组当成新窗口的值。
     */
    const paramViewRef = useRef<ParamViewSegment | null>(paramView);
    useEffect(() => {
        paramViewRef.current = paramView;
    }, [paramView]);
    /**
     * live 覆盖的**全局单调**版本号（跨手势、跨轨道都不复用）。
     *
     * 用全局计数而非"每个覆盖自增"：这样 `null → 覆盖` 与 `覆盖 → 覆盖` 的过渡都必然
     * 改变主画布签名的数值项，不必为"无覆盖"再约定一个哨兵值。
     */
    const liveEditVersionRef = useRef(0);
    /**
     * 自上次 `resetLiveEditPreview` 以来写过的**绝对帧**区间并集（开区间）。
     *
     * 直线 / 颤音工具的预览每帧都要"先擦掉上一帧、再画本帧"，擦除范围必须覆盖上一帧
     * 写过的全部帧（端点会来回移动）。并集即可：每帧一次擦除 + 一次写入，并集恰好等于
     * 上一帧的写入范围。旧实现记的是**窗口下标**，换窗口即失效；绝对帧不存在这个问题。
     */
    const liveEditWrittenRangeRef = useRef<{ scope: string; start: number; end: number } | null>(
        null,
    );
    /** 单笔超长只提示一次（见 MAX_STROKE_FRAMES）。 */
    const strokeTruncationReportedRef = useRef(false);

    /** 推进覆盖层的全局版本号（缓冲原地更新后必须显式通知画布签名）。 */
    const touchLiveEdit = useCallback((live: LiveEditOverride) => {
        liveEditVersionRef.current += 1;
        live.version = liveEditVersionRef.current;
    }, []);

    /** 丢弃 live 覆盖（连同其写过的区间记录）。 */
    const clearLiveEditOverride = useCallback(() => {
        liveEditOverrideRef.current = null;
        liveEditWrittenRangeRef.current = null;
    }, []);

    /**
     * 把当前覆盖标记为"已提交"——**保留**其值，只等快照追上（见
     * `LiveEditOverride.committed`）。提交成功路径用它替代 `clearLiveEditOverride`，
     * 以消除"松手闪回旧波形"。
     */
    const markLiveEditCommitted = useCallback(() => {
        const cur = liveEditOverrideRef.current;
        if (cur === null) return;
        cur.committed = true;
        // 已不是拖动中：区间回滚的记账随之作废。
        liveEditWrittenRangeRef.current = null;
    }, []);

    /** 快照刷新到位后撤下"已提交"的覆盖层（拖动中的覆盖层不受影响）。 */
    const clearCommittedLiveEditOverride = useCallback(() => {
        const cur = liveEditOverrideRef.current;
        if (cur !== null && cur.committed === true) clearLiveEditOverride();
    }, [clearLiveEditOverride]);

    /**
     * 参数窗口真正带上提交值后才撤 live 覆盖。
     *
     * 响度快照和曲线快照是两路独立 IPC；不能因为响度先回来就把曲线覆盖撤掉。
     */
    const clearCommittedLiveEditOverrideIfCaughtUp = useCallback(
        (pv: ParamViewSegment): boolean => {
            const cur = liveEditOverrideRef.current;
            if (cur === null || cur.committed !== true) return false;
            if (cur.scope !== paramWindowScope(pv.key)) {
                // 换了参数 / 轨道：这份占位已经与画面无关。
                clearLiveEditOverride();
                return true;
            }
            if (!liveEditCaughtUp(cur, pv)) return false;
            clearLiveEditOverride();
            return true;
        },
        [clearLiveEditOverride],
    );

    /**
     * 保证覆盖层属于 `pv` 的作用域，且处于"正在画"的状态。
     *
     * 【为什么不再重锚】缓冲是绝对帧的，换窗口天然无损 —— 旧实现那套
     * `reanchorLiveEditWindow` 正是"坐标是窗口下标"逼出来的补偿。
     *
     * 【为什么"已提交"要换新缓冲】上一笔的值已经交给后端，它作为占位只负责撑到快照追上；
     * 用户又落笔时继续往同一份缓冲里追加，会让上一笔的值被当成这一笔的一部分提交。
     */
    const ensureLiveEditBase = useCallback((pv: ParamViewSegment) => {
        const scope = paramWindowScope(pv.key);
        const cur = liveEditOverrideRef.current;
        if (cur !== null && cur.scope === scope && cur.committed !== true) return;
        liveEditVersionRef.current += 1;
        liveEditOverrideRef.current = createLiveEditOverride(scope, liveEditVersionRef.current);
        liveEditWrittenRangeRef.current = null;
    }, []);

    /**
     * 窗口变化时的处置：作用域变了才丢弃笔画。
     *
     * 【为什么换窗口什么都不做】覆盖层按绝对帧投影，换窗口只是换了个观察角度。这正是
     * 旧实现做不到、只能靠 `liveEditDeferral` 冻结取数来回避的那件事。
     */
    useEffect(() => {
        if (!paramView) {
            clearLiveEditOverride();
            return;
        }
        const cur = liveEditOverrideRef.current;
        if (cur === null) return;
        if (cur.scope === paramWindowScope(paramView.key)) return;
        clearLiveEditOverride();
    }, [paramView, clearLiveEditOverride]);

    /**
     * live 覆盖写入的值域钳制（与后端写入入口同构，见
     * `paramRanges::clampParamWriteValue`）。
     *
     * 【为什么在这里钳】这是"用户输入 → live 覆盖"的唯一写入点：拖拽预览画出的
     * 曲线必须与提交后后端存下的值逐值一致，否则参数线"停在顶端"而波形按超出值
     * 放大，松手后又跳回来。钳在此处即可覆盖全部手势（手绘 / 直线 / 颤音 /
     * 选区拖拽 / morph 预览），且成本只有受影响帧数那么多次。
     */
    const clampLiveValue = useCallback(
        (value: number) => clampParamWriteValue(editParam, value),
        [editParam],
    );

    const applyDenseToLiveEdit = useCallback(
        (
            pv: ParamViewSegment,
            denseStartFrame: number,
            dense: number[] | null,
            minF: number,
            maxF: number,
            mode: StrokeMode,
        ) => {
            ensureLiveEditBase(pv);
            const cur = liveEditOverrideRef.current;
            if (cur === null) return;
            // 重新进入"编辑中"：提交值可能还没进快照，但用户已经又画了，
            // 此时覆盖层必须跟着手走（不能再被快照到达时的清理规则撤下）。
            cur.committed = false;
            if (mode === "restore" || dense === null) {
                cur.buffer.erase(minF, maxF + 1);
            } else {
                // 写入的坐标是**绝对帧**；`minF` / `maxF` 之外的值即使出现在 dense 里也不写
                // （手势按窗口算 dense 时可能带上邻域的填充值）。
                cur.buffer.write(denseStartFrame, dense, {
                    clamp: clampLiveValue,
                    from: minF,
                    to: maxF,
                });
            }
            if (cur.buffer.truncated && !strokeTruncationReportedRef.current) {
                strokeTruncationReportedRef.current = true;
                frontendLog(
                    "warn",
                    "[liveEdit] stroke exceeded the frame budget and was truncated:",
                    cur.buffer.frameCount,
                );
            }
            touchLiveEdit(cur);
            // 记下写过的绝对帧区间**并集**（供"擦掉上一帧预览"精确回滚，见
            // resetLiveEditPreview）：一次擦除 + 一次写入为一帧，故并集恰好等于
            // 本帧写过的范围。空区间不记。
            if (maxF >= minF) {
                const written = liveEditWrittenRangeRef.current;
                if (written !== null && written.scope === cur.scope) {
                    if (minF < written.start) written.start = minF;
                    if (maxF + 1 > written.end) written.end = maxF + 1;
                } else {
                    liveEditWrittenRangeRef.current = {
                        scope: cur.scope,
                        start: minF,
                        end: maxF + 1,
                    };
                }
            }
        },
        [clampLiveValue, ensureLiveEditBase, touchLiveEdit],
    );

    /**
     * 擦除上一帧的 live 预览（这些帧回到**已提交曲线**）。
     *
     * 【为什么需要】直线 / 颤音工具的预览是"每帧按当前端点重算整段"：第 N 帧写入
     * `[起点, 端点_N]`，第 N+1 帧必须先把第 N 帧写过的点擦掉，否则端点往回拖时旧的
     * 预览会残留在图上。旧实现是逐下标还原成 `pv.edit`；缓冲里"擦除"就是这个语义
     * （投影时未覆盖的帧取基准值），而且不需要知道当前窗口在哪。
     *
     * @param pv 当前参数窗口（只用来校验作用域）。
     * @param opts.keepWrittenRange 保留"写过的区间并集"不清零。用于选区平移这类
     *   窗口逐帧移动的手势：第 N 帧还原时必须把**之前所有帧**写过的区间一起擦掉
     *   （窗口移动后，上一帧的并集盖不住更早的写入），并集在整个手势内持续累积。
     */
    const resetLiveEditPreview = useCallback(
        (pv: ParamViewSegment, opts?: { keepWrittenRange?: boolean }) => {
            const cur = liveEditOverrideRef.current;
            if (cur === null || cur.scope !== paramWindowScope(pv.key)) return;
            const range = liveEditWrittenRangeRef.current;
            if (range !== null && range.scope === cur.scope) {
                if (cur.buffer.erase(range.start, range.end)) touchLiveEdit(cur);
            }
            if (opts?.keepWrittenRange !== true) {
                liveEditWrittenRangeRef.current = null;
            }
        },
        [touchLiveEdit],
    );

    const commitStroke = useCallback(
        async (points: StrokePoint[], mode: StrokeMode) => {
            const trackId = rootTrackId;
            if (!trackId) return;
            if (points.length < 1) return;
            if (editParam === "pitch" && !pitchEnabled) return;

            // IMPORTANT:
            // - During pointer-move, we update the live preview by applying each segment
            //   in the *time order* of the stroke (later segments overwrite earlier ones).
            // - If we sort points by frame here, that overwrite order can change when the
            //   user slightly backtracks in X, causing the committed curve to differ from
            //   what was previewed (often perceived as "spikes" / "glitches").
            // So: keep stroke order, only de-dupe consecutive same-frame samples.
            const uniq = normalizeStrokePoints(points);
            if (uniq.length < 1) return;

            let minF = Number.POSITIVE_INFINITY;
            let maxF = 0;
            for (const point of uniq) {
                minF = Math.min(minF, point.frame);
                maxF = Math.max(maxF, point.frame);
            }
            minF = Math.max(0, Math.floor(minF));
            maxF = Math.max(minF, Math.floor(maxF));

            const pv = paramView;
            const scope = pv === null ? null : paramWindowScope(pv.key);
            const live = liveEditOverrideRef.current;

            // 提交是 fire-and-forget 调用（调用方不 await），任何 IPC 失败都必须
            // 在此兜底：否则 unhandledrejection，且 live 预览层会冻结在半提交状态。
            try {
                if (mode === "restore") {
                    if (live !== null && scope !== null && live.scope === scope) {
                        if (live.buffer.erase(minF, maxF + 1)) touchLiveEdit(live);
                    }
                    invalidate();
                    await paramsApi.restoreParamFrames(
                        trackId,
                        editParam,
                        minF,
                        maxF - minF + 1,
                        true,
                    );
                    // 提交成功：保留覆盖层，等快照带上还原后的值再撤下
                    //（见 markLiveEditCommitted）。失败路径由下方 catch 清掉。
                    markLiveEditCommitted();
                    bumpRefreshToken();
                    return;
                }

                // 【为什么提交的数据源是缓冲而不是点序列】缓冲就是**用户看到的曲线**
                // （手势在 pointer-move 里逐帧写进去的）。按点序列重新插值会与预览产生
                // 细微差异 —— 用户报告为"松手后曲线跳一下"。
                let source: StrokeBuffer | null =
                    live !== null && scope !== null && live.scope === scope ? live.buffer : null;
                if (source === null || source.frameCount === 0) {
                    // 只提交、不预览的路径：用点序列回放出一份等价缓冲，走同一条提交路径。
                    const replayed = createLiveEditOverride(scope ?? "", 0).buffer;
                    replayStrokePoints(replayed, uniq, clampLiveValue);
                    source = replayed;
                }
                // 分段提交之前先取快照：段间有 await，期间用户可能已经开始下一笔，
                // 那时缓冲会继续增长，不该被算进这一笔。
                const segments = source.segments(MAX_FRAMES_PER_WRITE);
                const sourceVersion = source.version;
                if (segments.length === 0) {
                    bumpRefreshToken();
                    return;
                }
                for (let index = 0; index < segments.length; index += 1) {
                    const segment = segments[index];
                    // 这一笔完整覆盖该段的每一帧：present 全 true。显式位图让后端
                    // 不需要从显示数组差分猜测用户改了什么。
                    const present = new Array<boolean>(segment.values.length).fill(true);
                    await paramsApi.setParamFrames(
                        trackId,
                        editParam,
                        segment.startFrame,
                        segment.values,
                        // 只有第一段开一个撤销检查点：一次 Ctrl+Z 回到画之前，
                        // 而不是按段回退（用户报告过"撤销一次只退一段"）。
                        index === 0,
                        present,
                    );
                }
                // 同上：不立刻撤下覆盖层，避免波形先退回旧快照再跳到新快照。
                // 只有缓冲没被下一笔动过，才把它标记为"已提交" —— 否则会把正在画的
                // 那一笔一起冻结成占位。
                const current = liveEditOverrideRef.current;
                if (
                    current !== null &&
                    current.buffer === source &&
                    current.buffer.version === sourceVersion
                ) {
                    markLiveEditCommitted();
                }
                invalidate();
                bumpRefreshToken();
            } catch (err) {
                // 失败时复位 live 预览层，避免覆盖层冻结；状态由下一次
                // refresh（paramsEpoch/滚动等）对账恢复。
                clearLiveEditOverride();
                invalidate();
                // 【为什么要立刻再要一份】后端拒绝这次写入时，屏幕上还留着用户刚画的那一笔。
                // 等下一次"自然"刷新（滚动 / paramsEpoch）意味着用户会先看到一条**并不存在**
                // 的线，随后它又凭空消失 —— 那正是"画完就回退"的观感。这里主动要一次对账，
                // 让屏幕在下一帧就回到后端的真实状态。
                bumpRefreshToken();
                frontendLog("error", "[commitStroke] failed:", err);
            }
        },
        [
            rootTrackId,
            editParam,
            pitchEnabled,
            paramView,
            clampLiveValue,
            invalidate,
            bumpRefreshToken,
            clearLiveEditOverride,
            markLiveEditCommitted,
            touchLiveEdit,
        ],
    );

    return {
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        clearLiveEditOverride,
        clearCommittedLiveEditOverride,
        clearCommittedLiveEditOverrideIfCaughtUp,
        markLiveEditCommitted,
        commitStroke,
    };
}
