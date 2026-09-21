import { useCallback, useEffect, useRef } from "react";

import { paramsApi } from "../../../services/api";

import { clampParamWriteValue } from "./paramRanges";

import { restoreLiveEditRange, writeDenseIntoLiveWindow } from "./liveEditWindow";
import type { ParamName, ParamViewSegment, StrokeMode, StrokePoint } from "./types";

/**
 * 绘制中的参数曲线**覆盖层**（live override）。
 *
 * 【为什么不进 React 状态】它在指针频率上更新，走 state 会让整块面板以指针
 * 频率重渲染。波形面与曲线层都以 ref + 修订号的方式消费它。
 *
 * 【为什么原地更新 + 全局版本号】
 * - `edit` 由 {@link
 *   useLiveParamEditing} 的 `ensureLiveEditBase` 造出一份**私有副本**（不指向
 *   React state），因此可以原地改写。此前每次采样都 `slice()` 出一份新数组 ——
 *   典型窗口 ~6400 个元素、极端缩放下上限 200000（`usePianoRollData` 的
 *   `viewFrames` 钳制）⇒ 单次 1.6MB，高刷笔一帧数个采样就是数 MB 的年轻代垃圾。
 * - 但"引用变了"此前同时承担了"内容变了"的通知职责。改为原地更新后必须显式
 *   给出这一信号，否则**主画布的内容签名会误判为未变**、绘制中的曲线不再刷新
 *  （见 `mainCanvasSignature.ts` 的约束说明）。故每次写入都取一个**全局单调
 *   递增**的版本号：`version` 单调且永不复用，任何一次内容变化都必然改变签名。
 */
export interface LiveEditOverride {
    /** 窗口键（`v2|track|param|start|count|stride`）。 */
    key: string;
    /** 造出本覆盖所用的已提交曲线（`paramView.edit` 的引用，用于检测基准变更）。 */
    base: number[];
    /** 与键中窗口对齐的曲线值（手势内原地更新）。 */
    edit: number[];
    /** 内容版本：每次写入自增（全局单调，不复用）。 */
    version: number;
    /**
     * `true` = 本次编辑**已提交**，覆盖层只是"等整工程快照追上"的占位。
     *
     * 【为什么需要】提交成功后若立刻撤下覆盖层，波形的幅度因子就退回**尚未刷新**
     * 的旧快照 —— 波形先跳回旧形状、等快照到达再跳到新形状（用户报告的"松手闪
     * 一下"）。保留覆盖层直到快照带着本次提交值到达，撤下时逐像素无损。
     *
     * 【为什么不会遮挡后续变化】撤下时机绑定在**快照引用变化**上（见
     * `PianoRollPanel` 的 effect），因此撤销 / 切轨等任何来源的快照更新都会让
     * 覆盖层退场，不会出现"旧覆盖层盖住撤销结果"。
     */
    committed?: boolean;
}

export function useLiveParamEditing(args: {
    rootTrackId: string | null;
    editParam: ParamName;
    pitchEnabled: boolean;

    paramView: ParamViewSegment | null;
    setParamView: (next: ParamViewSegment | null) => void;

    bumpRefreshToken: () => void;
    invalidate: () => void;
}) {
    const {
        rootTrackId,
        editParam,
        pitchEnabled,
        paramView,
        setParamView,
        bumpRefreshToken,
        invalidate,
    } = args;

    const liveEditOverrideRef = useRef<LiveEditOverride | null>(null);
    /**
     * live 覆盖的**全局单调**版本号（跨手势、跨轨道都不复用）。
     *
     * 用全局计数而非"每个覆盖自增"：这样 `null → 覆盖` 与 `覆盖 → 覆盖` 的
     * 过渡都必然改变主画布签名的数值项，不必为"无覆盖"再约定一个哨兵值。
     */
    const liveEditVersionRef = useRef(0);
    /**
     * 自上次 `resetLiveEditPreview` 以来写过的**索引区间并集**（当前覆盖的 key）。
     *
     * 直线 / 颤音工具的预览每帧都要"先擦掉上一帧、再画本帧"，擦除范围必须覆盖
     * 上一帧写过的全部下标（端点会来回移动）。记录并集即可：每帧一次擦除 +
     * 一次写入，并集恰好等于上一帧的写入范围。
     */
    const liveEditWrittenRangeRef = useRef<{ key: string; lo: number; hi: number } | null>(null);

    /** 丢弃 live 覆盖（连同其写过的区间记录）。 */
    const clearLiveEditOverride = useCallback(() => {
        liveEditOverrideRef.current = null;
        liveEditWrittenRangeRef.current = null;
    }, []);

    /**
     * 把当前覆盖标记为"已提交"——**保留**其值，只等快照追上（见
     * `LiveEditOverride.committed` 的说明）。提交成功路径用它替代
     * `clearLiveEditOverride`，以消除"松手闪回旧波形"。
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

    useEffect(() => {
        if (!paramView) {
            clearLiveEditOverride();
            return;
        }
        if (liveEditOverrideRef.current?.key !== paramView.key) {
            clearLiveEditOverride();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps -- 仅按 key 变化重置 live 覆盖；paramView 对象引用随编辑更新而变，全量依赖会无效重跑
    }, [paramView?.key]);

    const ensureLiveEditBase = useCallback((pv: ParamViewSegment) => {
        const cur = liveEditOverrideRef.current;
        // 键相同 = 同一个窗口；再确认底层已提交曲线的**引用**未换（`pv.edit` 变化
        // 说明期间发生过提交/刷新，必须重新取基准，否则预览会建立在一份过期
        // 数据上）。引用在拖动期稳定，故稳态下不触发拷贝。
        if (cur && cur.key === pv.key && cur.base === pv.edit) return;
        // 新窗口 / 基准换了：整份拷贝**仅此一次**，之后一路原地改写。
        liveEditVersionRef.current += 1;
        liveEditOverrideRef.current = {
            key: pv.key,
            base: pv.edit,
            edit: pv.edit.slice(),
            version: liveEditVersionRef.current,
        };
        liveEditWrittenRangeRef.current = null;
    }, []);

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
            if (!cur || cur.key !== pv.key) return;
            // 重新进入"编辑中"：提交值可能还没进快照，但用户已经又画了，
            // 此时覆盖层必须跟着手走（不能再被快照到达时的清理规则撤下）。
            cur.committed = false;
            // 写入的遍历与下标反解收口在 `liveEditWindow`（纯函数、可单测）；
            // 见该模块的文件头说明（为什么从 O(窗口) 降到 O(受影响)）。
            const range = writeDenseIntoLiveWindow({
                edit: cur.edit,
                orig: pv.orig,
                startFrame: pv.startFrame,
                stride: pv.stride,
                dense,
                denseStartFrame,
                minF,
                maxF,
                mode,
                clampValue: clampLiveValue,
            });
            // 原地写入 ⇒ 引用不变，必须显式推进版本号（波形面的几何缓存与主画布
            // 的内容签名都依赖它）。
            liveEditVersionRef.current += 1;
            cur.version = liveEditVersionRef.current;
            // 记下写过的下标区间**并集**（供"擦掉上一帧预览"精确回滚，见
            // resetLiveEditPreview）：一次擦除 + 一次写入为一帧，故并集恰好等于
            // 本帧写过的范围。空区间不记（窗口与受影响帧不相交）。
            if (range !== null && range.lo <= range.hi) {
                const written = liveEditWrittenRangeRef.current;
                if (written !== null && written.key === pv.key) {
                    if (range.lo < written.lo) written.lo = range.lo;
                    if (range.hi > written.hi) written.hi = range.hi;
                } else {
                    liveEditWrittenRangeRef.current = {
                        key: pv.key,
                        lo: range.lo,
                        hi: range.hi,
                    };
                }
            }
        },
        [clampLiveValue, ensureLiveEditBase],
    );

    /**
     * 擦除上一帧的 live 预览（回到**已提交曲线** `pv.edit` 的状态）。
     *
     * 【为什么需要】直线 / 颤音工具的预览是"每帧按当前端点重算整段"：第 N 帧
     * 写入 `[起点, 端点_N]`，第 N+1 帧必须先把第 N 帧写过的点擦掉，否则端点往
     * 回拖时旧的预览会残留在图上。旧实现是 `覆盖 = null` + `ensureLiveEditBase`
     * —— 即**每帧整份拷贝一次参数窗口**（典型 ~6400 个元素/帧）。这里改为只把
     * **上一帧写过的区间**逐点还原，成本降为 O(上一帧笔刷宽度)。
     *
     * @param pv 当前参数窗口。
     */
    const resetLiveEditPreview = useCallback((pv: ParamViewSegment) => {
        const cur = liveEditOverrideRef.current;
        if (!cur || cur.key !== pv.key) return;
        const range = liveEditWrittenRangeRef.current;
        if (range !== null && range.key === pv.key) {
            if (restoreLiveEditRange({ edit: cur.edit, committed: pv.edit, range })) {
                liveEditVersionRef.current += 1;
                cur.version = liveEditVersionRef.current;
            }
        }
        liveEditWrittenRangeRef.current = null;
    }, []);

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
            const ordered = points.filter(
                (p) => Number.isFinite(p.frame) && Number.isFinite(p.value),
            );
            const uniq: StrokePoint[] = [];
            for (const p of ordered) {
                const f = Math.max(0, Math.floor(p.frame));
                const v = p.value;
                const last = uniq[uniq.length - 1];
                if (last && last.frame === f) {
                    last.value = v;
                } else {
                    uniq.push({ frame: f, value: v });
                }
            }
            if (uniq.length < 1) return;

            let minF = Number.POSITIVE_INFINITY;
            let maxF = 0;
            for (const p of uniq) {
                minF = Math.min(minF, p.frame);
                maxF = Math.max(maxF, p.frame);
            }
            minF = Math.max(0, Math.floor(minF));
            maxF = Math.max(minF, Math.floor(maxF));

            const pv = paramView;

            function applyToParamViewDense(denseStartFrame: number, dense: number[] | null) {
                if (!pv) return;
                if (pv.stride <= 0) return;
                const start = pv.startFrame;
                const step = pv.stride;
                const nextEdit = pv.edit.slice();
                for (let i = 0; i < nextEdit.length; i += 1) {
                    const f = start + i * step;
                    if (f < minF || f > maxF) continue;
                    if (mode === "restore") {
                        nextEdit[i] = pv.orig[i] ?? nextEdit[i];
                    } else if (dense) {
                        const j = f - denseStartFrame;
                        if (j >= 0 && j < dense.length) nextEdit[i] = dense[j] ?? nextEdit[i];
                    }
                }
                setParamView({ ...pv, edit: nextEdit });
            }

            // 提交是 fire-and-forget 调用（调用方不 await），任何 IPC 失败都必须
            // 在此兜底：否则 unhandledrejection，且 live 预览层会冻结在半提交状态。
            try {
                if (mode === "restore") {
                    applyToParamViewDense(minF, null);
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

                const len = maxF - minF + 1;
                const out = new Array<number>(len);

                // Prefer committing the exact dense values that were shown in the live preview.
                // This keeps the committed curve identical to what the user saw.
                const pvEdit = (() => {
                    const pvNow = paramView;
                    if (!pvNow) return null;
                    const live = liveEditOverrideRef.current;
                    if (live && live.key === pvNow.key) return live.edit;
                    return pvNow.edit;
                })();
                const pvStart = paramView?.startFrame ?? 0;
                const pvStride = paramView?.stride ?? 1;

                const canSliceFromPv =
                    Boolean(paramView) &&
                    pvStride === 1 &&
                    Array.isArray(pvEdit) &&
                    pvEdit.length > 0;

                if (canSliceFromPv) {
                    for (let f = minF; f <= maxF; f += 1) {
                        const i = f - pvStart;
                        out[f - minF] =
                            i >= 0 && i < (pvEdit as number[]).length
                                ? ((pvEdit as number[])[i] ?? 0)
                                : 0;
                    }
                } else {
                    // Fallback: replay the stroke in time order onto a dense buffer.
                    for (let i = 0; i < len; i += 1) out[i] = uniq[0].value;
                    for (let sIdx = 0; sIdx < uniq.length - 1; sIdx += 1) {
                        const a = uniq[sIdx];
                        const b = uniq[sIdx + 1];
                        const minSeg = Math.min(a.frame, b.frame);
                        const maxSeg = Math.max(a.frame, b.frame);
                        const denom = b.frame - a.frame;
                        for (let f = minSeg; f <= maxSeg; f += 1) {
                            if (f < minF || f > maxF) continue;
                            const t = denom === 0 ? 1 : (f - a.frame) / denom;
                            out[f - minF] = a.value + (b.value - a.value) * t;
                        }
                    }
                }

                await paramsApi.setParamFrames(trackId, editParam, minF, out, true);
                applyToParamViewDense(minF, out);
                // 同上：不立刻撤下覆盖层，避免波形先退回旧快照再跳到新快照。
                markLiveEditCommitted();
                invalidate();
                bumpRefreshToken();
            } catch (err) {
                // 失败时复位 live 预览层，避免覆盖层冻结；状态由下一次
                // refresh（paramsEpoch/滚动等）对账恢复。
                clearLiveEditOverride();
                invalidate();
                console.error("[commitStroke] failed:", err);
            }
        },
        [
            rootTrackId,
            editParam,
            pitchEnabled,
            paramView,
            setParamView,
            invalidate,
            bumpRefreshToken,
            clearLiveEditOverride,
            markLiveEditCommitted,
        ],
    );

    return {
        liveEditOverrideRef,
        ensureLiveEditBase,
        applyDenseToLiveEdit,
        resetLiveEditPreview,
        clearLiveEditOverride,
        clearCommittedLiveEditOverride,
        markLiveEditCommitted,
        commitStroke,
    };
}
