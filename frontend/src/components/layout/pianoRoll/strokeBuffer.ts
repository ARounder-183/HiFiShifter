/**
 * 笔画的**绝对帧**缓冲：把"正在画的那一笔"从"参数窗口的下标数组"里解放出来。
 *
 * ## 这个模块解决什么
 *
 * 参数编辑器的实时预览（live override）此前是**按窗口下标寻址**的：覆盖层是
 * `paramView.edit` 的一份副本，写入靠"绝对帧 → 下标"的反解
 * （`f = startFrame + i·stride`），提交靠把窗口切片出来再逐帧查表。窗口因此成了
 * 笔画的坐标系 —— 而窗口是**后端驱动**的：滚动、缩放、`paramsEpoch`、音高分析
 * 完成，任何一件事都会换掉它。
 *
 * 于是整条链路上长出了两处补丁：
 *
 * - 笔画期间**冻结取数**（`liveEditDeferral`），否则换窗口会让正在画的轨迹消失；
 * - 边缘滚屏的上界被硬夹在"已取数窗口末端"，否则"提交会写进后端，画面上却什么都不会
 *   出现"。
 *
 * 两处补丁合起来正是用户报告的缺陷：**绘制工具拖到水平边缘自动滚屏，滚过一定距离后
 * 参数线不再更新**。"一定距离"就是那个夹紧上界；而夹紧一旦被绕过（用户就是要画长线），
 * 提交会把窗口外的那段**写成 0** —— 它按 `pv.edit[i]` 查表，越界即 0，同时
 * `present` 位图却声明"这些帧我都改过"。
 *
 * ## 不变式
 *
 * > **笔画的坐标是绝对帧；窗口只是投影。**
 *
 * 缓冲只认帧号，所以换窗口、换步长、换快照都不影响它。渲染层要"某个窗口的稠密数组"
 * 时调用 {@link StrokeBuffer.project}，提交时用 {@link StrokeBuffer.segments}。
 * `liveEditDeferral` 与"滚屏夹紧"随之消失 —— 它们本来就是在替这套坐标系统
 * 打补丁。
 *
 * ## 存储
 *
 * 有序、互不重叠、已合并的**连续段**（{@link StrokeRun}）。常见路径（向右连续落笔）
 * 是 O(1) 摊销追加：段内数组带 headroom，左右两侧都能原地生长；只有真正回溯、擦除或
 * 与旧段合并时才走"重建受影响区间"的一般路径，成本与受影响帧数成正比 —— 而不是与
 * 窗口大小或笔画总长成正比（后者会退化成 O(n²)，这正是不能简单地"每次写都重建整段"
 * 的原因）。
 *
 * 内存上界：5 ms 帧周期画 10 分钟 = 12 万个数 ≈ 1 MB。超过 {@link MAX_STROKE_FRAMES}
 * 后停止接受新帧并置 {@link StrokeBuffer.truncated}，由调用方提示一次 —— 静默丢弃
 * 比截断更糟：用户会以为那一笔生效了。
 */

import type { ParamViewSegment } from "./types";
import { reportParamBlockDiagnostic } from "./paramBlockDiagnostics";

/** 单笔长度上限（帧）。5 ms 帧周期下 200 万帧 ≈ 2.8 小时，正常操作不可能达到。 */
export const MAX_STROKE_FRAMES = 2_000_000;

/** 一段**连续**的帧值。 */
export interface StrokeRun {
    /** 段首帧（绝对帧号）。 */
    readonly startFrame: number;
    /** `values[i]` 对应 `startFrame + i`。 */
    readonly values: readonly number[];
}

export interface StrokeWriteOptions {
    /**
     * 写入前的值域钳制（缺省恒等）。
     *
     * 【为什么必须有】live 覆盖是"用户看到的曲线"，而提交值会被后端按参数值域钳制。
     * 若预览不钳、提交被钳，拖拽时参数线因画布裁切看着"停在顶端"、波形却按超出值放大，
     * 松手后又跳回来。传入与后端同构的钳制函数（`paramRanges::clampParamWriteValue`）
     * 即可让两条路径逐值一致。
     */
    clamp?: (value: number) => number;
    /** 只写这一帧区间（闭区间，绝对帧号）；缺省为 `dense` 覆盖的整个区间。 */
    from?: number;
    to?: number;
}

export interface StrokeBuffer {
    /** 作用域（`v2|trackId|paramId`）。变了就是换参数 / 换轨，缓冲作废。 */
    readonly scope: string;
    /** 内容版本：每次真正改变内容都自增，供画布签名与投影缓存使用。 */
    readonly version: number;
    /** 当前覆盖的帧总数（各段长度之和）。 */
    readonly frameCount: number;
    /** 是否因超过 {@link MAX_STROKE_FRAMES} 而拒收了新帧。 */
    readonly truncated: boolean;
    /**
     * 写入一段稠密值。
     *
     * @param startFrame `dense[0]` 对应的绝对帧号。
     * @param dense 逐帧值；`undefined` / 非有限值表示"这一帧没有值"（例如步长 > 1 的
     *   窗口上，拖拽预览只落在栅格帧上）。这类帧**不会被写入**，也不会在段中间留洞 ——
     *   写入按"最大连续有值区间"切分。
     */
    write(
        startFrame: number,
        dense: readonly (number | undefined)[],
        options?: StrokeWriteOptions,
    ): void;
    /** 擦除 `[startFrame, endFrameExclusive)`：这些帧回到基准值。 */
    erase(startFrame: number, endFrameExclusive: number): boolean;
    /** 查某一帧的值（未覆盖 = `undefined`）。 */
    valueAt(frame: number): number | undefined;
    /**
     * 投影到某个取数窗口：未覆盖的帧取 `base`，返回长度 == `base.length` 的数组。
     *
     * 只遍历**与窗口相交的段**，复杂度是 O(可见已画帧) 而不是 O(窗口)。
     * `out` 用于复用上一帧的数组（同一窗口连续重绘时零分配）。
     */
    project(base: readonly number[], startFrame: number, stride: number, out?: number[]): number[];
    /** 提交用：把覆盖切成不超过 `maxFrames` 的连续段（升序）。 */
    segments(maxFrames: number): { startFrame: number; values: number[] }[];
    /** 覆盖的帧区间（闭开区间）；空缓冲为 null。 */
    range(): { start: number; end: number } | null;
    clear(): void;
}

/** 内部表示：可原地左右生长的连续段。 */
interface MutableRun {
    startFrame: number;
    /** `buf[offset + i]` 对应 `startFrame + i`。 */
    buf: number[];
    offset: number;
    length: number;
}

function runEnd(run: MutableRun): number {
    return run.startFrame + run.length;
}

/**
 * 创建一份笔画缓冲。
 *
 * @param scope 作用域键（`v2|trackId|paramId`）。缓冲只属于这个作用域。
 */
export function createStrokeBuffer(scope: string): StrokeBuffer {
    let runs: MutableRun[] = [];
    let version = 0;
    let frameCount = 0;
    let truncated = false;

    /** 把段的数据搬到一份更大的缓冲正中（两侧各留一半空位）。 */
    function reallocate(run: MutableRun, capacity: number): void {
        const buf = new Array<number>(capacity);
        const offset = Math.max(0, Math.floor((capacity - run.length) / 2));
        for (let i = 0; i < run.length; i += 1) buf[offset + i] = run.buf[run.offset + i];
        run.buf = buf;
        run.offset = offset;
    }

    function reserveLeft(run: MutableRun, count: number): void {
        if (count <= 0 || run.offset >= count) return;
        reallocate(run, Math.max((run.length + count) * 2, 64));
    }

    function reserveRight(run: MutableRun, count: number): void {
        if (count <= 0 || run.buf.length - run.offset - run.length >= count) return;
        reallocate(run, Math.max((run.length + count) * 2, 64));
    }

    function sliceRun(run: MutableRun, fromFrame: number, toFrameExclusive: number): MutableRun {
        const length = toFrameExclusive - fromFrame;
        const buf = new Array<number>(Math.max(length * 2, 64));
        const offset = Math.floor((buf.length - length) / 2);
        const sourceOffset = run.offset + (fromFrame - run.startFrame);
        for (let i = 0; i < length; i += 1) buf[offset + i] = run.buf[sourceOffset + i];
        return { startFrame: fromFrame, buf, offset, length };
    }

    function writeRange(
        rangeStart: number,
        rangeEnd: number,
        denseStart: number,
        dense: readonly (number | undefined)[],
        clampValue: ((value: number) => number) | undefined,
    ): void {
        const rangeLength = rangeEnd - rangeStart + 1;
        if (frameCount + rangeLength > MAX_STROKE_FRAMES) {
            truncated = true;
            return;
        }
        const last = runs.length > 0 ? runs[runs.length - 1] : null;

        // ① 与最后一段重叠或相接：原地覆盖 + 向右追加。连续落笔（最常见）走这里。
        if (last !== null && rangeStart >= last.startFrame && rangeStart <= runEnd(last)) {
            const previousEnd = runEnd(last);
            const overlapEnd = Math.min(rangeEnd + 1, previousEnd);
            for (let frame = rangeStart; frame < overlapEnd; frame += 1) {
                const value = dense[frame - denseStart] as number;
                last.buf[last.offset + (frame - last.startFrame)] = clampValue
                    ? clampValue(value)
                    : value;
            }
            if (rangeEnd + 1 > previousEnd) {
                const appendStart = Math.max(rangeStart, previousEnd);
                const appendCount = rangeEnd - appendStart + 1;
                reserveRight(last, appendCount);
                const base = last.offset + last.length;
                for (let i = 0; i < appendCount; i += 1) {
                    const value = dense[appendStart + i - denseStart] as number;
                    last.buf[base + i] = clampValue ? clampValue(value) : value;
                }
                last.length += appendCount;
                frameCount += appendCount;
            }
            version += 1;
            return;
        }

        // ② 向左延伸：与最后一段的头部重叠（用户反向拖拽）。整段原地前伸，O(新增帧)。
        //
        // 【为什么还要看**倒数第二段**】这条捷径只允许动最后一段。若新范围跨过了它前面
        // 的空洞、碰到了更早的段（例如段 A=0..4、B=10..14，写入 3..12），前伸会把 B 拉成
        // 3..14 与 A 重叠 —— 段与段互不重叠是全部查询的前提，一旦破坏，覆盖的帧会被
        // 静默吞掉。这种输入必须走一般路径去合并。
        const previous = runs.length > 1 ? runs[runs.length - 2] : null;
        if (
            last !== null &&
            rangeEnd >= last.startFrame &&
            rangeEnd + 1 <= runEnd(last) &&
            (previous === null || runEnd(previous) <= rangeStart)
        ) {
            const grow = last.startFrame - rangeStart;
            reserveLeft(last, grow);
            last.offset -= grow;
            last.startFrame = rangeStart;
            last.length += grow;
            for (let frame = rangeStart; frame <= rangeEnd; frame += 1) {
                const value = dense[frame - denseStart] as number;
                last.buf[last.offset + (frame - last.startFrame)] = clampValue
                    ? clampValue(value)
                    : value;
            }
            frameCount += grow;
            version += 1;
            return;
        }

        // ③ 一般路径：与若干旧段相交 / 相接 —— 重建受影响区间（成本 O(受影响帧)）。
        let lo = 0;
        while (lo < runs.length && runEnd(runs[lo]) <= rangeStart) lo += 1;
        let mergeLo = lo;
        if (lo > 0 && runEnd(runs[lo - 1]) === rangeStart) mergeLo = lo - 1;
        let hi = lo;
        while (hi < runs.length && runs[hi].startFrame <= rangeEnd) hi += 1;
        if (hi < runs.length && runs[hi].startFrame === rangeEnd + 1) hi += 1;

        let mergedStart = rangeStart;
        let mergedEnd = rangeEnd + 1;
        let replacedFrames = 0;
        for (let i = mergeLo; i < hi; i += 1) {
            mergedStart = Math.min(mergedStart, runs[i].startFrame);
            mergedEnd = Math.max(mergedEnd, runEnd(runs[i]));
            replacedFrames += runs[i].length;
        }
        const mergedLength = mergedEnd - mergedStart;
        const buf = new Array<number>(Math.max(mergedLength * 2, 64));
        const offset = Math.floor((buf.length - mergedLength) / 2);
        for (let i = mergeLo; i < hi; i += 1) {
            const run = runs[i];
            const shift = offset + (run.startFrame - mergedStart);
            for (let k = 0; k < run.length; k += 1) buf[shift + k] = run.buf[run.offset + k];
        }
        for (let frame = rangeStart; frame <= rangeEnd; frame += 1) {
            const value = dense[frame - denseStart] as number;
            buf[offset + (frame - mergedStart)] = clampValue ? clampValue(value) : value;
        }
        runs.splice(mergeLo, hi - mergeLo, {
            startFrame: mergedStart,
            buf,
            offset,
            length: mergedLength,
        });
        frameCount = frameCount - replacedFrames + mergedLength;
        version += 1;
    }

    return {
        scope,
        get version() {
            return version;
        },
        get frameCount() {
            return frameCount;
        },
        get truncated() {
            return truncated;
        },

        write(startFrame, dense, options) {
            if (dense.length === 0) return;
            const denseEnd = startFrame + dense.length - 1;
            const from = Math.max(options?.from ?? startFrame, startFrame);
            const to = Math.min(options?.to ?? denseEnd, denseEnd);
            if (to < from) return;
            const clampValue = options?.clamp;
            // 按"最大连续有值区间"切分：段中间不允许有洞，所以空洞把一次写入拆成多次。
            //
            // 【为什么用 null 而不是 -1 当哨兵】帧号是数字，负数在合成 / 负偏移的调用里
            // 是合法的；"区间起点 < 0 就等于没有起点"会让第一帧被后续帧覆盖掉 —— 表现
            // 为"这一笔最左边那一帧永远画不上"。
            let rangeStart: number | null = null;
            for (let frame = from; frame <= to + 1; frame += 1) {
                const value = frame <= to ? dense[frame - startFrame] : undefined;
                const usable = value !== undefined && Number.isFinite(value);
                if (usable) {
                    if (rangeStart === null) rangeStart = frame;
                    continue;
                }
                if (rangeStart !== null) {
                    writeRange(rangeStart, frame - 1, startFrame, dense, clampValue);
                    rangeStart = null;
                }
            }
        },

        erase(startFrame, endFrameExclusive) {
            if (endFrameExclusive <= startFrame || runs.length === 0) return false;
            const next: MutableRun[] = [];
            let changed = false;
            for (const run of runs) {
                const end = runEnd(run);
                if (end <= startFrame || run.startFrame >= endFrameExclusive) {
                    next.push(run);
                    continue;
                }
                changed = true;
                if (run.startFrame < startFrame) {
                    next.push(sliceRun(run, run.startFrame, startFrame));
                }
                if (end > endFrameExclusive) {
                    next.push(sliceRun(run, endFrameExclusive, end));
                }
            }
            if (!changed) return false;
            runs = next;
            frameCount = 0;
            for (const run of runs) frameCount += run.length;
            version += 1;
            return true;
        },

        valueAt(frame) {
            let lo = 0;
            let hi = runs.length - 1;
            while (lo <= hi) {
                const mid = (lo + hi) >> 1;
                const run = runs[mid];
                if (frame < run.startFrame) hi = mid - 1;
                else if (frame >= run.startFrame + run.length) lo = mid + 1;
                else return run.buf[run.offset + (frame - run.startFrame)];
            }
            return undefined;
        },

        project(base, startFrame, stride, out) {
            const length = base.length;
            const target =
                out !== undefined && out.length === length ? out : new Array<number>(length);
            for (let i = 0; i < length; i += 1) target[i] = base[i];
            if (length === 0 || !(stride > 0)) return target;
            const lastFrame = startFrame + (length - 1) * stride;
            for (const run of runs) {
                const end = runEnd(run);
                if (end <= startFrame || run.startFrame > lastFrame) continue;
                const firstIndex = Math.max(0, Math.ceil((run.startFrame - startFrame) / stride));
                const lastIndex = Math.min(length - 1, Math.floor((end - 1 - startFrame) / stride));
                for (let i = firstIndex; i <= lastIndex; i += 1) {
                    const frame = startFrame + i * stride;
                    target[i] = run.buf[run.offset + (frame - run.startFrame)];
                }
            }
            return target;
        },

        segments(maxFrames) {
            const out: { startFrame: number; values: number[] }[] = [];
            const limit = Math.max(1, Math.floor(maxFrames));
            for (const run of runs) {
                let taken = 0;
                while (taken < run.length) {
                    const count = Math.min(limit, run.length - taken);
                    const values = new Array<number>(count);
                    for (let i = 0; i < count; i += 1) {
                        values[i] = run.buf[run.offset + taken + i];
                    }
                    out.push({ startFrame: run.startFrame + taken, values });
                    taken += count;
                }
            }
            return out;
        },

        range() {
            if (runs.length === 0) return null;
            return { start: runs[0].startFrame, end: runEnd(runs[runs.length - 1]) };
        },

        clear() {
            if (runs.length === 0) return;
            runs = [];
            frameCount = 0;
            version += 1;
        },
    };
}

/**
 * 绘制中的参数曲线**覆盖层**。
 *
 * 【为什么不进 React 状态】它在指针频率上更新，走 state 会让整块面板以指针频率重渲染。
 * 波形面与曲线层都以 ref + 版本号的方式消费它。
 *
 * 【为什么带 `scope` 而不是窗口键】旧实现的 `key` 同时承载"属于谁"与"窗口在哪"，
 * 而这两件事的变更语义相反：换参数该丢弃笔画，换窗口什么都不该发生。缓冲是绝对帧的，
 * 于是"窗口在哪"从类型里彻底消失，只剩作用域。
 */
export interface LiveEditOverride {
    /** 作用域（`v2|trackId|paramId`）。 */
    readonly scope: string;
    /** 笔画缓冲（绝对帧）。 */
    readonly buffer: StrokeBuffer;
    /** 内容版本：每次写入自增（全局单调，不复用）。 */
    version: number;
    /**
     * `true` = 本次编辑**已提交**，覆盖层只是"等整工程快照追上"的占位。
     *
     * 【为什么需要】提交成功后若立刻撤下覆盖层，波形的幅度因子就退回**尚未刷新**的旧快照
     * —— 波形先跳回旧形状、等快照到达再跳到新形状（用户报告的"松手闪一下"）。保留覆盖层
     * 直到快照带着本次提交值到达，撤下时逐像素无损。
     *
     * 【为什么不会遮挡后续变化】撤下时机绑定在**快照内容追上**上（见
     * {@link liveEditCaughtUp}），因此撤销 / 切轨等任何来源的快照更新都会让覆盖层退场；
     * 而用户再次落笔时 {@link createLiveEditOverride} 会换一份新缓冲，旧值不会盖住新笔画。
     */
    committed?: boolean;
    /** 投影缓存（同一窗口 + 同一缓冲版本只算一次）。 */
    cache: LiveProjectionCache | null;
}

export interface LiveProjectionCache {
    key: string;
    bufferVersion: number;
    /**
     * 投影所依据的**基底数组身份**（`window.edit`）。
     *
     * 【为什么必须是键的一部分】投影 = 基底 + 缓冲覆盖。只按窗口键与缓冲版本
     * 缓存时，"缓冲没动、基底换了"会命中旧值：同一窗口重取（`paramsEpoch` /
     * 提交后刷新 / 撤销）时 `window.key` 不变，而 `window.edit` 是**新数组**。
     * 还原笔画正是这种形状（它从不写缓冲 ⇒ 版本恒为 1），于是后端已经还原成功、
     * 屏幕上却停在旧曲线，直到滚动 / 切参数换了窗口键才刷新。
     *
     * 【为什么用身份而不是深比较】每次取数都会换数组（`usePianoRollData` 的
     * `setParamView` 新建 `edit`），引用比较既准确又零成本；深比较会把这个
     * 几何重建热路径（一次数万次调用）拖垮。
     */
    base: readonly number[];
    values: number[];
}

/** 创建一份空的覆盖层。 */
export function createLiveEditOverride(scope: string, version: number): LiveEditOverride {
    return { scope, buffer: createStrokeBuffer(scope), version, cache: null };
}

/** 投影所需的窗口形状（`ParamViewSegment` 的子集）。 */
export interface LiveProjectionWindow {
    readonly key: string;
    readonly edit: readonly number[];
    readonly startFrame: number;
    readonly stride: number;
}

/**
 * 把覆盖层投影到某个取数窗口。
 *
 * 【为什么带缓存】曲线层与波形幅度映射都在渲染热路径上读它（幅度映射一次几何重建可询问
 * 数万次）。同一窗口 + 同一缓冲版本下必须返回**同一个数组**，否则每帧要分配几十万个
 * 短命对象。窗口换了才重新投影，且复用同一个输出数组。
 */
export function projectLiveEdit(live: LiveEditOverride, window: LiveProjectionWindow): number[] {
    const cached = live.cache;
    if (
        cached !== null &&
        cached.key === window.key &&
        cached.bufferVersion === live.buffer.version &&
        cached.base === window.edit
    ) {
        return cached.values;
    }
    // 复用输出数组只在**同一窗口**下成立（长度相同、`project` 会逐元素覆盖）；
    // 基底换了也照样复用，省一次分配，不影响正确性。
    const reuse = cached !== null && cached.key === window.key ? cached.values : undefined;
    const values = live.buffer.project(window.edit, window.startFrame, window.stride, reuse);
    live.cache = {
        key: window.key,
        bufferVersion: live.buffer.version,
        base: window.edit,
        values,
    };
    return values;
}

/**
 * 覆盖层是否已经把当前窗口的每一帧都兑现（提交后"等快照追上"的判据）。
 *
 * 【为什么不是"引用相等"】覆盖层与快照是两路独立 IPC；判据必须是**内容**：窗口里每一帧
 * 的投影值都等于后端刚给的曲线值，才算追上。容差与旧实现同口径（1e-9），因为快照是
 * f32 往返，逐位比较会把"已经一致"误判成"还没追上"，覆盖层就永远撤不下来。
 */
export function liveEditCaughtUp(
    live: LiveEditOverride,
    window: LiveProjectionWindow,
    tolerance = 1e-9,
): boolean {
    const projected = projectLiveEdit(live, window);
    const base = window.edit;
    if (projected.length !== base.length) return false;
    for (let i = 0; i < base.length; i += 1) {
        const left = projected[i];
        const right = base[i];
        if (left === right) continue;
        if (!Number.isFinite(left) || !Number.isFinite(right)) return false;
        if (Math.abs(left - right) > tolerance) return false;
    }
    return true;
}

/** 视口覆盖的帧区间（含端点，绝对帧号）。 */
export interface CurveViewport {
    readonly startFrame: number;
    readonly endFrame: number;
}

/**
 * 把"取数窗口视图"扩展到覆盖视口；窗口外的那一段由笔画缓冲直接供给。
 *
 * ## 为什么需要
 *
 * 曲线的 GL 顶点由 `projectCurvePoints` 产出，而它的循环是**遍历传入的数组**
 * （`if (tSec > visibleEndSec) break` 只裁右缘、从不补右缘）。于是"数组到哪儿，线就到
 * 哪儿"：取样域不是"用户在看什么"，而是"上一次取回了哪一段"。
 *
 * 笔画改成绝对帧缓冲之后，滚出取数窗口只是"暂时看不见"—— 那句话在**渲染域恰好等于
 * 窗口**时是假的：缓冲里已经画到第 N 帧、窗口只到第 M 帧（M < N）时，`[M, N]` 既不在
 * 数组里也不在画面上，直到下一个 256 px 提交台阶换来新窗口，才整段补画出来。用户看到
 * 的就是"拖到一定距离后参数线没了，再拖一会儿才出现"。
 *
 * ## 契约
 *
 * * 定义域 = 视口 ∩ (窗口 ∪ 笔画域)，与"上一次取回哪一段"无关（I-D1）；
 * * 窗口内的帧取窗口值（`view.edit`），窗口外且被笔画覆盖的帧取 `buffer.valueAt`，
 *   两者都不覆盖的帧写 `NaN` —— `projectCurvePoints` 对非有限值 skip，折线在缺口处
 *   自然断开。**写 0 会画出一条贴着底边的假曲线**。
 * * 不需要扩展时返回 `null`：调用方据此保留原对象身份，避免每帧换新数组（GL 侧按
 *   引用做去重，换引用等于每帧重建整条曲线）。
 *
 * @param args.view 已 resolve 过的帧视图（窗口投影）
 * @param args.live 绝对帧缓冲（`LiveEditOverride`）
 * @param args.viewport 视口覆盖的帧区间（含端点）
 * @param args.maxFrames 预算上界，缺省 `MAX_STROKE_FRAMES`
 */
export function extendCurveViewToViewport(args: {
    readonly view: ParamViewSegment;
    readonly live: LiveEditOverride;
    readonly viewport: CurveViewport;
    readonly maxFrames?: number;
}): ParamViewSegment | null {
    const { view, live, viewport } = args;
    const stride = Math.max(1, Math.floor(view.stride));
    const viewEnd = view.startFrame + Math.max(0, view.edit.length - 1) * stride;
    const strokeRange = live.buffer.range();

    let start = Math.min(view.startFrame, viewport.startFrame);
    let end = Math.max(viewEnd, viewport.endFrame);
    if (strokeRange !== null) {
        start = Math.min(start, strokeRange.start);
        end = Math.max(end, strokeRange.end - 1);
    }
    // 【为什么只延伸半屏】一次超长笔画（用户真的从工程头画到尾）不该让每一帧的投影变成
    // O(工程时长)。视口之外再给半屏，足够覆盖"刚滚过去、取数还没回来"的那一段；
    // 更远的空白是取数的责任（`usePianoRollData` 的视口订阅），不是渲染的。
    const halfSpan = Math.max(stride, Math.round((viewport.endFrame - viewport.startFrame) / 2));
    start = Math.max(start, Math.min(view.startFrame, viewport.startFrame - halfSpan));
    end = Math.min(end, Math.max(viewEnd, viewport.endFrame + halfSpan));
    // 帧号从 0 起；lattice 必须与窗口同一格（两者都是 stride 的整数倍）。
    start = Math.max(0, Math.floor(start / stride) * stride);
    if (start === view.startFrame && end === viewEnd) return null;
    if (end < start) return null;

    const count = Math.floor((end - start) / stride) + 1;
    const budget = args.maxFrames ?? MAX_STROKE_FRAMES;
    if (!Number.isFinite(count) || count <= 0) return null;
    if (count > budget) {
        // 宁可退回旧行为（只画窗口那一段），也不许一次超长笔画把每帧的投影变成
        // O(工程时长)。诊断只报一次/秒，见 `reportParamBlockDiagnostic` 的限频。
        reportParamBlockDiagnostic({
            code: "curve_domain_budget",
            gesture: "layout",
            rootTrackId: null,
            blockId: null,
            frame: count,
            detail: `curve view needs ${count} frames (budget ${budget})`,
        });
        return null;
    }

    const edit = new Array<number>(count);
    const orig = new Array<number>(count);
    const edited = view.edited === undefined ? null : new Array<boolean>(count);
    for (let index = 0; index < count; index += 1) {
        const frame = start + index * stride;
        const source = (frame - view.startFrame) / stride;
        if (Number.isInteger(source) && source >= 0 && source < view.edit.length) {
            edit[index] = view.edit[source];
            const base = view.orig[source];
            orig[index] = Number.isFinite(base) ? base : Number.NaN;
            if (edited !== null && view.edited !== undefined) {
                edited[index] = view.edited[source] === true;
            }
            continue;
        }
        const value = live.buffer.valueAt(frame);
        // 【没有数据写 NaN，不写 0】NaN 让折线在缺口处断开；0 会画出一条贴底的假线。
        edit[index] = value === undefined ? Number.NaN : value;
        orig[index] = Number.NaN;
        if (edited !== null) edited[index] = false;
    }
    return {
        ...view,
        // 与 `projectLiveEdit` 的缓存键分开：那张缓存的键是窗口，这里是"窗口 + 扩展域"。
        key: `${view.key}|live:${start}:${end}`,
        startFrame: start,
        edit,
        orig,
        ...(edited === null ? {} : { edited }),
    };
}
