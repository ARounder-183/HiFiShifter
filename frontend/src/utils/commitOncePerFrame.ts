/**
 * 帧合并提交。
 *
 * 【要解决的问题】滚轮 / 拖拽这类**连续手势**在单位时间内会产生几十个事件，而每个
 * 事件都触发一次完整的状态提交。提交的代价不只是"写一个数"：订阅方会重算整条派生
 * 链（标尺刻度、网格、波形）并重渲染。一次滚轮手势因此产生几十次全量重算，用户看到
 * 的就是"抽搐"，且**滚得越快越明显**。
 *
 * 本模块把一段手势里的多次写入合并成**每帧最多一次**：中间值只记录、不提交，帧到达
 * 时提交最后一个值。语义上等价于"只关心手势当前落在哪里"，这正是连续调值想要的。
 *
 * 特殊说明：**不能只在结束时提交一次**——那样拖动过程中画面完全不跟随，手感崩掉。
 * 合并的粒度必须是帧，而不是手势。
 */
export interface FrameCommitter<T> {
    /** 记下最新值，并确保本帧会有一次提交（同一帧内的多次调用只提交最后一次）。 */
    schedule(value: T): void;
    /** 立即提交挂起的值（手势结束 / 卸载 / 需要同步读取最终值前调用）。 */
    flush(): void;
    /** 丢弃挂起的值，不提交（卸载时若不想再触碰 store）。 */
    cancel(): void;
    /** 是否有挂起的值尚未提交。 */
    isPending(): boolean;
}

/** 帧调度器（默认走 rAF；测试可注入）。 */
export interface FrameScheduler {
    request(callback: () => void): number;
    cancel(handle: number): void;
}

const rafScheduler: FrameScheduler = {
    request: (callback) =>
        typeof requestAnimationFrame === "function"
            ? requestAnimationFrame(callback)
            : (setTimeout(callback, 16) as unknown as number),
    cancel: (handle) => {
        if (typeof cancelAnimationFrame === "function") cancelAnimationFrame(handle);
        else clearTimeout(handle);
    },
};

/**
 * 创建一个帧合并提交器。
 *
 * @param commit 真正执行提交的回调（每帧最多调用一次）。
 * @param scheduler 帧调度器；缺省用 rAF（非浏览器环境退化为 16ms 定时器）。
 * @returns 提交器句柄。
 */
export function createFrameCommitter<T>(
    commit: (value: T) => void,
    scheduler: FrameScheduler = rafScheduler,
): FrameCommitter<T> {
    // 用对象包一层：`T` 本身可能是 `null`，需要区分"没有挂起值"与"挂起值是 null"。
    let pending: { value: T } | null = null;
    let handle: number | null = null;

    const run = () => {
        handle = null;
        const next = pending;
        pending = null;
        if (next === null) return;
        commit(next.value);
        // 提交期间又 schedule 了新的值：再排一帧（绝不在同一帧里递归提交）。
        if (pending !== null && handle === null) handle = scheduler.request(run);
    };

    return {
        schedule(value: T) {
            pending = { value };
            if (handle !== null) return;
            handle = scheduler.request(run);
        },
        flush() {
            if (handle !== null) {
                scheduler.cancel(handle);
                handle = null;
            }
            run();
        },
        cancel() {
            if (handle !== null) {
                scheduler.cancel(handle);
                handle = null;
            }
            pending = null;
        },
        isPending() {
            return pending !== null;
        },
    };
}
