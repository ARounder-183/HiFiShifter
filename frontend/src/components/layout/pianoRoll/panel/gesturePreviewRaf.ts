/**
 * 指针手势预览的 rAF 合帧器。
 *
 * 【为什么存在】pointermove 在高刷鼠标上可达 125–1000Hz，而一次预览重算
 * （计划构建 + 渲染规格）是 O(数据量) 的，逐事件执行在长工程上明显卡顿。
 * schedule() 把重算合到**每帧至多一次**；flush() 在提交前同步冲刷挂起帧
 * （提交读的就是挂起的计划）；cancel() 在取消路径丢弃挂起帧。与
 * paramBlockDrag 的 schedule/flush 同一模式，收口成可复用的小工厂。
 */
export function createRafCoalescedReplay(replay: () => void): {
    schedule: () => void;
    flush: () => void;
    cancel: () => void;
} {
    let rafId: number | null = null;
    let hasPendingReplay = false;
    // 每次调度带独立令牌：cancel / flush 之后，浏览器已排队的旧 rAF
    // 回调仍可能触发；它不能重放新一轮的挂起预览，也不能清掉新帧。
    let activeToken: object | null = null;
    const runPending = (token: object | null) => {
        if (token === null || token !== activeToken) return;
        activeToken = null;
        rafId = null;
        if (!hasPendingReplay) return;
        hasPendingReplay = false;
        replay();
    };
    return {
        schedule() {
            // 每次 schedule 都更新"待回放"标记：调用方在 rAF 回调之外也可能
            // 写入新的输入（如边缘滚动回调），提交时要用**最后一份**。
            hasPendingReplay = true;
            if (rafId != null) return;
            const token = {};
            activeToken = token;
            rafId = requestAnimationFrame(() => runPending(token));
        },
        flush() {
            if (rafId != null) cancelAnimationFrame(rafId);
            rafId = null;
            // 没有挂起帧时不要重放：正常松手前的最后一次 move 已经显示过，
            // 无条件 replay 会在提交路径上再跑一遍 O(数据量) 的重算。
            runPending(activeToken);
        },
        cancel() {
            if (rafId != null) cancelAnimationFrame(rafId);
            activeToken = null;
            rafId = null;
            hasPendingReplay = false;
        },
    };
}
