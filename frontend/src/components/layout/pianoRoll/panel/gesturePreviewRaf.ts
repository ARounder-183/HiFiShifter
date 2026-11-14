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
    return {
        schedule() {
            if (rafId != null) return;
            rafId = requestAnimationFrame(() => {
                rafId = null;
                replay();
            });
        },
        flush() {
            if (rafId != null) {
                cancelAnimationFrame(rafId);
                rafId = null;
            }
            replay();
        },
        cancel() {
            if (rafId != null) {
                cancelAnimationFrame(rafId);
                rafId = null;
            }
        },
    };
}
