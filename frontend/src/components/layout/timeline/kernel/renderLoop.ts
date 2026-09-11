/**
 * 时间轴渲染内核 · 渲染循环（脏标记 + rAF）
 *
 * 【主要内容】
 * 提供 `invalidate()` 标脏 + rAF 合并的绘制调度：同一帧内多次标脏只触发一次
 * `draw()`；无标脏时不调度任何帧（空闲零 CPU）。
 *
 * 【作用】
 * 自绘滚动取代原生 scroller 后，渲染不再由「滚动事件必须同帧提交」约束——
 * 没有随原生滚动移动的 DOM 内容层需要对齐，绘制可以完全交给 rAF 调度。
 * 本模块把这个调度收敛成唯一入口：所有会改变画面的操作只需 `invalidate()`，
 * 由循环负责合并与执行。
 *
 * 【与其他模块的关系】
 * - 上游：`scrollKernel.subscribe`（滚动 / 缩放）、交互 controller（手势预览）、
 *   内容变更（Redux 提交后的场景重建）都会调用 `invalidate()`。
 * - 下游：`draw` 回调由宿主视图提供，内部按层序提交各 program 的绘制。
 * - 独立性：不依赖 DOM（rAF 通过参数注入，测试可传同步实现）。
 *
 * 【设计约束】
 * 1. **按需调度**：`start()` 本身不绘制，也不常驻 rAF——只有标脏才调度。
 *    播放头等逐帧动画由调用方每帧标脏自然形成连续绘制，不需要循环自身常驻。
 * 2. `draw()` 在 `dirty = false` **之后**执行：回调内再次 `invalidate()`
 *    （例如绘制过程中发现新的变更）会正确调度下一帧，而不是被本次清理吞掉。
 * 3. `stop()` 取消未执行的帧并清空脏标记；停止后 `invalidate()` 不再调度
 *    （防止卸载后仍持有回调）。
 */

/** 渲染循环构造参数。 */
export interface RenderLoopOptions {
    /** 绘制回调（同步执行；内部不得抛出未捕获异常）。 */
    draw: () => void;
    /**
     * 帧调度（默认 `requestAnimationFrame`）。
     *
     * @param callback 帧回调。
     * @returns 帧句柄。
     */
    requestFrame?: (callback: FrameRequestCallback) => number;
    /**
     * 取消帧（默认 `cancelAnimationFrame`）。
     *
     * @param handle 帧句柄。
     */
    cancelFrame?: (handle: number) => void;
}

/** 渲染循环句柄。 */
export interface RenderLoop {
    /** 标脏并调度一帧（同一帧内多次调用只绘制一次）。 */
    invalidate(): void;
    /** 启动循环（挂载时调用）；本身不绘制。 */
    start(): void;
    /** 停止循环并取消未执行的帧。 */
    stop(): void;
    /** 当前是否有未处理的脏标记（诊断 / 测试用）。 */
    isDirty(): boolean;
}

/**
 * 创建渲染循环。
 *
 * 流程：`invalidate` 置脏并按需调度 → 帧回调检查运行状态与脏标记 → 清脏后执行
 * `draw`。
 *
 * 特殊说明：帧回调即使在被 `stop()` 取消后仍被调用（浏览器边界情况），也会因
 * `running === false` 直接返回，不会在卸载后执行绘制。
 *
 * @param options 构造参数。
 * @returns 循环句柄；须长生命周期持有（随宿主视图创建 / 销毁）。
 */
export function createRenderLoop(options: RenderLoopOptions): RenderLoop {
    const requestFrame =
        options.requestFrame ??
        ((callback: FrameRequestCallback) => requestAnimationFrame(callback));
    const cancelFrame = options.cancelFrame ?? ((handle: number) => cancelAnimationFrame(handle));

    let frameHandle: number | null = null;
    let running = false;
    let dirty = false;

    /** 帧回调：清脏后绘制。 */
    function tick(): void {
        frameHandle = null;
        if (!running || !dirty) return;
        dirty = false;
        options.draw();
    }

    /** 按需调度一帧（已在队列中则跳过）。 */
    function schedule(): void {
        if (!running || frameHandle !== null) return;
        frameHandle = requestFrame(tick);
    }

    return {
        invalidate() {
            dirty = true;
            schedule();
        },

        start() {
            running = true;
        },

        stop() {
            running = false;
            dirty = false;
            if (frameHandle !== null) {
                cancelFrame(frameHandle);
                frameHandle = null;
            }
        },

        isDirty() {
            return dirty;
        },
    };
}
