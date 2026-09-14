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
 * 【唯一例外：`flush()`】面板的输入路径若在事件任务里**同步**写了 DOM / Canvas2D
 * （参数编辑器的缩放落地、值域滚动条等），就把那次绘制改为 `flush()`——否则同一份
 * 视口会被两套图层分两帧呈现（用户报告"参数线 / 原始音高线 / 播放头迟缓几帧"）。
 * 见 `flush` 的说明。
 *
 * 【与其他模块的关系】
 * - 上游：`scrollKernel.subscribe`（滚动 / 缩放）、交互 controller（手势预览）、
 *   内容变更（Redux 提交后的场景重建）都会调用 `invalidate()`；需要同任务提交的
 *   输入路径调用 `flush()`。
 * - 下游：`draw` 回调由宿主视图提供，内部按层序提交各 program 的绘制。
 * - 独立性：不依赖 DOM（rAF 通过参数注入，测试可传同步实现）。
 *
 * 【设计约束】
 * 1. **按需调度**：`start()` 本身不绘制，也不常驻 rAF——只有标脏才调度。
 *    播放头等逐帧动画由调用方每帧标脏自然形成连续绘制，不需要循环自身常驻。
 * 2. `draw()` 在 `dirty = false` **之后**执行：回调内再次 `invalidate()`
 *    （例如绘制过程中发现新的变更）会正确调度下一帧，而不是被本次清理吞掉。
 *    `flush()` 保持同一顺序，因此绘制内的标脏同样不会被吞掉。
 * 3. `stop()` 取消未执行的帧并清空脏标记；停止后 `invalidate()` / `flush()`
 *   都不再绘制（防止卸载后仍持有回调）。
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
    /**
     * **立即**提交当前脏状态（不等待下一帧）：取消已排队的帧 → 清脏 → 绘制。
     *
     * 未标脏时什么都不做（与 `invalidate` 的去重语义一致），因此可以安全地放在高频
     * 路径上。停止后不绘制（同 `invalidate`）。
     *
     * 【为什么需要它：DOM 与 GL 必须落在同一个任务里】
     * 默认调度（rAF 合并）对**纯自绘**面板是完备的：没有随原生滚动移动的 DOM 内容层，
     * 绘制早晚一帧用户看不出来。但参数编辑器不是纯自绘——它的输入路径会在事件任务里
     * **同步**写下 DOM / Canvas2D（缩放落地时的标尺 transform、网格层、主画布、值域
     * 滚动条），此时 GL 侧（参数线 / 原始音高线 / 播放头）要等下一帧才跟上，屏幕上就
     * 出现"同一份视口、两套图层不同帧"的分裂。
     *
     * 实测（Chrome，参数编辑器内滚轮缩小）：面板在 t=6384 写下 DOM/Canvas2D，GL 到
     * t=6446 才用同一个 scrollLeft 重绘——相差一帧到数帧（React 提交 + rAF 重新排队），
     * 用户看到的就是"这些线迟缓几帧"。这类路径改为调用本方法后，DOM 与 GL 在同一任务
     * 内提交，手感与旧实现（原生滚动 + 同步重绘）一致。
     */
    flush(): void;
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

        flush() {
            if (!running || !dirty) return;
            // 先取消已排队的帧再绘制：本次绘制已经覆盖了那一帧要做的全部工作，
            // 留着它只会多画一次（绘制很贵——曲线层每帧重建几何）。
            if (frameHandle !== null) {
                cancelFrame(frameHandle);
                frameHandle = null;
            }
            dirty = false;
            options.draw();
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
