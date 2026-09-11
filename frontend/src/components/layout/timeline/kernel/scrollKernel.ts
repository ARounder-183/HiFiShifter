/**
 * 时间轴渲染内核 · 滚动内核（ScrollKernel）
 *
 * 【主要内容】
 * 持有新内核唯一的视口真值（scrollLeft / scrollTop / pxPerSec / rowHeight），并提供
 * 两个写入入口：直接滚动（setScrollLeft / setScrollTop）与锚点缩放（setZoom）。
 * 内容尺寸不缓存，而是每次读取时经注入的 projectSec / trackCount / viewportHeightPx
 * 现算——增删轨道、改工程时长、宿主 resize 都无需重建内核。
 *
 * 【作用】
 * 自绘滚动取代原生 scroller 后，滚动位置不再由浏览器维护，边界必须由内核自己负责。
 * 本模块是内核里**唯一做钳制的地方**：写入时即钳到 [0, maxScroll]，读到的值恒为真值，
 * 从根上消除「写原生 scrollLeft 被浏览器钳制后回读」导致的视口状态与渲染不一致
 * （sticky 层错位、缩放锚点漂移）这一类历史问题。
 *
 * 【与其他模块的关系】
 * - 上游（输入）：wheel / 自绘滚动条 / 键盘 / 中键平移把归一化后的像素增量交给
 *   setScrollLeft / setScrollTop / setZoom。
 * - 下游（消费）：RenderLoop、SceneBuilder、GeometryStore、HitTestIndex 从 get() 读
 *   视口，据此算可见窗口与 viewOrigin uniform；变更经 subscribe 通知标脏。
 * - 独立性：Spike 期间不引用 `runtime/` 下任何模块，也不依赖 React；本模块是纯 TS
 *   状态容器，可在任意环境（含测试）构造。
 *
 * 【水平上限语义（与既有实现对齐）】
 * maxScrollLeft() = **工程宽度**（不是「工程宽 − 视口宽」），与既有
 * `runtime/timelineScrollRange.resolveTimelineScrollRange` 的语义一致：允许把工程右端
 * 滚到视口左缘，使「以指针 / 播放头为锚点」的缩放在任意缩放下都能成立，且上限随缩放
 * **连续变化**、不会在「工程宽 = 视口宽」处骤降产生跳变。竖直方向无此需求，沿用标准
 * 语义「内容高 − 视口高」。
 *
 * 【生命周期】
 * 内核持有滚动状态，必须**长生命周期创建一次**（宿主用 ref / 模块级持有），
 * 不得随 React 渲染重建，否则滚动位置会静默归零。
 *
 * 【强制约束（评审检查项）】
 * 1. 钳制只在本文件做一次；外部不得再钳制，也不得依赖「写入后回读被修正的值」。
 *    外部边界（宿主尺寸 / 工程时长 / 轨道数）变化后，宿主必须调用 reclamp()。
 * 2. get() 返回的引用在字段未变化时保持稳定（冻结对象 + 变更时整体替换）：下游每帧
 *    读取不产生分配，且可用引用比较判定「视口是否变化」。
 * 3. 只有状态真正变化才通知订阅者；滚轮亚像素噪声不得触发空重绘。
 */

/**
 * pxPerSec 的默认下限。
 *
 * 与 `runtime/timelineAxis.ts` 的 MIN_PX_PER_SEC 取同一量级：0 或负的 pxPerSec 会让
 * 「时间 × pxPerSec」退化为零宽内容，并让锚点换算出现除零 / 反向坐标。
 */
const DEFAULT_MIN_PX_PER_SEC = 1e-9;

/** 滚动位置（px）的变化判定容差：差异小于此值视为未变化（滤掉滚轮亚像素噪声）。 */
const SCROLL_EPSILON_PX = 1e-6;

/**
 * pxPerSec 的变化判定容差（相对比例）。
 *
 * 必须用**相对**容差而非绝对容差：pxPerSec 跨度大（生产区间约 0.5–8000），
 * 绝对容差在极小缩放下会把真实的缩放步进误判为「未变化」，导致缩放卡死。
 */
const ZOOM_EPSILON_RATIO = 1e-9;

/**
 * 时间轴视口状态：渲染与命中测试共用的唯一一份视口真值。
 *
 * 特殊说明：所有字段均为**已钳制的真值**，读取方不得再做边界修正；
 * 对象在字段变化时整体替换并冻结，因此可安全做引用比较。
 */
export interface TimelineViewportState {
    /** 水平滚动位置（CSS px），恒在 [0, maxScrollLeft()] 内。 */
    readonly scrollLeft: number;
    /** 竖直滚动位置（CSS px），恒在 [0, maxScrollTop()] 内。 */
    readonly scrollTop: number;
    /** 每秒对应的 CSS 像素数，恒 >= minPxPerSec 且 > 0。 */
    readonly pxPerSec: number;
    /** 单条轨道高度（CSS px）；本任务为构造时的定值（纵向缩放由后续任务引入）。 */
    readonly rowHeight: number;
}

/**
 * 内核构造参数。
 *
 * 尺寸类参数一律以**函数**注入而非快照：工程数据与宿主尺寸在运行时变化频繁，
 * 函数形式让内核无需订阅外部事件即可读到最新值（避免「忘记同步」类缺陷）。
 *
 * 【注入契约（硬性，宿主必须满足）】
 * 这些 getter 会在每次写入与读取路径被调用，因此：
 * - 必须 **O(1) 且不触发同步布局**：`viewportHeightPx` 请使用 ResizeObserver 缓存的
 *   尺寸，**不要**直接读 `clientHeight`（会强制样式重算，单次可达 0.1–1ms，
 *   直接吃掉帧预算）；
 * - `projectSec` / `trackCount` 请使用 memo 后的派生值，不要每次遍历全部 clip。
 */
export interface ScrollKernelOptions {
    /** 初始水平缩放（每秒像素数），构造时会被夹到 [minPxPerSec, maxPxPerSec]。 */
    pxPerSec: number;
    /** 单条轨道高度（CSS px）。 */
    rowHeight: number;
    /** 内容总时长（秒），用于算内容宽度。 */
    projectSec: () => number;
    /** 轨道总数，用于算内容高度。 */
    trackCount: () => number;
    /**
     * 宿主视口高度（CSS px），用于竖直滚动上限（内容高 − 视口高）。
     *
     * 特殊说明：水平方向**不需要**视口宽——水平上限 = 工程宽度（见文件头
     * 「水平上限语义」），这是与既有 `resolveTimelineScrollRange` 对齐的结果，
     * 因此本接口不接收 `viewportWidthPx`。
     */
    viewportHeightPx: () => number;
    /**
     * pxPerSec 下限，缺省 1e-9。
     *
     * 支持函数形式：下限随**工程长度与视口宽度**变化（工程短于视口时允许缩得更小），
     * 必须与滚轮缩放解析所用的 `resolveTimelineMinPxPerSec` 同源——冻结成常量会让
     * 内核钳制与解析结果不一致（标尺能缩得比网格更小）。
     */
    minPxPerSec?: number | (() => number);
    /** pxPerSec 上限，缺省 +Infinity；同样支持函数形式。 */
    maxPxPerSec?: number | (() => number);
}

/** 滚动内核公开 API（输入层写、渲染层读、订阅者标脏）。 */
export interface ScrollKernel {
    /**
     * 读取当前视口状态。
     *
     * @returns 冻结的状态对象；字段未变化时**返回同一个引用**，故可直接用 `===`
     *          判断「自上次读取以来视口是否变化」。
     */
    get(): TimelineViewportState;

    /**
     * 设置水平滚动位置。
     *
     * 流程：非法值（NaN / Infinity）直接忽略 → 钳制到 [0, maxScrollLeft()] → 变化才提交。
     *
     * @param px 目标水平滚动位置（CSS px，可为越界值，内部负责钳制）。
     */
    setScrollLeft(px: number): void;

    /**
     * 设置竖直滚动位置。
     *
     * 流程：非法值忽略 → 钳制到 [0, maxScrollTop()] → 变化才提交。
     *
     * @param px 目标竖直滚动位置（CSS px，可为越界值）。
     */
    setScrollTop(px: number): void;

    /**
     * 以 anchorScreenX（视口内 CSS px）为锚点缩放。
     *
     * 流程：
     * 1. 把 pxPerSec 夹到 [minPxPerSec, maxPxPerSec]；
     * 2. 由「锚点下的工程时间不变」反算新 scrollLeft：
     *    `anchorSec = (scrollLeft + anchorScreenX) / 旧pxPerSec`，
     *    `新scrollLeft = anchorSec × 新pxPerSec − anchorScreenX`；
     * 3. 用**新 pxPerSec**算出的上限钳制 scrollLeft（上限 = 工程宽度，见文件头「水平上限语义」）；
     * 4. pxPerSec 与 scrollLeft 一并提交，变化才通知。
     *
     * 特殊说明：anchorScreenX 允许落在宿主视口之外（指针移出宿主、拖拽缩放），
     * 故不夹取到屏幕范围 [0, 视口宽)，否则锚点会被钉在视口边缘造成缩放漂移。
     *
     * @param pxPerSec 目标缩放（每秒像素数），非法值忽略。
     * @param anchorScreenX 锚点在视口内的水平位置（CSS px）。
     * @returns **实际生效**的 pxPerSec（可能因下限 / 上限钳制而与入参不同）。
     *          调用方必须用它同步 React 侧派生量（标尺刻度 / 内容宽度），
     *          否则会显示成未钳制的请求值。
     */
    setZoom(pxPerSec: number, anchorScreenX: number): number;

    /**
     * 原子地设置缩放与水平滚动位置（外部命令式视口写入）。
     *
     * 【为什么需要它】旧实现里「键盘缩放 / 聚焦播放光标 / 自动滚屏 / 参数编辑器同步」
     * 都是「先算出目标 pxPerSec 与目标 scrollLeft，再一起落到原生 scroller」。自绘
     * 滚动后这条路径必须能一次性提交两个字段，否则会出现「先用旧上限钳制新缩放」
     * 的中间态（与 setZoom 的注释同因）。
     *
     * 流程：解析候选 pxPerSec（缺省沿用当前，非法回退下限）→ 用**目标 pxPerSec**
     * 对应的上限钳制候选 scrollLeft（缺省沿用当前）→ 一次 commit（变化才通知）。
     *
     * @param next 需要覆盖的字段；未给出的字段沿用当前值。
     * @returns 提交后的视口状态（已钳制真值）；未发生变化时返回同一引用。
     */
    setViewport(next: { pxPerSec?: number; scrollLeft?: number }): TimelineViewportState;

    /**
     * 订阅状态变化。
     *
     * 特殊说明：派发是**同步**的（渲染在 rAF 内，输入→标脏必须同帧完成）；
     * 回调内退订或新增订阅不影响本轮派发。
     *
     * @param listener 状态变化回调，不接收参数（变化后用 get() 读最新值）。
     * @returns 退订函数；重复退订是安全的空操作。
     */
    subscribe(listener: () => void): () => void;

    /**
     * 内容总宽度（CSS px）= projectSec × pxPerSec。
     *
     * @returns 内容宽度；projectSec 非法时按 0 处理。
     */
    contentWidthPx(): number;

    /**
     * 内容总高度（CSS px）= trackCount × rowHeight。
     *
     * @returns 内容高度；trackCount 非法时按 0 处理。
     */
    contentHeightPx(): number;

    /**
     * 水平滚动上限 = **工程宽度**（见文件头「水平上限语义」）。
     *
     * @returns 上限（CSS px）；允许把工程右端滚到视口左缘，故工程不足一屏时上限
     *          仍为工程宽度（不为 0）。
     */
    maxScrollLeft(): number;

    /**
     * 竖直滚动上限 = max(0, 内容高 − 视口高)。
     *
     * @returns 上限（CSS px）；轨道不足一屏时为 0。
     */
    maxScrollTop(): number;

    /**
     * 设置行高（竖直缩放）。
     *
     * 特殊说明：行高由 React 侧持有（左侧轨道头与内核必须同源），内核只接收
     * 结果。行高变化会改变内容高度，因此内部顺带按新边界重新钳制 `scrollTop`
     * （否则轨道变矮后可能停留在越界位置，画面出现空白）。
     *
     * @param px 新行高（CSS px）；非有限值或与当前值几乎相同时不做任何变更。
     * @returns 无返回值；值真正变化时经 commit 通知订阅者。
     */
    setRowHeight(px: number): void;

    /**
     * 按当前外部边界重新钳制两轴滚动位置。
     *
     * 特殊说明：钳制只在写入时发生（约束 1），因此宿主尺寸变化（resize）、
     * 工程变短（删 clip）或轨道变少（删轨）之后，已提交的 scrollLeft / scrollTop
     * 可能超出新的上限。宿主**必须**在这些时机调用本方法；内部经 commit 提交，
     * 位置真正变化时会正常通知订阅者。
     *
     * @returns 无返回值；若位置未越界则不产生任何状态变化与通知。
     */
    reclamp(): void;
}

/**
 * 数值夹取。
 *
 * @param value 待夹取值。
 * @param min 下限（含）。
 * @param max 上限（含）。
 * @returns 夹取结果；min > max 时结果为 max（Math.min/Math.max 组合的既有语义）。
 */
function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
}

/**
 * 滚动位置（px）的浮点容差比较。
 *
 * 作用：滚动位置经缩放换算后必然带浮点噪声，严格 `!==` 会把 1e-13 级差异当成
 * 「状态变化」，导致渲染循环空转。
 *
 * @param a 待比较值。
 * @param b 待比较值。
 * @returns 两值差异小于 SCROLL_EPSILON_PX 时为 true（视为未变化）。
 */
function nearlyEqual(a: number, b: number): boolean {
    return Math.abs(a - b) < SCROLL_EPSILON_PX;
}

/**
 * pxPerSec 的浮点容差比较（相对容差）。
 *
 * 作用：与滚动位置不同，pxPerSec 是"比例"量——必须按相对比例判定，否则在极小
 * 缩放下会把真实的缩放步进误判为未变化（见 ZOOM_EPSILON_RATIO 注释）。
 *
 * @param a 待比较值。
 * @param b 待比较值。
 * @returns 相对差异不超过 ZOOM_EPSILON_RATIO 时为 true（视为未变化）。
 */
function zoomEquals(a: number, b: number): boolean {
    return Math.abs(a - b) <= ZOOM_EPSILON_RATIO * Math.max(1, Math.abs(a), Math.abs(b));
}

/**
 * 规范化 pxPerSec。
 *
 * 规则：非有限值（NaN / ±Infinity）回退到下限，保证内核内部 pxPerSec 恒 > 0，
 * 除法与锚点换算不会产生 NaN（渲染热路径宁可回退到安全值也不抛错）。
 *
 * @param value 候选缩放值。
 * @param min 下限。
 * @param max 上限。
 * @returns 位于 [min, max] 内的有限值。
 */
function sanitizePxPerSec(value: number, min: number, max: number): number {
    if (!Number.isFinite(value)) return min;
    return clamp(value, min, max);
}

/**
 * 创建滚动内核。
 *
 * 流程：
 * 1. 解析 minPxPerSec / maxPxPerSec（非法值回退到默认），据此规范化初始 pxPerSec；
 * 2. 建立冻结的初始状态（scrollLeft / scrollTop 恒从 0 起步）与订阅者集合；
 * 3. 返回写入 API：所有写入先算目标值 → 用**目标 pxPerSec**对应的上限钳制 → 变化才提交。
 *
 * 特殊说明：本函数不注册任何 DOM 事件、不持有定时器，输入绑定由输入层负责；
 * 内容尺寸每次调用现算，因此工程数据变化后无需重建内核。
 *
 * @param options 构造参数（尺寸类为惰性 getter，见 ScrollKernelOptions）。
 * @returns 内核实例；钳制与通知去重均已在内部完成。
 */
export function createScrollKernel(options: ScrollKernelOptions): ScrollKernel {
    /**
     * 读取缩放下限。
     *
     * 特殊说明：下限**不是常量**——它随工程长度与视口宽度变化（见
     * `resolveTimelineMinPxPerSec`：工程短于视口时允许缩到 0.5，否则用基础下限）。
     * 若在这里冻结成常量，内核的钳制就会与滚轮缩放解析所用的下限不一致，
     * 表现为「标尺能缩得更小而网格缩不动」。因此支持函数形式，每次读取现算。
     *
     * @returns 当前下限（CSS px/秒）。
     */
    function readMinPxPerSec(): number {
        const raw =
            typeof options.minPxPerSec === "function" ? options.minPxPerSec() : options.minPxPerSec;
        return Number.isFinite(raw)
            ? Math.max(DEFAULT_MIN_PX_PER_SEC, raw as number)
            : DEFAULT_MIN_PX_PER_SEC;
    }

    /**
     * 读取缩放上限（与下限同源的惰性解析，保证上限恒不低于下限）。
     *
     * @returns 当前上限（CSS px/秒）。
     */
    function readMaxPxPerSec(): number {
        const raw =
            typeof options.maxPxPerSec === "function" ? options.maxPxPerSec() : options.maxPxPerSec;
        const min = readMinPxPerSec();
        return Number.isFinite(raw) ? Math.max(min, raw as number) : Number.POSITIVE_INFINITY;
    }

    // 视口真值：冻结对象 + 变更时整体替换（约束 2）。
    let state: TimelineViewportState = Object.freeze({
        scrollLeft: 0,
        scrollTop: 0,
        pxPerSec: sanitizePxPerSec(options.pxPerSec, readMinPxPerSec(), readMaxPxPerSec()),
        rowHeight: Number.isFinite(options.rowHeight) ? Math.max(0, options.rowHeight) : 0,
    });

    const listeners = new Set<() => void>();

    /**
     * 读取宿主视口高度。
     *
     * @returns 量测高度（CSS px）；未量测 / 非法值按 0 处理，使竖直上限退化为
     *          「可整屏滚出」，不会因为宿主尺寸尚未就绪而把内容卡死在 0 位置。
     */
    function viewportHeight(): number {
        const height = options.viewportHeightPx();
        return Number.isFinite(height) ? Math.max(0, height) : 0;
    }

    /**
     * 计算指定缩放下的内容宽度。
     *
     * @param pxPerSec 目标缩放（必须为内核已规范化的值）。
     * @returns 内容宽度（CSS px）；projectSec 非法或为负时按 0 处理。
     */
    function contentWidthFor(pxPerSec: number): number {
        const sec = options.projectSec();
        return Number.isFinite(sec) ? Math.max(0, sec) * pxPerSec : 0;
    }

    /**
     * 计算指定行高下的内容高度。
     *
     * @param rowHeight 行高（CSS px）。
     * @returns 内容高度（CSS px）；trackCount 非法或为负时按 0 处理。
     */
    function contentHeightFor(rowHeight: number): number {
        const count = options.trackCount();
        return Number.isFinite(count) ? Math.max(0, count) * rowHeight : 0;
    }

    /**
     * 计算指定缩放下的水平滚动上限。
     *
     * 规则：上限 = **工程宽度**（不是「工程宽 − 视口宽」），与既有
     * `resolveTimelineScrollRange` 的语义一致（见文件头「水平上限语义」）：
     * - 工程比视口小时，允许把工程右端滚到视口左缘，缩放锚点不会被卡死；
     * - 工程比视口大时，允许继续向工程右侧的空余区域滚动；
     * - 公式在「工程宽 = 视口宽」处连续，缩放时不会出现上限骤降导致的跳变。
     *
     * @param pxPerSec 目标缩放。
     * @returns 水平滚动上限（CSS px，>= 0，等于工程宽度）。
     */
    function maxScrollLeftFor(pxPerSec: number): number {
        return contentWidthFor(pxPerSec);
    }

    /**
     * 计算指定行高下的竖直滚动上限。
     *
     * @param rowHeight 行高（CSS px）。
     * @returns 竖直滚动上限（CSS px，>= 0）。
     */
    function maxScrollTopFor(rowHeight: number): number {
        return Math.max(0, contentHeightFor(rowHeight) - viewportHeight());
    }

    /**
     * 通知所有订阅者。
     *
     * 特殊说明：遍历快照副本，订阅者在回调内退订或新增订阅都不会破坏本轮派发。
     */
    function emit(): void {
        for (const listener of Array.from(listeners)) {
            listener();
        }
    }

    /**
     * 提交一组候选字段（内部写入路径）。
     *
     * 流程：候选值缺省沿用当前值 → 逐字段容差比较 → 全部等价则直接返回（不换引用、
     * 不通知）→ 否则整体替换冻结状态并通知订阅者。
     *
     * 特殊说明：候选值必须**已钳制**，本函数不做钳制。因为水平上限依赖 pxPerSec
     * （上限 = 工程宽度 = projectSec × pxPerSec），setZoom 需要先用目标 pxPerSec 算上限，
     * 再把两个字段一起提交，避免出现「用旧上限钳制新缩放」的中间态。
     *
     * @param next 需要覆盖的字段；未给出的字段沿用当前值。
     */
    function commit(next: Partial<TimelineViewportState>): void {
        const prev = state;
        const scrollLeft = next.scrollLeft ?? prev.scrollLeft;
        const scrollTop = next.scrollTop ?? prev.scrollTop;
        const pxPerSec = next.pxPerSec ?? prev.pxPerSec;
        const rowHeight = next.rowHeight ?? prev.rowHeight;

        if (
            nearlyEqual(scrollLeft, prev.scrollLeft) &&
            nearlyEqual(scrollTop, prev.scrollTop) &&
            // pxPerSec 是比例量，必须用相对容差（见 zoomEquals 注释）。
            zoomEquals(pxPerSec, prev.pxPerSec) &&
            nearlyEqual(rowHeight, prev.rowHeight)
        ) {
            return;
        }

        state = Object.freeze({ scrollLeft, scrollTop, pxPerSec, rowHeight });
        emit();
    }

    return {
        get() {
            return state;
        },

        setScrollLeft(px) {
            if (!Number.isFinite(px)) return;
            // 写入即钳制（约束 1）：读侧永远拿到真值，下游无需再防御。
            commit({ scrollLeft: clamp(px, 0, maxScrollLeftFor(state.pxPerSec)) });
        },

        setScrollTop(px) {
            if (!Number.isFinite(px)) return;
            commit({ scrollTop: clamp(px, 0, maxScrollTopFor(state.rowHeight)) });
        },

        setZoom(pxPerSec, anchorScreenX) {
            // 非法值：不改变状态，但仍返回当前真值（调用方据此同步派生量）。
            if (!Number.isFinite(pxPerSec)) return state.pxPerSec;
            const nextPxPerSec = sanitizePxPerSec(pxPerSec, readMinPxPerSec(), readMaxPxPerSec());
            const anchorX = Number.isFinite(anchorScreenX) ? anchorScreenX : 0;
            // 锚点不变式：缩放前后，视口内 anchorX 处对应的工程时间必须相等。
            // 这里刻意不夹取 anchorX（指针可在视口外），否则锚点会被钉在边缘。
            const anchorSec = (state.scrollLeft + anchorX) / state.pxPerSec;
            const nextScrollLeft = clamp(
                anchorSec * nextPxPerSec - anchorX,
                0,
                maxScrollLeftFor(nextPxPerSec),
            );
            commit({ pxPerSec: nextPxPerSec, scrollLeft: nextScrollLeft });
            // 返回**实际生效**值：钳制可能改变请求值（下限随工程长度变化），
            // 调用方必须用这个值同步 React 侧派生量，否则标尺会显示成未钳制的
            // 请求值（表现为「标尺能缩得比网格更小」）。
            return nextPxPerSec;
        },

        setViewport(next) {
            const prev = state;
            const nextPxPerSec =
                next.pxPerSec === undefined
                    ? prev.pxPerSec
                    : sanitizePxPerSec(next.pxPerSec, readMinPxPerSec(), readMaxPxPerSec());
            // 非有限值沿用当前值：外部（React 侧派生量）偶尔会传出 NaN，直接 clamp
            // 会把 NaN 写进状态并污染后续所有换算。
            const nextScrollLeft = Number.isFinite(next.scrollLeft)
                ? (next.scrollLeft as number)
                : prev.scrollLeft;
            commit({
                pxPerSec: nextPxPerSec,
                scrollLeft: clamp(nextScrollLeft, 0, maxScrollLeftFor(nextPxPerSec)),
            });
            return state;
        },

        subscribe(listener) {
            listeners.add(listener);
            return () => {
                listeners.delete(listener);
            };
        },

        contentWidthPx() {
            return contentWidthFor(state.pxPerSec);
        },

        contentHeightPx() {
            return contentHeightFor(state.rowHeight);
        },

        maxScrollLeft() {
            return maxScrollLeftFor(state.pxPerSec);
        },

        maxScrollTop() {
            return maxScrollTopFor(state.rowHeight);
        },

        setRowHeight(px: number) {
            const next = Number.isFinite(px) ? Math.max(1, px) : state.rowHeight;
            if (nearlyEqual(next, state.rowHeight)) return;
            // 行高变化改变内容高度 → 顺带重新钳制竖直位置（见接口注释）。
            commit({
                rowHeight: next,
                scrollTop: clamp(state.scrollTop, 0, maxScrollTopFor(next)),
            });
        },

        reclamp() {
            // 外部边界（resize / 工程变短 / 删轨）变化后，已提交的位置可能越界；
            // 用当前边界重算并提交，位置真正变化时经 commit 正常通知订阅者。
            commit({
                scrollLeft: clamp(state.scrollLeft, 0, maxScrollLeftFor(state.pxPerSec)),
                scrollTop: clamp(state.scrollTop, 0, maxScrollTopFor(state.rowHeight)),
            });
        },
    };
}
