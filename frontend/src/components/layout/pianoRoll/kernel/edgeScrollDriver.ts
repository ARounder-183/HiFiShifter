/**
 * 参数编辑器内核 · 拖拽边缘自动滚屏驱动
 *
 * 【主要内容】
 * `createEdgeScrollDriver()`：把「指针停在视口边缘」变成**持续的**滚动，并在滚动
 * 之后回调调用方，让它按滚动后的投影重算笔画 / 选区。
 *
 * 【为什么需要它（而不是在 pointermove 里滚一下就算）】
 * 1. **指针停住时也要滚**。`pointermove` 只在指针**移动**时派发；用户把笔画拖到
 *    视口边缘后往往会停在那里等视图滚过来 —— 只靠事件驱动，此刻滚动会完全停止，
 *    而这正是绘制类工具最需要它的时候（笔画断了就只能松手重画，一条连续曲线被
 *    切成两段，还多打一个撤销点）。
 * 2. **滚动后必须重算**。`scrollLeft` 变了，同一个 `clientX` 对应的帧号也随之改变。
 *    不回调重算，画面上的笔画就会在滚动处跳变（视图过去了、线没跟上）。
 * 3. **速度必须与帧率 / 事件频率解耦**。见 `shared/edgeAutoScroll` 的模块头说明：
 *    按事件计步的旧实现在高轮询率鼠标上快一个数量级。
 *
 * 【为什么可以单测】宿主能力（取边界、读写滚动位置、上界、时间源、rAF）全部由
 * 调用方注入，本模块不直接碰 DOM。因此"停在边缘持续滚"、"松手后不再滚"、
 * "滚动量被上界钳住"这些**只能靠手拖复现**的行为都有用例钉住。
 *
 * 【与其他模块的关系】
 * - 依赖：`shared/edgeAutoScroll`（纯几何）。
 * - 上游：`usePianoRollInteractions` 的绘制类分支（freehand / line / vibrato）
 *   与选择工具分支。
 * - 下游：宿主回调把新的 `scrollLeft` 采纳进渲染内核（面板的 `syncScrollLeft`）。
 */

import { resolveEdgeScrollDeltaPx, type EdgeScrollBounds } from "../../shared/edgeAutoScroll";
import { EDGE_SCROLL_MAX_SPEED_PX_PER_SEC } from "./dragArithmetic";

/** 判定"确实滚动了"的最小像素量：低于它视为没动（避免浮点抖动触发重算）。 */
const MIN_STEP_PX = 0.01;

/**
 * 两次滚动之间超过这个间隔就视为**新手势**，按默认帧时长（1/60 秒）起步。
 *
 * 【为什么需要】驱动是长生命周期对象（跨手势复用），指针停着不动的间隙会被算成
 * 一个巨大的"本帧时长"。纯函数虽然把时长夹到 100ms，但 100ms × 最大速度仍是一步
 * 上百像素的跳变 —— 新手势第一下就跳一下，比不滚还糟。
 */
const GESTURE_GAP_RESET_MS = 200;

/** 默认帧时长（60Hz）：新手势第一步用它。 */
const DEFAULT_FRAME_MS = 1000 / 60;

/** 指针与视口的横向边界。 */
export type { EdgeScrollBounds };

/** 驱动所需的外部能力（全部注入，便于单测）。 */
export interface EdgeScrollDriverOptions {
    /** 视口左右边界；容器尚未挂载时返回 `null`（本帧不滚）。 */
    getBounds: () => EdgeScrollBounds | null;
    /** 当前原生滚动位置。 */
    getScrollLeft: () => number;
    /** 写入原生滚动位置（调用方同时负责采纳进渲染内核）。 */
    setScrollLeft: (next: number) => void;
    /** 滚动位置上界（每次滚动前重新求值：缩放 / 工程长度 / 同步偏移都会变）。 */
    getMaxScrollLeft: () => number;
    /**
     * 本帧确实滚动了。
     *
     * 【为什么带 clientX】调用方要用**同一个指针位置**重新投影：视图滚了，指针
     * 屏幕坐标没变，但它对应的帧号变了 —— 绘制类工具据此补一笔 / 重算端点。
     */
    onScrolled?: (clientX: number) => void;
    /** 最大滚屏速度（CSS px/秒）。默认取参数编辑器的常量（见 `dragArithmetic`）。 */
    maxSpeedPxPerSec?: number;
    /** 时间源（毫秒）。默认 `performance.now`。 */
    now?: () => number;
    /** 请求下一帧。默认 `requestAnimationFrame`。 */
    requestFrame?: (cb: () => void) => number;
    /** 取消挂起帧。默认 `cancelAnimationFrame`。 */
    cancelFrame?: (id: number) => void;
}

/** 拖拽期间的边缘自动滚屏驱动。 */
export interface EdgeScrollDriver {
    /**
     * 按给定指针位置滚**一帧**，返回本帧是否真的滚动了。
     *
     * 选择工具用它（指针到哪选到哪，停在边缘不动没有意义，因此由 pointermove
     * 事件驱动即可）。时钟是共享的：连续调用按真实间隔累计，速度与事件频率无关。
     */
    step: (clientX: number) => boolean;
    /**
     * 记录指针位置并维持 rAF 循环 —— 指针停在边缘不动时也继续滚。
     *
     * 绘制类工具用它（笔画必须跟着视图继续延伸）。
     */
    track: (clientX: number) => void;
    /**
     * 停止 rAF 循环并复位时钟。
     *
     * **手势收尾（pointerup / pointercancel / blur）必须调用**，否则松手后视图会
     * 一直滚下去。幂等。
     */
    stop: () => void;
    /** 是否正在持续滚动（供断言与调试）。 */
    isRunning: () => boolean;
}

/**
 * 创建驱动。
 *
 * 生命周期约定：**每个手势一个**（时钟随构造复位，新手势第一步因此必然按 1/60 秒
 * 起步），手势收尾调 `stop()`。跨手势复用会让两次拖拽之间的空闲被算进"本帧时长"，
 * 新手势第一下就跳一大步。
 */
export function createEdgeScrollDriver(options: EdgeScrollDriverOptions): EdgeScrollDriver {
    const now = options.now ?? (() => performance.now());
    const requestFrame = options.requestFrame ?? ((cb: () => void) => requestAnimationFrame(cb));
    const cancelFrame = options.cancelFrame ?? ((id: number) => cancelAnimationFrame(id));

    let rafId: number | null = null;
    /** 最后一次已知的指针 x（rAF 循环每帧用它重新求步长）。 */
    let lastClientX: number | null = null;
    /** 上一次滚动发生的时刻；`null` 表示尚未滚过（或已被 stop 复位）。 */
    let lastStepAtMs: number | null = null;
    const maxSpeedPxPerSec = options.maxSpeedPxPerSec ?? EDGE_SCROLL_MAX_SPEED_PX_PER_SEC;

    /** 本帧应使用的时长：新手势 / 长间隔按 1/60 秒起步，否则用真实间隔。 */
    const frameMsFor = (atMs: number): number => {
        if (lastStepAtMs === null) return DEFAULT_FRAME_MS;
        const elapsed = atMs - lastStepAtMs;
        if (!Number.isFinite(elapsed) || elapsed <= 0 || elapsed > GESTURE_GAP_RESET_MS) {
            return DEFAULT_FRAME_MS;
        }
        return elapsed;
    };

    const step = (clientX: number): boolean => {
        lastClientX = clientX;
        const bounds = options.getBounds();
        if (!bounds) return false;

        const atMs = now();
        const frameMs = frameMsFor(atMs);
        const deltaPx = resolveEdgeScrollDeltaPx({
            clientX,
            leftPx: bounds.left,
            rightPx: bounds.right,
            frameMs,
            maxSpeedPxPerSec,
        });
        // 记账放在"是否在带内"判定之后、位置写入之前：即使这一帧因为已到上界而
        // 没有位移，间隔也必须被消费掉 —— 否则下一步会把这段时间又算一遍。
        lastStepAtMs = atMs;
        if (Math.abs(deltaPx) < MIN_STEP_PX) return false;

        const current = options.getScrollLeft();
        const max = Math.max(0, options.getMaxScrollLeft());
        const next = Math.min(max, Math.max(0, current + deltaPx));
        if (Math.abs(next - current) < MIN_STEP_PX) return false;

        options.setScrollLeft(next);
        options.onScrolled?.(clientX);
        return true;
    };

    /** 指针是否仍落在边缘带内（决定 rAF 循环要不要继续）。 */
    const pointerInBand = (clientX: number): boolean => {
        const bounds = options.getBounds();
        if (!bounds) return false;
        return (
            resolveEdgeScrollDeltaPx({
                clientX,
                leftPx: bounds.left,
                rightPx: bounds.right,
                frameMs: DEFAULT_FRAME_MS,
                maxSpeedPxPerSec,
            }) !== 0
        );
    };

    const tick = () => {
        rafId = null;
        const clientX = lastClientX;
        if (clientX === null) return;
        step(clientX);
        // 指针已离开边缘带 → 结束循环。指针仍在带内则继续：即便这一刻已到滚动
        // 上界（step 无位移），上界也可能随缩放 / 编辑而变化，保持循环比"停下来
        // 等下一次 pointermove"更符合用户预期（他还在边缘按着）。
        if (pointerInBand(clientX)) {
            rafId = requestFrame(tick);
        }
    };

    const track = (clientX: number): void => {
        lastClientX = clientX;
        // 只记录位置并保证循环在跑：**真正的滚动在 rAF 回调里做**（每帧至多一次
        // 位置写入 + 重绘）。绘制类工具本身就把指针事件合帧到每渲染帧一次，这里
        // 逐事件写 `scrollLeft` 会在高采样率数位笔下变成几百次/秒的无谓开销，
        // 与那条路径的既有设计相悖。代价是首次滚动最多晚一帧（≤16ms），无感。
        if (rafId === null && pointerInBand(clientX)) {
            rafId = requestFrame(tick);
        }
    };

    const stop = (): void => {
        if (rafId !== null) {
            cancelFrame(rafId);
            rafId = null;
        }
        lastClientX = null;
        lastStepAtMs = null;
    };

    return {
        step,
        track,
        stop,
        isRunning: () => rafId !== null,
    };
}
