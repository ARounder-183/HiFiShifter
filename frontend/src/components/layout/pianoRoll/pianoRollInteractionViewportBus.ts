/**
 * 参数编辑器“交互视口”总线。
 *
 * 与只服务 sticky 图层的 `pianoRollViewportBus` 分开：交互重放需要同时知道
 * 水平投影与竖直值域（center / span / 高度）。拖拽中任一字段变化都必须按
 * 最近指针位置重放一次 pointermove，否则块/选区会停在旧内容坐标。
 *
 * 【它不做什么】不判「该不该重放」（那是 `viewportReplay` 的纯函数），
 * 不读 React state，不排帧。
 *
 * 【为什么发布是**微任务**而不是同步】值轴的 `apply()` 在滚动帧内部调用发布，
 * 而订阅者（`viewportReplay`）会**同步**合成一个 `pointermove`。手势的 `onMove`
 * 收到它就会重入 `edgeScrollDriver.track()` —— 那正是「每帧多排一个 rAF」的引信
 * （见 `kernel/edgeScrollDriver` 的 `ticking` 说明）。改成微任务后，重入发生在
 * 本帧 tick 返回之后，帧令牌已经重新就位；同时同一帧内的多次发布被合并成一次。
 *
 * 【为什么 get() 仍然同步】新订阅者必须能立刻拿到当前值。只有**通知**被推迟，
 * 状态本身不推迟。
 *
 * 【违反边界的症状】
 * - 发布改回同步：拖得越久越卡（每帧多排一个 rAF，见 driver 模块头）；
 * - 去掉「值不变不广播」：同一帧里三处提交各触发一次重放，采样跑三遍；
 * - 把 `get()` 也改成异步：挂载瞬间读到 EMPTY，拖拽中的块闪一下。
 */
export interface PianoRollInteractionViewport {
    readonly scrollLeftPx: number;
    readonly pxPerSec: number;
    readonly viewportWidthPx: number;
    readonly viewportHeightPx: number;
    readonly param: string;
    readonly valueCenter: number;
    readonly valueSpan: number;
    readonly dpr: number;
}

const EMPTY: PianoRollInteractionViewport = {
    scrollLeftPx: 0,
    pxPerSec: 150,
    viewportWidthPx: 1,
    viewportHeightPx: 1,
    param: "pitch",
    valueCenter: 60,
    valueSpan: 24,
    dpr: 1,
};

let current = EMPTY;
const listeners = new Set<(next: PianoRollInteractionViewport) => void>();
/** 待投递的最新值；同一微任务内的多次发布只保留最后一个。 */
let pendingDelivery: PianoRollInteractionViewport | null = null;
let deliveryScheduled = false;

function deliver(): void {
    deliveryScheduled = false;
    const next = pendingDelivery;
    pendingDelivery = null;
    if (next === null) return;
    for (const listener of listeners) listener(next);
}

export const pianoRollInteractionViewportBus = {
    /**
     * 发布一次交互视口。**同一帧内的多次发布合并成一次**，且值不变不广播。
     *
     * 状态（`get()`）立刻生效，通知推迟到下一个微任务 —— 见模块头。
     */
    publish(next: PianoRollInteractionViewport): void {
        if (isSameInteractionViewport(current, next)) return;
        current = next;
        pendingDelivery = next;
        if (deliveryScheduled) return;
        deliveryScheduled = true;
        queueMicrotask(deliver);
    },
    get(): PianoRollInteractionViewport {
        return current;
    },
    subscribe(fn: (next: PianoRollInteractionViewport) => void): () => void {
        listeners.add(fn);
        return () => {
            listeners.delete(fn);
        };
    },
};

export function isSameInteractionViewport(
    a: PianoRollInteractionViewport | null,
    b: PianoRollInteractionViewport,
): boolean {
    return (
        a !== null &&
        a.scrollLeftPx === b.scrollLeftPx &&
        a.pxPerSec === b.pxPerSec &&
        a.viewportWidthPx === b.viewportWidthPx &&
        a.viewportHeightPx === b.viewportHeightPx &&
        a.param === b.param &&
        a.valueCenter === b.valueCenter &&
        a.valueSpan === b.valueSpan &&
        a.dpr === b.dpr
    );
}
