/**
 * 参数编辑器“交互视口”总线。
 *
 * 与只服务 sticky 图层的 `pianoRollViewportBus` 分开：交互重放需要同时知道
 * 水平投影与竖直值域（center / span / 高度）。拖拽中任一字段变化都必须按
 * 最近指针位置重放一次 pointermove，否则块/选区会停在旧内容坐标。
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

export const pianoRollInteractionViewportBus = {
    emit(next: PianoRollInteractionViewport): void {
        current = next;
        for (const listener of listeners) listener(next);
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
