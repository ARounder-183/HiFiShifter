export interface ParamBlockMarqueeHitRect {
    readonly id: string;
    readonly startSec: number;
    readonly endSec: number;
    readonly topPx: number;
    readonly heightPx: number;
}

export interface ParamBlockMarqueeRect {
    readonly leftPx: number;
    readonly rightPx: number;
    readonly topPx: number;
    readonly bottomPx: number;
}

export type ParamBlockRightPointerEnd =
    { readonly kind: "marquee-selection" } | { readonly kind: "context-menu" };

/** 右键拖过死区才是框选；原地右键用于打开块菜单。 */
export function resolveParamBlockRightPointerEnd(args: {
    readonly marqueeActive: boolean;
}): ParamBlockRightPointerEnd {
    return args.marqueeActive ? { kind: "marquee-selection" } : { kind: "context-menu" };
}

/** 块工具里的右键拖拽可从空白处开始框选。 */
export function shouldStartParamBlockRightDragMarquee(args: {
    readonly blockToolActive: boolean;
    readonly contextMenu: boolean;
}): boolean {
    return args.blockToolActive && args.contextMenu;
}

/** `~`：块工具与音高选区工具双向切换。 */
export function toggleParamBlockEditingTool(toolMode: string): "block" | "draw" {
    return toolMode === "block" ? "draw" : "block";
}

/** 右键原地点击：点在多选内保留整组，点在组外把选择收敛到该块。 */
export function resolveParamBlockRightClickSelection(args: {
    readonly selectedIds: readonly string[];
    readonly hitId: string;
}): string[] {
    return args.selectedIds.includes(args.hitId) ? [...args.selectedIds] : [args.hitId];
}

/**
 * 用宿主本帧的块矩形解析框选结果。
 *
 * 横向秒数必须和同一帧的滚动偏移一起投影，不能只乘 px/s；否则视图滚动后
 * 鼠标框住的是一组块，最终选中的却是另一组。重复实例按 id 去重并按时间稳定排序。
 */
export function resolveParamBlockMarqueeSelection(args: {
    readonly rects: readonly ParamBlockMarqueeHitRect[];
    readonly marquee: ParamBlockMarqueeRect;
    readonly pxPerSec: number;
    readonly scrollLeftPx: number;
}): string[] {
    const pxPerSec = Math.max(1, args.pxPerSec);
    const left = Math.min(args.marquee.leftPx, args.marquee.rightPx);
    const right = Math.max(args.marquee.leftPx, args.marquee.rightPx);
    const top = Math.min(args.marquee.topPx, args.marquee.bottomPx);
    const bottom = Math.max(args.marquee.topPx, args.marquee.bottomPx);
    const selected = new Map<string, { readonly startSec: number; readonly endSec: number }>();

    for (const rect of args.rects) {
        if (
            !Number.isFinite(rect.startSec) ||
            !Number.isFinite(rect.endSec) ||
            !Number.isFinite(rect.topPx) ||
            !Number.isFinite(rect.heightPx)
        ) {
            continue;
        }
        const startSec = Math.min(rect.startSec, rect.endSec);
        const endSec = Math.max(rect.startSec, rect.endSec);
        const startPx = startSec * pxPerSec - args.scrollLeftPx;
        const endPx = endSec * pxPerSec - args.scrollLeftPx;
        const rectTop = rect.topPx;
        const rectBottom = rect.topPx + Math.max(1, rect.heightPx);
        if (rectBottom < top || rectTop > bottom) continue;
        if (endPx < left || startPx > right) continue;
        selected.set(rect.id, { startSec, endSec });
    }

    return [...selected.entries()]
        .sort(([leftId, leftSpan], [rightId, rightSpan]) => {
            const byStart = leftSpan.startSec - rightSpan.startSec;
            if (Math.abs(byStart) > 1e-9) return byStart;
            const byEnd = leftSpan.endSec - rightSpan.endSec;
            if (Math.abs(byEnd) > 1e-9) return byEnd;
            return leftId < rightId ? -1 : leftId > rightId ? 1 : 0;
        })
        .map(([id]) => id);
}
