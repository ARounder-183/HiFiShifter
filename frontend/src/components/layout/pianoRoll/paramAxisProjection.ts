import { isPanParam, resolvePanAxisDirection, type PanAxisDirections } from "./paramPan";

/** 当前参数纵轴是否上下翻转（目前只有 pan 支持）。 */
export function isAxisInverted(
    param: string,
    directions: PanAxisDirections | null | undefined,
): boolean {
    return isPanParam(param) && resolvePanAxisDirection(directions, param) === "left-top";
}

/** 视口内归一化 t（0=顶、1=底）→ 参数值。 */
export function valueAtViewportT(args: {
    readonly param: string;
    readonly directions: PanAxisDirections | null | undefined;
    readonly t: number;
    readonly min: number;
    readonly span: number;
}): number {
    const inverted = isAxisInverted(args.param, args.directions);
    return args.min + (inverted ? args.t : 1 - args.t) * args.span;
}

/** 以指针处值为锚点缩放后的视口中心。 */
export function centerForZoomAnchor(args: {
    readonly param: string;
    readonly directions: PanAxisDirections | null | undefined;
    readonly valueAtPointer: number;
    readonly t: number;
    readonly nextSpan: number;
}): number {
    const inverted = isAxisInverted(args.param, args.directions);
    return inverted
        ? args.valueAtPointer + (0.5 - args.t) * args.nextSpan
        : args.valueAtPointer - (0.5 - args.t) * args.nextSpan;
}

/** 纵向拖拽的视口中心增量；翻转轴要取反，保持“跟手”。 */
export function centerForPanDelta(args: {
    readonly param: string;
    readonly directions: PanAxisDirections | null | undefined;
    readonly center: number;
    readonly deltaCenter: number;
}): number {
    return isAxisInverted(args.param, args.directions)
        ? args.center - args.deltaCenter
        : args.center + args.deltaCenter;
}
