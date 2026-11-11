/**
 * 参数块选择模式与参数线选择共用的背景拖动语义。
 *
 * 左键（无修饰键）替换选区；右键或 Ctrl/Cmd 左键在已有选区上追加。
 * 指针到帧的换算是调用方职责 —— 它必须使用与渲染同源的投影，
 * 这里只负责“由两个连续帧坐标得到选区”。
 */
import { addPointerRange, selectionFromPointers, type ParamSelection } from "./paramSelection";

export type RangeDragMode = "replace" | "add";

export function selectionForRangeDrag(args: {
    readonly base: ParamSelection | null;
    readonly anchorFrame: number;
    readonly pointerFrame: number;
    readonly mode: RangeDragMode;
}): ParamSelection | null {
    return args.mode === "add"
        ? addPointerRange(args.base, args.anchorFrame, args.pointerFrame)
        : selectionFromPointers(args.anchorFrame, args.pointerFrame);
}
