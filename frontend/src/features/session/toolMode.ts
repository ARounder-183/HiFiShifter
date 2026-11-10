/** 工具模式切换的纯函数。 */
export type ParamDrawToolMode = "draw" | "line" | "vibrato";

/** `~`：在当前参数的块工具与上一次的绘制工具之间切换。 */
export function toggleParamBlockEditingTool(
    toolMode: string,
    drawToolMode: string = "draw",
): "block" | ParamDrawToolMode {
    if (toolMode === "block") {
        return drawToolMode === "line" || drawToolMode === "vibrato" ? drawToolMode : "draw";
    }
    return "block";
}
