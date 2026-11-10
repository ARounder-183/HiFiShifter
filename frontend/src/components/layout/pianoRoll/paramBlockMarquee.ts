/**
 * 块工具的纯工具函数。
 *
 * 【为什么只剩一个函数】块的多选已经并入**参数选区**（见 `paramSelectionDrag` 与
 * `paramBlockSelection`）：框选按时间范围命中块，而不是按矩形。原先这里那套
 * 基于像素矩形的 marquee 解析在生产代码里没有任何调用者（只有测试在用），
 * 与面板里的真实实现并存只会制造"两套语义"的错觉，因此删除。
 */

/** `~`：切换当前参数的块工具与绘制工具。 */
export function toggleParamBlockEditingTool(toolMode: string): "block" | "draw" {
    return toolMode === "block" ? "draw" : "block";
}
