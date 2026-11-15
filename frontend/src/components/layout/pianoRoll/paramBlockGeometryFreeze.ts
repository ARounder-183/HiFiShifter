/**
 * 手势期间的块几何来源。
 *
 * ## 问题
 *
 * 参数块是参数线的**分段**：它的几何是 `(clips, editSet, framePeriodMs)` 的派生值，
 * 而 `clips` 在时间轴手势期间只做 **Redux 乐观更新**（后端要到松手才收到提交）。
 * 参数线却相反：它来自后端 `get_param_frames`，要等 `paramsEpoch` 递增（= 落库）
 * 才会重取。
 *
 * 于是整个手势期间"块跟着 Clip Take 走、参数线原地不动"——块从它标注的那条曲线上
 * 滑走了。Slip / 移动 / 拉伸三种手势都会这样，越拖越明显。
 *
 * ## 本模块的职责（只有一件事）
 *
 * 决定块层这一帧用哪一份 clip 几何：**手势期间用手势起点的快照，其余时间用实时值**。
 *
 * 为什么不是"让参数线也跟着走"：那需要在前端重算整条曲线（源域仿射 + Loop 回绕 +
 * 能量域融合），等于把后端的装配逻辑写第二遍 —— 响度那条（`loudnessGeometryWarp`）
 * 已经证明代价，而且它只作用于波形的幅度乘数，不是参数线本身。冻结块层是零数学的
 * 等价解，也与用户给出的判据一致："拖拽过程中不渲染参数块的变化，松手之后再正确渲染"。
 *
 * ## 为什么快照必须是"手势起点"
 *
 * 交互锁（`_interactionLockCount`）在**首个真实位移帧**才 +1，而同一帧里乐观写入
 * 已经派发（React 合并成一次渲染）。所以"看到 `lock > 0` 的那一帧"里 `s.clips` 已经
 * 带着第一帧的位移，读它就会漏掉这一帧。快照由 `clipGeometryPreviewBus` 在任何
 * 乐观写入之前建立（发布方是 `beginKernelGestureInteraction`）。
 *
 * ## 边界
 *
 * - 只冻结**块层**。自动音节分析、波形、时间轴都用实时 clip —— 它们描述的是素材
 *   本身，与块的分段无关。
 * - 参数块自身的手势（拖动 / 伸缩 / 过渡）不经过这里：它改的是 `editSet`，几何
 *   输入不变。
 * - 松手后快照由发布方清空，块层在**下一次渲染**就跟上权威几何；这一帧与 Clip
 *   图层换新同步，不会出现"先跳乐观位置、再跳权威位置"的两次跳变。
 */

/**
 * 块层这一帧该用哪一份 clip 几何。
 *
 * @param live 实时 clips（Redux 乐观更新中）。
 * @param frozen 手势起点的快照；`null` 表示没有进行中的几何手势。
 * @returns 冻结期间恒等于 `frozen`（**引用稳定**，因此块视图不会重算），否则 `live`。
 */
export function resolveParamBlockGeometryClips<T>(
    live: readonly T[],
    frozen: readonly T[] | null,
): readonly T[] {
    return frozen ?? live;
}
