/**
 * 时间轴渲染内核 · 实例缓冲容量策略
 *
 * 【主要内容】
 * 计算实例缓冲（`Float32Array`）在需要容纳 `neededFloats` 时的目标容量：
 * 容量足够则保持不变，不足则倍增（`max(needed, current × 2)`）。
 *
 * 【作用】
 * 滚动 / 缩放过程中可见 clip 数会小幅波动（窗口化边界），若每次精确分配，
 * GC 压力会回到「每帧一次大分配」；倍增策略把稳态分配次数降到 O(log n)。
 * 这是渲染热路径的容量契约，单独成模块以便单测。
 *
 * 【与其他模块的关系】
 * - 上游：`scene/clipInstances`（clip 实例）与后续的网格/字形实例共用同一策略；
 * - 下游：GL 渲染器据此决定是否重新分配 `Float32Array` 并上传。
 * - 独立性：纯函数，不依赖 DOM / WebGL。
 */

/**
 * 计算实例缓冲的目标容量（float 数）。
 *
 * 流程：非法输入归零 → 容量已足够则原样返回 → 否则取 `max(needed, current × 2)`。
 *
 * 特殊说明：倍增而非精确分配，且新容量至少为 `needed`（保证一次调用后一定能容纳）；
 * `current = 0` 时直接返回 `needed`（首次分配不做无意义的翻倍）。
 *
 * @param currentFloats 当前缓冲容量（float 数）。
 * @param neededFloats 本次需要容纳的 float 数。
 * @returns 目标容量（float 数）；不会小于 `currentFloats`。
 */
export function resolveBufferFloats(currentFloats: number, neededFloats: number): number {
    const current =
        Number.isFinite(currentFloats) && currentFloats > 0 ? Math.floor(currentFloats) : 0;
    const needed = Number.isFinite(neededFloats) && neededFloats > 0 ? Math.ceil(neededFloats) : 0;
    if (needed <= current) return current;
    return Math.max(needed, current * 2);
}
