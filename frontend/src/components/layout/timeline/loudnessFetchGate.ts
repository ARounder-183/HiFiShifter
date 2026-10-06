/**
 * 响度快照取数的「提交期闸门」（跨组件模块级总线，与 `clipGeometryPreviewBus` 同构）。
 *
 * ## 为什么需要它
 *
 * 参数编辑器波形画的是可听结果：`源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)`，
 * 其中 `dyn增益 = 目标电平 / 原声基线`，而**基线与用户曲线都由后端按 clip 几何组装**。
 * 时间轴「拉伸 / 缩短」提交时，后端要做**两件**事：
 *
 * 1. 写入新几何（`setClipsStateBulkRemote`）；
 * 2. 把该轨道锁定的参数线按新范围重采样（`stretchLinkedParams`）。
 *
 * 而第 1 步的 fulfilled handler 必然 `applyTimelineState()` ⇒ `paramsEpoch++`
 * ⇒ `useLoudnessCurves` **立刻发起一次取数**。这次取数发生在第 2 步**之前**，
 * 于是它带回的是「**新几何的基线 × 旧范围**的用户曲线」这一**自相矛盾**的组合。
 *
 * 若让它落地：收尾判据（`baselineKey` 变了 ⇔ 基线反映新几何）会判定"权威数据已到"，
 * 撤下拖拽期的几何映射，波形就此停在「新几何 × 旧曲线」上 —— 直到用户再做一次别的
 * 操作触发取数才恢复。这正是用户报告的「松手后波形完全对不上、要再做一次操作才好」。
 *
 * 组拉伸路径此前靠"改写后再 `bumpParamsEpoch()`"补了第二次取数，症状因此只是**短暂**
 * 错乱；单 clip 路径连这句都没有，于是**永久**错乱。
 *
 * ## 闸门做什么
 *
 * 在提交链的**最前面**（早于落库派发）合上，`useLoudnessCurves` 在此期间
 * **不发起取数、也不落地任何在飞结果**（在飞的那次可能由后端按新几何组装，
 * 仍是"新几何 × 旧曲线"，必须丢弃）。曲线改写完成后打开闸门，调用方随即
 * `bumpParamsEpoch()` 触发**唯一一次**权威取数（新几何 + 改写后的曲线）。
 *
 * 于是收尾判据"基线键变了"重新变得**充分**：能落地的那份快照必然同时包含
 * 新几何与改写后的曲线（`loudnessGeometryWarp` 文件头的映射因此始终正确）。
 *
 * ## 契约
 *
 * - 每次 {@link holdLoudnessFetch} 必须在 `finally` 里配对 {@link releaseLoudnessFetch}
 *   —— 漏放会让取数永久停摆（波形再也收不到新数据）。
 * - 释放**必须**紧跟一次 `bumpParamsEpoch()`（或等价的取数触发），否则释放本身
 *   不会发起取数（effect 依赖的是 epoch / refreshToken，不是本闸门）。
 */

let held = false;

/** 合上闸门：提交链起点（早于落库派发）调用。 */
export function holdLoudnessFetch(): void {
    held = true;
}

/** 打开闸门：曲线改写完成后调用（必须紧跟一次 `bumpParamsEpoch()`）。 */
export function releaseLoudnessFetch(): void {
    held = false;
}

/** 闸门是否合上（`useLoudnessCurves` 在发起取数与落地结果前读取）。 */
export function isLoudnessFetchHeld(): boolean {
    return held;
}
