/**
 * 响度快照取数的「提交期闸门」。
 *
 * ## 为什么需要它
 *
 * 参数编辑器波形画的是可听结果：源峰值 × clip增益×淡化 × volume(t) × dyn增益(t)，
 * 其中 dyn增益 = 目标电平 / 原声基线，而基线与用户曲线都由后端按 clip 几何组装。
 * 时间轴「拉伸 / 缩短」提交时，后端要做两件事：
 *
 * 1. 写入新几何（setClipsStateBulkRemote）；
 * 2. 把该轨道锁定的参数线按新范围重采样（stretchLinkedParams）。
 *
 * 第 1 步的 fulfilled handler 必然 applyTimelineState() ⇒ paramsEpoch++ ⇒
 * useLoudnessCurves 立刻发起一次取数。这次取数发生在第 2 步之前，带回的是
 * 「新几何的基线 × 旧范围的用户曲线」这一自相矛盾组合。
 *
 * ## 跨窗口
 *
 * 状态住在 interaction.loudnessFetchHeld，经 detachBridge 复制到卫星窗口。
 * 本模块的 held 是**派生缓存**：store 订阅器在每次状态变化后调用
 * syncLoudnessFetchGate；hold/release 只负责派发 action。这样
 * useLoudnessCurves 在 Promise 回调 / flushPending 等非 React 时机仍可同步读取。
 *
 * ## 契约
 *
 * - 每次 holdLoudnessFetch 必须在 finally 里配对 releaseLoudnessFetch —— 漏放会
 *   让取数永久停摆。
 * - 释放必须紧跟一次 bumpParamsEpoch()（或等价取数触发）。
 */

let held = false;
let dispatchHeld: ((held: boolean) => void) | null = null;

/**
 * 由 app/store.ts 在创建 store 后注入。只允许 store 层调用；组件直接调用会把
 * 状态写进错误的窗口实例。
 */
export function installLoudnessFetchGateDispatcher(
    dispatch: ((held: boolean) => void) | null,
): void {
    dispatchHeld = dispatch;
}

/** 由 app/store.ts 的订阅器调用：把 Redux 权威值同步到同步读取缓存。 */
export function syncLoudnessFetchGate(value: boolean): void {
    held = value;
}

/** 合上闸门：提交链起点（早于落库派发）调用。 */
export function holdLoudnessFetch(): void {
    held = true;
    dispatchHeld?.(true);
}

/** 打开闸门：曲线改写完成后调用（必须紧跟一次 bumpParamsEpoch()）。 */
export function releaseLoudnessFetch(): void {
    held = false;
    dispatchHeld?.(false);
}

/** 闸门是否合上（useLoudnessCurves 在发起取数与落地结果前读取）。 */
export function isLoudnessFetchHeld(): boolean {
    return held;
}
