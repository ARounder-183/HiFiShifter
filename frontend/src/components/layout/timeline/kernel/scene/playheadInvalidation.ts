/**
 * 时间轴渲染内核 · 播放头标脏判定（纯函数）
 *
 * 【主要内容】
 * 判定「当前播放头位置」相对「上一次绘制用的位置」是否值得请求一次重绘。
 *
 * 【作用：修的是什么】
 * 内核的渲染循环是**纯脏标记驱动**的（`renderKernel/renderLoop`：`start()` 不绘制，
 * 也没有常驻 rAF）。而播放头位置的真值由面板的视觉插值 ref 持有、经数据镜像
 * 传进内核——**镜像变化不会自动标脏**。于是点标尺 seek 后：标尺播放头（React
 * 声明式渲染）动了，内核自绘的轨道区播放头却永不重绘，冻结在旧位置。
 *
 * 修复要在「播放头变了」时主动请求重绘，本模块就是这个判定的唯一来源。
 *
 * 【与其他模块的关系】
 * - 上游：`timelineKernelHost` 的帧提交与外部重绘入口。
 * - 独立性：纯函数，无 DOM / WebGL / React 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 用**绝对容差**而不是严格相等：播放头位置来自 `performance.now()` 外推，
 *    同一逻辑位置在两次读取间会有极小浮点抖动；严格相等会每帧都判定"变了"，
 *    把空闲状态也变成持续重绘。
 * 2. 上一次为 `NaN` 视为"尚未绘制过"，必须返回真（首帧不能跳过）。
 * 3. 当前值非有限时返回假：宁可不动，也不能把 NaN 写进 `style.transform`
 *    （NaN 会让该元素整层失效，且不会报错）。
 */

/** 位置比较的绝对容差（秒）。远小于一个像素对应的时间，又足以吸收浮点抖动。 */
const PLAYHEAD_EPSILON_SEC = 1e-6;

/**
 * 播放头位置变化是否需要重绘。
 *
 * 流程：当前值有限性校验 → 上次值哨兵判定 → 绝对容差比较。
 *
 * @param nextSec 本帧的播放头位置（秒）。
 * @param lastSec 上一次绘制用的位置（秒）；`NaN` 表示尚未绘制过。
 * @returns 需要请求一次重绘时为 true。
 */
export function shouldRepaintForPlayhead(nextSec: number, lastSec: number): boolean {
    if (!Number.isFinite(nextSec)) return false;
    if (!Number.isFinite(lastSec)) return true;
    return Math.abs(nextSec - lastSec) > PLAYHEAD_EPSILON_SEC;
}
