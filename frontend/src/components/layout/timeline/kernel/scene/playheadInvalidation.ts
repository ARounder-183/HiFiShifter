/**
 * 时间轴渲染内核 · 播放头取值与标脏判定（纯函数）
 *
 * 【主要内容】
 * 1. `resolvePlayheadSec`：在「实时 getter」与「数据镜像」两个来源间取值；
 * 2. `shouldRepaintForPlayhead`：判定当前播放头位置相对上一次绘制是否值得重绘。
 *
 * 【作用：修的是什么】
 * 内核的渲染循环是**纯脏标记驱动**的（`renderKernel/renderLoop`：`start()` 不绘制，
 * 也没有常驻 rAF）。而播放头位置的真值由面板的视觉插值 ref 持有、经数据镜像
 * 传进内核——**镜像变化不会自动标脏**。于是点标尺 seek 后：标尺播放头（React
 * 声明式渲染）动了，内核自绘的轨道区播放头却永不重绘，冻结在旧位置。
 *
 * 单有标脏还不够：数据镜像在**面板 render 期**写入，而视觉插值 ref 在 effect 里
 * 更新，镜像因此**滞后一次提交**。用滞后值定位播放头会停在上一帧的位置，所以
 * 取值也必须走实时 getter——两个函数一个解决"取到真值"、一个解决"变化时重绘"，
 * 缺一不可（这也是它们放在同一模块的原因）。
 *
 * 【与其他模块的关系】
 * - 上游：`timelineKernelHost` 的帧提交（`syncDom` / `draw`）与滚轮缩放锚点。
 * - 独立性：纯函数，无 DOM / WebGL / React 依赖，可在 node 环境单测。
 *
 * 【设计约束】
 * 1. 容差用**绝对值**而不是严格相等：播放头位置来自 `performance.now()` 外推，
 *    同一逻辑位置在两次读取间会有极小浮点抖动；严格相等会每帧都判定"变了"，
 *    把空闲状态也变成持续重绘。
 * 2. 上一次为 `NaN` 视为"尚未绘制过"，必须返回真（首帧不能跳过）。
 * 3. 当前值非有限时返回假：宁可不动，也不能把 NaN 写进 `style.transform`
 *    （NaN 会让该元素整层失效，且不会报错）。
 */

/** 位置比较的绝对容差（秒）。远小于一个像素对应的时间，又足以吸收浮点抖动。 */
const PLAYHEAD_EPSILON_SEC = 1e-6;

/**
 * 解析本帧应使用的播放头位置。
 *
 * 流程：优先取实时 getter → 非有限则回退数据镜像。
 *
 * 特殊说明 1：**实时 getter 优先**是修「镜像滞后一次提交」的关键。面板在 render
 * 期写镜像，而视觉插值 ref 在 effect 里更新，两者之间恰好差一次提交；实测连续
 * 两次 seek 时镜像每次都停在**上一次**的位置。
 *
 * 特殊说明 2：getter 缺省（未接线 / 单测）或返回非有限值时回退镜像——宁可用略旧的
 * 值，也不能让 NaN 流进样式写入（见设计约束 3）。
 *
 * @param liveSec 实时 getter 的返回值；getter 不存在时传 `undefined`。
 * @param mirrorSec 数据镜像里的播放头位置（秒），作为兜底。
 * @returns 本帧用于绘制与标脏判定的播放头位置（秒）。
 */
export function resolvePlayheadSec(liveSec: number | undefined, mirrorSec: number): number {
    if (liveSec !== undefined && Number.isFinite(liveSec)) return liveSec;
    return mirrorSec;
}

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
