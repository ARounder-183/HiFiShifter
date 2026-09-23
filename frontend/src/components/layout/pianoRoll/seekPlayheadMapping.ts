/**
 * Parameter editor playhead seek mapping helpers.
 *
 * 【主要内容】把视口内的客户端 X 坐标换算成工程时间（秒）与**参数帧号**，即统一
 * 投影的**逆**映射。
 *
 * 【作用】语义刻意绕开 beat 换算（避免 BPM 变更造成拖动漂移），但逆投影一律
 * 走 `viewportPxToSec`：与时间线侧共用同一个 axis 换算，避免两侧各自实现
 * `(x + scrollLeft) / pxPerSec` 而在边界处差半像素。
 *
 * 【与其他模块的关系】
 * - 依赖：`renderKernel/timelineAxis.ts`（唯一投影）。
 * - 消费方：`usePianoRollInteractions.ts` 的 `pointerSec` / `pointerFrame`
 *   与标尺点击定位。
 */

import { viewportPxToSec, type TimelineAxis } from "../renderKernel/timelineAxis";

/**
 * 视口客户端 X → 工程时间（秒）。
 *
 * @param input.clientX 指针的客户端 X 坐标。
 * @param input.viewportLeft 视口左缘的客户端 X 坐标（画布 boundingRect.left）。
 * @param input.axis 当前投影。
 * @returns 工程时间（秒），恒 >= 0。
 */
export function secFromViewportClientX(input: {
    clientX: number;
    viewportLeft: number;
    axis: TimelineAxis;
}): number {
    const { clientX, viewportLeft, axis } = input;
    return Math.max(0, viewportPxToSec(axis, clientX - viewportLeft));
}

/**
 * 视口客户端 X → 参数帧坐标（**连续值**，未取整、**不夹负**）。
 *
 * 公式唯一：`sec × 1000 / framePeriodMs`（与 `utils.ts` 的 `timeToFrame` 同一口径）。
 * 帧是工程级常量栅格（`frame_period_ms` 恒为 5.0），因此这个映射**与 BPM 无关** ——
 * 这正是选区改用帧制的原因：改 BPM 不该让选区跟着动。
 *
 * 【为什么刻意不走 `secFromViewportClientX` 的 `>= 0` 夹取】"将参数编辑器的水平
 * 位置与缩放同步到时间轴"启用时，参数编辑器左侧会腾出与轨道头等宽的空间，那一段
 * 对应**负时间**、也就是负帧（第 -1、-2… 帧的领地）。指针位置必须**照实**换算：
 * 若把整片左侧空间夹到 0，则"第 0 帧的判定范围"会从 1 帧宽膨胀成整片空间宽 ——
 * 用户在很左边按下也会被算成第 0 帧（见 `paramSelection` 的两条边界规则）。
 *
 * 【为什么不在这里取整】调用方有两类需求：选区构造（交给 `paramSelection` 的
 * 指针规则 `frameBoundLeftFromPointer` / `frameBoundRightFromPointer` 量化）与曲线
 * 取值（`frameToIndex` 自己会四舍五入）。在此取整会让"量化"出现第二处定义。
 *
 * @param input.clientX 指针的客户端 X 坐标。
 * @param input.viewportLeft 视口左缘的客户端 X 坐标（画布 boundingRect.left）。
 * @param input.axis 当前投影。
 * @param input.framePeriodMs 帧周期（毫秒）；非有限或 ≤0 时退化为 5ms。
 * @returns 连续帧坐标（工程起点左侧为负）。
 */
export function frameFromViewportClientX(input: {
    clientX: number;
    viewportLeft: number;
    axis: TimelineAxis;
    framePeriodMs: number;
}): number {
    const { clientX, viewportLeft, axis, framePeriodMs } = input;
    const fp = Number.isFinite(framePeriodMs) && framePeriodMs > 0 ? framePeriodMs : 5;
    return (viewportPxToSec(axis, clientX - viewportLeft) * 1000) / fp;
}
