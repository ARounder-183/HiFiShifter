/**
 * Parameter editor playhead seek mapping helpers.
 *
 * 【主要内容】把视口内的客户端 X 坐标换算成工程时间（秒），即统一投影的
 * **逆**映射。
 *
 * 【作用】语义刻意绕开 beat 换算（避免 BPM 变更造成拖动漂移），但逆投影一律
 * 走 `viewportPxToSec`：与时间线侧共用同一个 axis 换算，避免两侧各自实现
 * `(x + scrollLeft) / pxPerSec` 而在边界处差半像素。
 *
 * 【与其他模块的关系】
 * - 依赖：`timeline/runtime/timelineAxis.ts`（唯一投影）。
 * - 消费方：`usePianoRollInteractions.ts` 的 `pointerSec` 与标尺点击定位。
 */

import { viewportPxToSec, type TimelineAxis } from "../timeline/runtime/timelineAxis";

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
