/**
 * 视觉播放头（60 Hz 插值值）的**只读广播**。
 *
 * 【为什么需要它】捕获带右缘必须与光标一起走，而两者此前的数据源差 12 倍：光标是
 * 「宿主推送 + rAF 插值」，带子是 **200 ms 轮询**（`captureStore.POLL_MS`）。用户看到的
 * 是带子每 200 ms 往前跳一格 —— 就是"一顿一顿"。
 *
 * 【为什么不是 props】`CaptureRangeLayer` 挂在时间轴内核视图里，而视觉播放头的 ref 由
 * `TimelinePanel` 持有、经 `TimelineTransportBridge` 的 `onFrame` 逐帧写入。把那条 ref
 * 一路透传进内核视图要改四层签名，而这里只需要**一个数**。
 *
 * 【为什么不是订阅 store】store 里的 `playheadSec` 是**采样值**（约 20~50 Hz 且带往返
 * 时延），不是插值值。用它画带子右缘仍然会一顿一顿，只是从 5 Hz 变成 20 Hz。
 *
 * 与 `clipGeometryPreviewBus` 同一形态：模块级单值 + 显式发布，不进 React 状态，
 * 因为它的消费者是 rAF 回调而不是渲染。
 */

let currentSec = 0;

/** 由 `TimelineTransportBridge` 的 `onFrame` 每帧调用（与绘制同源）。 */
export function publishVisualPlayheadSec(sec: number): void {
    if (Number.isFinite(sec)) currentSec = sec;
}

/** 读最近一帧的视觉播放头（秒）。 */
export function readVisualPlayheadSec(): number {
    return currentSec;
}
