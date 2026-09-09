/**
 * Piano Roll coordinate transformation utilities
 *
 * Ensures consistent frame ↔ time conversion across all components.
 */

/**
 * Convert frame number to time in seconds.
 *
 * @param frame - Frame index (0-based)
 * @param framePeriodMs - Frame period in milliseconds (typically 5.0ms)
 * @returns Time in seconds
 *
 * Formula: time_sec = (frame * framePeriodMs) / 1000
 */
export function framesToTime(frame: number, framePeriodMs: number): number {
    const fp = Math.max(1e-6, framePeriodMs); // Prevent division by zero
    return (frame * fp) / 1000;
}

/**
 * Convert time in seconds to frame number.
 *
 * @param timeSec - Time in seconds
 * @param framePeriodMs - Frame period in milliseconds (typically 5.0ms)
 * @returns Frame index (floored to integer)
 *
 * Formula: frame = floor(timeSec * 1000 / framePeriodMs)
 */
export function timeToFrame(timeSec: number, framePeriodMs: number): number {
    const fp = Math.max(1e-6, framePeriodMs); // Prevent division by zero
    return Math.floor((timeSec * 1000) / fp);
}

/**
 * Convert time in seconds to canvas pixel position.
 *
 * 注意：这是**旧坐标公式**，生产路径已统一走 timelineAxis 的
 * `secToViewportPx`（二者等价性由 renderProjection.test.ts 的随机比对守护）。
 * 保留它仅为给该回归测试提供独立的参照实现，勿在新代码中使用。
 *
 * @param timeSec - Time in seconds
 * @param visibleStartSec - Start of visible time range
 * @param visibleDurSec - Duration of visible time range
 * @param canvasWidth - Width of canvas in pixels
 * @returns Pixel position (0 = left edge, canvasWidth = right edge)
 */
export function timeToPixel(
    timeSec: number,
    visibleStartSec: number,
    visibleDurSec: number,
    canvasWidth: number,
): number {
    const denom = Math.max(1e-9, visibleDurSec);
    return ((timeSec - visibleStartSec) / denom) * canvasWidth;
}
