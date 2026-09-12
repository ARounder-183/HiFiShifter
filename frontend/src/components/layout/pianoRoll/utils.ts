/**
 * 参数编辑器（Piano Roll）无依赖工具函数
 *
 * 【主要内容】
 * 1. 帧 ↔ 时间 ↔ 像素的坐标换算；
 * 2. 音乐基础判定（`isBlackKey`）。
 *
 * 【作用】本模块是**无依赖的叶子模块**：既不 import 任何组件，也不触发 Redux /
 * DOM。因此渲染层（Canvas2D 与 GL 两条路径）与 node 环境的单测都能安全引用它。
 *
 * 【为什么这件事重要】`render.ts` 经 `../timeline` 桶文件间接引入了 Redux store
 * 等浏览器专属模块；任何在 node 环境（本工程 Vitest 无 jsdom）导入 `render.ts`
 * 的测试都会因裸 `localStorage` 崩溃。共享的纯逻辑必须放在这里，而不是放在
 * 渲染模块里"顺便导出"。
 *
 * 【与其他模块的关系】被 `render.ts`（Canvas2D 路径）与
 * `kernel/host/pianoRollKernelHost.ts`（GL 路径）共同引用，保证两条路径口径一致。
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

/**
 * 判断是否为黑键（12 音里 1/3/6/8/10 为黑键）。
 *
 * 【为什么放在本模块】本模块是**无依赖的叶子模块**：`render.ts` 与
 * `kernel/host/pianoRollKernelHost.ts` 都要用同一判定来构建键盘几何，复制一份
 * 必然分叉（表现为"某个键在两种渲染模式下颜色不同"，极难归因）。
 *
 * 特殊说明：**不能**把它放在 `render.ts` 里导出——`render.ts` 经
 * `../timeline` 桶文件间接引入了 Redux store 等浏览器专属模块，任何在 node
 * 环境（无 jsdom）导入它的测试都会因裸 `localStorage` 崩溃。放在叶子模块里，
 * GL 侧与单测都能安全引用。
 *
 * @param midi MIDI 音高。
 * @returns 黑键为 true。
 */
export function isBlackKey(midi: number): boolean {
    const pc = ((midi % 12) + 12) % 12;
    return pc === 1 || pc === 3 || pc === 6 || pc === 8 || pc === 10;
}
