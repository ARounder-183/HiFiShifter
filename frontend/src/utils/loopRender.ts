/**
 * Loop（循环源）渲染共享工具
 *
 * Loop 语义（对齐 REAPER / VEGAS 的 item LOOP）：
 *   - 音频 Clip（含由音频转换来的音高参考块）：回绕发生在**整个原始媒体
 *     文件**上（floor_mod 映射，D = 媒体总时长），与后端引擎 / 离线渲染
 *     一致 —— 波形按"头部进入段 + 整文件重复段"分片渲染，回绕节点位于
 *     头部段结束处及此后每个整文件周期边界；
 *   - 纯 MIDI 导入 Clip：没有源媒体可循环，回绕周期退化为音符内容窗口
 *     跨度 [sourceStartSec, sourceEndSec]；
 *   - 循环周期（时间线时间）= 周期源秒 / |playbackRate|；
 *   - 回绕节点在 clip 局部时间 t = k·周期（k = 1, 2, …）处绘制
 *     "倒三角"标记；恰好在 clip 起点 / 终点的回绕点不绘制。
 *
 * 周期解析逻辑见 MidiPitchTrackCanvas 的 resolveLoopCycleDescriptor
 * （有源媒体 → D；无源媒体 → 窗口跨度），与音频波形画布的推导保持一致。
 */

/**
 * 在波形渲染区域内绘制回绕节点的"倒三角"标记。
 *
 * 坐标系：与 WaveformTrackCanvas / MidiPitchTrackCanvas 一致 ——
 * canvas 左边缘对应视口起点，x 单位为 CSS 像素，y 从波形区顶部开始。
 *
 * @param ctx          Canvas 2D 上下文
 * @param markers      每个标记的水平位置（canvas 本地 CSS 像素）
 * @param displayH     波形区高度（CSS 像素）
 * @param color        标记颜色
 */
export function drawLoopMarkers(
    ctx: CanvasRenderingContext2D,
    markers: number[],
    displayH: number,
    color: string,
): void {
    if (markers.length === 0) return;
    const size = Math.min(7, Math.max(4.5, displayH * 0.16));
    const halfWidth = size * 0.62;
    ctx.save();
    ctx.fillStyle = color;
    for (const x of markers) {
        // 三角形顶边贴着波形区顶部，尖端向下指向波形中心方向。
        ctx.beginPath();
        ctx.moveTo(x - halfWidth, 0.5);
        ctx.lineTo(x + halfWidth, 0.5);
        ctx.lineTo(x, size);
        ctx.closePath();
        ctx.fill();
    }
    ctx.restore();
}
