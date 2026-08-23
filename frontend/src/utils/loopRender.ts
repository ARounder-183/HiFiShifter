/**
 * Loop（循环源）渲染共享工具
 *
 * Loop 语义（对齐 REAPER / VEGAS 的 item LOOP）：
 *   - 循环区间 = Clip 的源窗口 [sourceStartSec, sourceEndSec]；
 *   - 播放回绕发生在源窗口**内部**（u = t·rate mod span），因此每个循环
 *     周期显示的都是同一份窗口内容 —— 波形分片渲染时所有瓦片共用同一窗口；
 *   - 循环周期（时间线时间）= |sourceEnd − sourceStart| / |playbackRate| 秒，
 *     倒放同样成立（内容方向镜像，回绕位置相同）；
 *   - 回绕节点出现在 clip 局部时间 t = k·周期（k = 1, 2, …），即波形区域中
 *     需要绘制"倒三角"标记的位置；恰好在 clip 起点 / 终点的回绕点不绘制。
 */

export const MIN_LOOP_SPAN_SEC = 1e-6;

/** 判断给定 clip 参数是否会产生有效的循环周期（返回周期秒数，0 = 无效/未启用）。 */
export function loopCycleSec(args: {
    loopEnabled: boolean;
    sourceStartSec: number;
    sourceEndSec: number;
    playbackRate: number;
}): number {
    if (!args.loopEnabled) return 0;
    const span = Math.abs(
        Number(args.sourceEndSec ?? 0) - Number(args.sourceStartSec ?? 0),
    );
    if (!Number.isFinite(span) || span <= MIN_LOOP_SPAN_SEC) return 0;
    const rate = Math.abs(Number(args.playbackRate ?? 1) || 1);
    if (!Number.isFinite(rate) || rate < 1e-6) return 0;
    return span / rate;
}

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
