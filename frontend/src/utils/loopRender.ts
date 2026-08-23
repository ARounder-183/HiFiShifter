/**
 * Loop（循环源）渲染共享工具
 *
 * Loop 语义（对齐 REAPER / VEGAS 的 item LOOP）：
 *   - 音频 Clip（含由音频转换来的音高参考块）：回绕发生在**整个原始媒体
 *     文件**上（floor_mod 映射，D = 媒体总时长），与后端引擎 / 离线渲染
 *     一致 —— 波形按"头部进入段 + 整文件重复段"分片渲染，回绕节点位于
 *     头部段结束处及此后每个整文件周期边界；
 *   - 纯音高参考块（Pitch Reference，无源媒体）：与普通媒体 Clip 完全
 *     一致 —— 回绕周期 D = **音符内容的最大结束时间**（整个内容），
 *     窗口之外为静音；
 *   - 循环周期（时间线时间）= 周期源秒 / |playbackRate|；
 *   - 回绕节点在 clip 局部时间 t = k·周期（k = 1, 2, …）处绘制
 *     "倒三角"标记；恰好在 clip 起点 / 终点的回绕点不绘制。
 *
 * 周期解析逻辑见 MidiPitchTrackCanvas 的 resolveLoopCycleDescriptor
 * （有源媒体 → D；无源媒体 → 窗口跨度），与音频波形画布的推导保持一致。
 */

/**
 * floor_mod（欧几里得模）：结果始终落在 [0, n)。
 *
 * 引擎的回绕数学是 floor_mod —— 锚点可为负 / 超界，渲染端必须用同一
 * 归一化方式（而不是 clamp），否则存储在域外的锚点会让波形相位与
 * 实际播放错位。
 */
export function modEuclid(a: number, n: number): number {
    if (!(n > 0)) return a;
    return ((a % n) + n) % n;
}

/**
 * 非 Loop 正放 Clip 的**派生源终点**（REAPER 派生窗口模型）：
 * se' = source_start + length×rate。
 *
 * Clip 的消费区间为 [source_start, se')，落在媒体 [0, D) 之外的部分是静音
 * （右缘延伸的尾部静音 / REAPER 左延伸的前导静音）。存储的 sourceEndSec 在
 * 循环开关切换、历史工程等场景下可能与长度脱钩 —— 渲染端必须使用派生值，
 * 否则陈旧窗口会把"需要有音频的地方"冻结成空白。Loop（回绕锚点）与倒放
 * （反向锚点）保持原字段。
 */
export function resolveSourceEndSec(clip: {
    loopEnabled: boolean;
    reversed: boolean;
    sourceStartSec: number;
    playbackRate: number;
    lengthSec: number;
    sourceEndSec: number;
}): number {
    if (!clip.loopEnabled && !clip.reversed) {
        const rate =
            Number.isFinite(clip.playbackRate) && clip.playbackRate > 1e-6
                ? clip.playbackRate
                : 1;
        return (
            (Number(clip.sourceStartSec) || 0) +
            Math.max(0, Number(clip.lengthSec) || 0) * rate
        );
    }
    return Number(clip.sourceEndSec) || 0;
}

/**
 * 解析 Loop 回绕使用的**媒体总时长**（秒）。
 *
 * 统一取值顺序：`durationFrames / sourceSampleRate`（精确）优先，
 * `durationSec`（可能被量化/舍入）兜底 —— 与后端
 * clip_source_media_duration_sec 及 piano-roll / MIDI 画布保持一致，
 * 避免不同消费端对同一文件推导出不同的回绕周期。
 */
export function resolveLoopMediaDurationSec(args: {
    durationFrames?: number | null;
    sourceSampleRate?: number | null;
    durationSec?: number | null;
}): number {
    if (args.durationFrames && args.sourceSampleRate && args.sourceSampleRate > 0) {
        const d = args.durationFrames / args.sourceSampleRate;
        if (Number.isFinite(d) && d > 0) return d;
    }
    const fallback = Number(args.durationSec ?? 0);
    return Number.isFinite(fallback) && fallback > 0 ? fallback : 0;
}

/**
 * Clip 的**内容时长**（秒）：循环/边界逻辑的周期 D。
 *
 * - 有源媒体的 Clip（含由音频转换来的音高参考块）：源媒体总时长；
 * - 纯音高参考块（Pitch Reference，无源媒体）：**音符内容的最大结束时间**
 *   —— 与普通媒体 Clip 完全一致：Loop 回绕整个内容，窗口之外为静音；
 * - 两者都无法确定时返回 null（调用方自行退化，如窗口跨度）。
 */
export function resolveClipContentDurationSec(clip: {
    sourcePath?: string | null;
    midiNoteData?: ReadonlyArray<{ endSec: number }> | null;
    durationFrames?: number | null;
    sourceSampleRate?: number | null;
    durationSec?: number | null;
}): number | null {
    if (clip.sourcePath) {
        const d = resolveLoopMediaDurationSec({
            durationFrames: clip.durationFrames,
            sourceSampleRate: clip.sourceSampleRate,
            durationSec: clip.durationSec,
        });
        return d > 1e-9 ? d : null;
    }
    const notes = clip.midiNoteData;
    if (notes && notes.length > 0) {
        let maxEnd = 0;
        for (const n of notes) {
            const end = Number(n.endSec);
            if (Number.isFinite(end) && end > maxEnd) maxEnd = end;
        }
        return maxEnd > 1e-9 ? maxEnd : null;
    }
    return null;
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
