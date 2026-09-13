/**
 * 参数编辑器内核 · 曲线采样点投影（纯函数）
 *
 * 【主要内容】
 * 把一条参数曲线的**采样值序列**投影为视口坐标点序列，复刻 `render.ts` 的
 * `drawCurveTimed` 循环语义：帧 → 秒 → 视口 x，以及值 → 视口 y（pitch 曲线带
 * `+0.5` 半音偏移）。
 *
 * 【作用】
 * 阶段 3 把曲线搬上 GL 后，几何构建需要"先得到点序列"这一步。把它抽成纯函数
 * 有两个好处：一是可在 node 环境完整单测（GL 与 Canvas2D 两条路径共用同一份投影，
 * 杜绝分叉）；二是让"只返回可见点"这个契约显式化——它直接决定虚线相位是否正确
 * （见下）。
 *
 * 【为什么必须只返回**可见**点（而不是整条曲线）】
 * Canvas2D 的虚线相位从**子路径起点**开始算，而 `drawCurveTimed` 的子路径起点是
 * **首个可见采样点**（它用 `started = false; continue` 跳过视口左缘之前的点，
 * 且从不设置 `lineDashOffset`）。若这里返回整条曲线、由调用方按绝对弧长推进虚线，
 * 滚动时虚线图案会**滑动**，与 Canvas2D 不一致。因此本函数的契约是"返回会被绘制
 * 的点"，弧长从索引 0 开始累加。
 *
 * 【与其他模块的关系】
 * - 上游：面板提供的曲线采样值 + `TimelineAxis` + `valueToY`。
 * - 下游：`renderKernel/gl/polylineGeometry`（点 → 三角形）。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 */

import {
    secToViewportPx,
    viewportEndSec,
    viewportStartSec,
    type TimelineAxis,
} from "../../../renderKernel/timelineAxis";
import { framesToTime } from "../../utils";

/** 投影参数（与 `drawCurveTimed` 的入参一一对应）。 */
export interface CurvePointsArgs {
    /** 采样值（pitch 为 MIDI，其余为参数内部值）。 */
    readonly values: readonly number[];
    /** 参数名（决定是否施加 pitch 的 `+0.5` 偏移）。 */
    readonly param: string;
    /** 首个采样值对应的帧号。 */
    readonly startFrame: number;
    /** 采样步长（帧）。 */
    readonly stride: number;
    /** 每帧时长（毫秒）。 */
    readonly framePeriodMs: number;
    /** 统一投影。 */
    readonly axis: TimelineAxis;
    /** 值 → 视口 y 的投影（宽度无关，由调用方按当前视口高绑定）。 */
    readonly valueToY: (value: number) => number;
    /** 视口高度（CSS px），仅用于把投影结果限制在合理范围（可省略）。 */
    readonly heightPx?: number;
}

/** 一个视口坐标点。 */
export interface CurvePoint {
    readonly x: number;
    readonly y: number;
}

/**
 * 把曲线采样值投影为**可见**点序列。
 *
 * 流程（与 `drawCurveTimed` 逐行等价）：
 * 1. 由 `axis` 取可见时间区间 `[start, end]`；
 * 2. 逐采样点算帧 → 秒；超过右缘即 **break**（采样点时间单调递增，无需继续）；
 *    早于左缘则 **跳过**（对应原实现的 `started = false; continue`）；
 * 3. 视口 x 一律经 `secToViewportPx`；
 * 4. y 经 `valueToY`，其中 pitch 参数先加 0.5（MIDI 值 N 画在 N 键中心）。
 *
 * 特殊说明 1：**返回值只含可见点**，索引 0 即"子路径起点"（见文件头）。
 *
 * 特殊说明 2：采样点数可能上万（实测可见约 1650 点），因此实现里不做任何逐点
 * 对象以外的分配；调用方若需要更高性能可复用输出数组（当前未做，属于过早优化）。
 *
 * 特殊说明 3：非有限值（NaN / Infinity）会被**跳过**而不是产出 NaN 点。设计上
 * 应不会出现，但一旦出现 NaN 会让整条折线的顶点缓冲失效（整层消失），
 * 因此这里按"安全侧"处理。
 *
 * @param args 投影参数。
 * @returns 可见点序列；无可见点或采样不足 2 点时为空数组。
 */
export function projectCurvePoints(args: CurvePointsArgs): CurvePoint[] {
    const { values, param, startFrame, stride, framePeriodMs, axis, valueToY } = args;
    if (values.length < 2) return [];

    const fp = Math.max(1e-6, framePeriodMs);
    const step = Math.max(1, Math.floor(stride));
    const visibleStartSec = viewportStartSec(axis);
    const visibleEndSec = viewportEndSec(axis);
    const isPitch = param === "pitch";

    const out: CurvePoint[] = [];
    for (let i = 0; i < values.length; i += 1) {
        const frame = startFrame + i * step;
        const tSec = framesToTime(frame, fp);
        // 右缘之后：采样时间单调递增，直接结束（与 drawCurveTimed 的 break 一致）
        if (tSec > visibleEndSec) break;
        // 左缘之前：跳过，并且不产生子路径（与 started=false; continue 一致）
        if (tSec < visibleStartSec) continue;

        const x = secToViewportPx(axis, tSec);
        const rawValue = values[i] ?? 0;
        const y = valueToY(isPitch ? rawValue + 0.5 : rawValue);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        out.push({ x, y });
    }
    return out;
}

/**
 * 剪贴板预览曲线的**专属**投影（它不走 `drawCurveTimed`）。
 *
 * 流程（与 `render.ts:1150-1170` 的循环逐行等价）：从**选区起点**开始，按
 * `framePeriodMs` 的**原始帧距**依次排列每个采样值，超过选区终点即停止。
 *
 * 【为什么不能复用 `projectCurvePoints`】两者的时间基准不同：
 * - 普通曲线：`framesToTime(startFrame + i * stride, fp)`，从**曲线自身起点**推算；
 * - 剪贴板预览：`selStartSec + i * cbFp / 1000`，从**选区起点**推算，且忽略
 *   `startFrame` / `stride`（数据是刚复制的片段，直接按原始帧距铺开）。
 * 混用会让预览曲线整体平移 `selStartSec − curveStartSec`，即"粘贴后曲线跳到别处"。
 *
 * 特殊说明：本函数**不做**视口裁剪（原实现也不做）——它依赖选区裁剪，
 * 而选区可能比视口窄。因此返回的点可能落在视口外，由 GL 侧统一裁剪。
 *
 * @param args 投影参数。
 * @returns 预览点序列（视口坐标）；采样不足 2 点或选区非法时为空数组。
 */
export function projectClipboardPreviewPoints(args: {
    readonly values: readonly number[];
    readonly param: string;
    readonly framePeriodMs: number;
    readonly selStartSec: number;
    readonly selEndSec: number;
    readonly axis: TimelineAxis;
    readonly valueToY: (value: number) => number;
}): CurvePoint[] {
    const { values, param, framePeriodMs, selStartSec, selEndSec, axis, valueToY } = args;
    if (values.length < 2) return [];
    if (!(selEndSec > selStartSec)) return [];
    const fp = Math.max(1e-6, framePeriodMs);
    const isPitch = param === "pitch";

    const out: CurvePoint[] = [];
    for (let i = 0; i < values.length; i += 1) {
        // 不缩放、不套用 stride：直接按原始帧间距排列（与原实现一致）
        const tSec = selStartSec + (i * fp) / 1000;
        if (tSec > selEndSec) break;
        const x = secToViewportPx(axis, tSec);
        const rawValue = values[i] ?? 0;
        const y = valueToY(isPitch ? rawValue + 0.5 : rawValue);
        if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
        out.push({ x, y });
    }
    return out;
}

/**
 * 检测曲线（`clipPitchCurves`）的专用投影。
 *
 * 流程（与 `render.ts:972-1010` 的检测曲线循环逐行等价）：
 * 1. 时间基准是 `curveStartSec + (i × fp) / 1000`——**每帧按原始帧距排列**，
 *    没有 `startFrame` / `stride` 概念（曲线自带绝对起始秒）；
 * 2. `midi <= 0`（无声帧）**跳过**，前后有声点直接相连（保持连续性）；
 * 3. 超过右缘 `break`，早于左缘 `continue`；
 * 4. pitch 加 0.5 偏移，与其余曲线同源。
 *
 * 【为什么必须单独一个函数而不是复用 `projectCurvePoints`】
 * 那两处差异都会造成可见缺陷：
 * - **时间基准**：把 `startFrame: 0` 传进去会丢掉 `curveStartSec`，曲线整体平移
 *   到时间轴原点（`curveStartSec` 非 0 的 clip 全部错位）。
 * - **无声帧**：`projectCurvePoints` 保留所有点（对 `paramView` 是对的，那里的 0
 *   是合法值），但检测曲线的 0 表示"无音高"。照画会得到一条从上一个有声点直落
 *   到底部的**垂直尖刺**——GL 迁移后实际出现过（用户截图里粉/紫曲线的密集竖线）。
 *
 * 【为什么不加开关参数】两者的入参语义本就不同（一个收 `startFrame`/`stride`，
 * 一个收 `curveStartSec` 且无 stride）；合并后每个调用点都要传无关参数，而
 * `midi <= 0` 的语义只对检测曲线成立。分开后各自的可测契约更清晰。
 *
 * @param args 投影参数。
 * @returns 可见的有声点序列；无有效点时为空的数组。
 */
export function projectDetectedCurvePoints(args: {
    /** MIDI 音高曲线（每帧一个值，`<= 0` 表示无声）。 */
    readonly midiCurve: readonly number[];
    /** 曲线第 0 帧对应的 timeline 绝对时间（秒）。 */
    readonly curveStartSec: number;
    /** WORLD 帧周期（毫秒）。 */
    readonly framePeriodMs: number;
    /** 统一投影。 */
    readonly axis: TimelineAxis;
    /** 值 → 视口 y 的投影。 */
    readonly valueToY: (value: number) => number;
}): CurvePoint[] {
    const { midiCurve, curveStartSec, framePeriodMs, axis, valueToY } = args;
    if (midiCurve.length < 2) return [];
    if (!Number.isFinite(curveStartSec)) return [];

    const fp = Math.max(1e-6, framePeriodMs);

    // 与旧实现一致：检测曲线按 **x** 判可见性（`x > w + 10` / `x < -10`），
    // 不是按时间区间——因为它只有绝对秒、没有 `startFrame`，用时间判会多一次换算。
    const rightLimitPx = axis.viewportWidthPx + 10;
    const leftLimitPx = -10;

    const out: CurvePoint[] = [];
    for (let i = 0; i < midiCurve.length; i += 1) {
        const midi = midiCurve[i];
        // 非有限值：跳过（与旧实现 `midi == null || !isFinite(midi)` 一致）
        if (midi == null || !Number.isFinite(midi)) continue;

        const frameSec = curveStartSec + (i * fp) / 1000;
        const x = secToViewportPx(axis, frameSec);
        if (x > rightLimitPx) break;
        if (x < leftLimitPx) continue;

        // 无声帧：跳过但保持连续性（旧实现的 `if (midi <= 0) continue`）
        if (midi <= 0) continue;

        const y = valueToY(midi + 0.5);
        if (!Number.isFinite(y)) continue;
        out.push({ x, y });
    }
    return out;
}
