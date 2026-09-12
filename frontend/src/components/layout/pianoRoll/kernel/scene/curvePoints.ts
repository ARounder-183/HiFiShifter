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
