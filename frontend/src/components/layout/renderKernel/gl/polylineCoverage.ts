/**
 * 折线覆盖率计算（纯函数，着色器数学的参考实现）
 *
 * 【主要内容】
 * 把 `polylineProgram` 片元着色器里的两段覆盖率数学提取为 TypeScript 纯函数：
 * 1. **横向覆盖率**：由「到中心线的有符号距离」得到亚像素抗锯齿；
 * 2. **虚线覆盖率**：由「沿线的累积弧长」得到虚线图案（含端点抗锯齿）。
 *
 * 【作用】
 * 着色器里的数学无法在 node 环境单测（无 WebGL），但它恰恰是"曲线看起来对不对"
 * 的核心：抗锯齿过渡宽度、虚线疏密、边界处的半覆盖值。把同一套数学写成 TS 参考
 * 实现后可以完整单测，GLSL 里保留逐行等价的副本并在此注明——这是本工程对
 * "无法直接测试的着色器数学"的既有处理方式（对照 `sdfBoxProgram` 的注释约定）。
 *
 * 【为什么不做成"着色器读 TS"】GLSL 无法 import TS；把公式生成成字符串再拼接
 * 会让着色器失去可读性与编译期检查。因此选择"两份等价实现 + 参考实现有测试 +
 * 顶部注明同步义务"。
 *
 * 【同步义务】修改本文件的公式时，**必须**同步修改 `polylineProgram.ts` 的
 * `FRAGMENT_SHADER` 中对应段落，否则两者会静默分叉。
 *
 * 【与其他模块的关系】
 * - 上游：无（纯数学）。
 * - 下游：`polylineProgram`（GLSL 副本）；单测直接覆盖本文件。
 * - 独立性：纯函数，不依赖 DOM / WebGL / React。
 */

/** 覆盖率结果，取值为 `[0, 1]`。 */
export type Coverage = number;

/** 抗锯齿过渡宽度的下限（CSS px），也是非法输入的回退值。 */
const MIN_AA_WIDTH = 1e-6;

/**
 * 归一化抗锯齿过渡宽度。
 *
 * 【为什么不能只用 `Math.max`】`Math.max(NaN, 1e-6)` 返回 **NaN**（NaN 与任何数
 * 取 max 都是 NaN），于是后续比较全部为 false、函数返回 NaN。调用方一旦把 NaN
 * 写进 uniform，整个 draw call 的几何都会消失。因此必须**显式判有限性**——
 * 这与 `glyphRasterizer.resolveGlyphRasterizerParams` 处理 NaN 页边长的教训同源。
 *
 * @param aaWidth 原始过渡宽度（CSS px）。
 * @returns 合法的过渡宽度。
 */
function normalizeAaWidth(aaWidth: number): number {
    return Number.isFinite(aaWidth) && aaWidth > 0 ? aaWidth : MIN_AA_WIDTH;
}

/**
 * 横向覆盖率：一条到中心线距离为 `across` 的片元被线体覆盖的比例。
 *
 * 流程：`|across|` 在 `[halfWidth - aa/2, halfWidth + aa/2]` 区间内线性衰减，
 * 内侧饱和到 1、外侧饱和到 0。
 *
 * 特殊说明：过渡宽度取 **1 个设备像素**（CSS px 下即 `1/dpr`）——这与 Canvas2D
 * 的边缘过渡尺度一致，也是"非整数线宽（1.8 / 2.6 / 3.2 / 3.6）看起来一样粗"的
 * 前提：几何宽度给的是 1.8，但视觉宽度还取决于过渡的落点。
 *
 * @param across 到中心线的有符号距离（CSS px）。
 * @param halfWidth 线宽的一半（CSS px）。
 * @param aaWidth 抗锯齿过渡宽度（CSS px），必须为正。
 * @returns 覆盖率（0..1）。
 */
export function lateralCoverage(across: number, halfWidth: number, aaWidth: number): Coverage {
    const aa = normalizeAaWidth(aaWidth);
    const edge = Math.abs(across);
    const inner = halfWidth - aa * 0.5;
    const outer = halfWidth + aa * 0.5;
    if (edge <= inner) return 1;
    if (edge >= outer) return 0;
    return 1 - (edge - inner) / (outer - inner);
}

/**
 * 虚线覆盖率：累积弧长 `along` 处的片元被虚线"墨"覆盖的比例。
 *
 * 流程：`phase = along mod (dash + gap)` → 求该相位到最近"墨/隙边界"的有符号距离
 * → 按 `aaWidth` 做过渡。
 *
 * 特殊说明 1：**相位从传入序列的第一个点起算**（`along` 由调用方从 0 开始累加）。
 * Canvas2D 的虚线相位从子路径起点开始，而曲线的子路径起点是**首个可见采样点**；
 * 若调用方传整条曲线（绝对弧长），滚动时虚线会滑动。
 *
 * 特殊说明 2：`dash < 0` 表示实线，直接返回 1（与着色器里 `u_dash.x < 0` 一致）。
 *
 * 特殊说明 3：墨隙边界两侧都做过渡，因此虚线端点也有抗锯齿——与 Canvas2D 的
 * 虚线渲染一致（它在端点处同样有覆盖率过渡）。
 *
 * @param along 沿折线的累积弧长（CSS px）。
 * @param dash 实线段长度（CSS px）；负值表示实线。
 * @param gap 空隙段长度（CSS px）。
 * @param aaWidth 抗锯齿过渡宽度（CSS px）。
 * @returns 覆盖率（0..1）。
 */
export function dashCoverage(along: number, dash: number, gap: number, aaWidth: number): Coverage {
    if (!(dash >= 0)) return 1;
    const aa = normalizeAaWidth(aaWidth);
    // `dash + gap` 为 NaN 时同样会穿透 max，故显式判有限性。
    const rawPeriod = dash + gap;
    const period =
        Number.isFinite(rawPeriod) && rawPeriod > MIN_AA_WIDTH ? rawPeriod : MIN_AA_WIDTH;
    // 负数取模在 JS/GLSL 中都可能返回负值，统一归一到 [0, period)。
    const phase = ((along % period) + period) % period;

    // 有符号距离：正数在"墨"内、负数在"隙"内，0 为边界。
    let distance: number;
    if (phase < dash) {
        distance = Math.min(phase, dash - phase);
    } else {
        distance = -Math.min(phase - dash, period - phase);
    }
    return Math.max(0, Math.min(1, distance / aa + 0.5));
}
