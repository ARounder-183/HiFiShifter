/**
 * 参数编辑器（Piano Roll）主题配色表
 *
 * 【主要内容】
 * 集中定义参数编辑器画布的全部颜色：琴键区、网格线、曲线、叠加文字与播放头，
 * 按深色 / 浅色两套主题给出。
 *
 * 【作用】
 * 这些颜色此前**内联在** `render.ts` 的 `drawPianoRoll` 里。阶段 2 起同一份配色
 * 要被两条渲染路径消费（Canvas2D 细节层、GL 静态层），内联会立刻产生"两份配色
 * 各自演进"的风险——而颜色分叉的表现是"某个图层在两种模式下色值差一点点"，
 * 极难归因。因此提取为单一来源。
 *
 * 【与其他模块的关系】
 * - 上游：`render.ts` 的 `drawPianoRoll`（Canvas2D 路径）与 `PianoRollPanel`
 *   在构建 GL 场景层的网格输入时调用。
 * - 独立性：纯函数 + 常量，不依赖 DOM / React / WebGL。
 */

/** 参数编辑器画布配色。 */
export interface PianoRollColors {
    /** 键盘轴右缘分隔线。 */
    readonly axisBorder: string;
    /** 白键底色。 */
    readonly whiteKey: string;
    /** 黑键底色。 */
    readonly blackKey: string;
    /** 黑键右缘渐变的深色端（另一端透明）。 */
    readonly blackKeyGradient: string;
    /** C 音名标签（加粗）。 */
    readonly cLabel: string;
    /** 白键音名标签。 */
    readonly whiteKeyLabel: string;
    /** 黑键音名标签。 */
    readonly blackKeyLabel: string;
    /** C 键分隔线（强）。 */
    readonly cSeparator: string;
    /** 其余键分隔线（弱）。 */
    readonly keySeparator: string;
    /** 非音高参数的刻度标签文字。 */
    readonly tensionLabel: string;
    /** 非音高参数的刻度线。 */
    readonly tensionLine: string;
    /** 音高网格：C 线。 */
    readonly pitchGridC: string;
    /** 音高网格：其余半音线。 */
    readonly pitchGridOther: string;
    /** 原始曲线（虚线）。 */
    readonly origCurve: string;
    /** 编辑后曲线（实线）。 */
    readonly editCurve: string;
    /** 选区内高亮曲线。 */
    readonly selectionCurve: string;
    /** 画布中央的操作提示文字。 */
    readonly overlayTextColor: string;
    /** 播放头竖线。 */
    readonly playheadLine: string;
}

/** 深色主题配色。 */
const DARK_COLORS: PianoRollColors = {
    // 琴键区（白键降亮一档、黑键提亮一档：在深色画布上既不刺眼也不淹没）
    axisBorder: "rgba(255,255,255,0.08)",
    whiteKey: "#d7dade",
    blackKey: "#2e3136",
    blackKeyGradient: "rgba(0,0,0,0.35)",
    cLabel: "#3b82f6",
    whiteKeyLabel: "rgba(60,63,70,0.75)",
    blackKeyLabel: "rgba(220,220,220,0.80)",
    cSeparator: "rgba(100,100,100,0.45)",
    keySeparator: "rgba(160,160,160,0.20)",
    tensionLabel: "rgba(255,255,255,0.55)",
    tensionLine: "rgba(255,255,255,0.10)",
    // 网格线
    pitchGridC: "rgba(255,255,255,0.10)",
    pitchGridOther: "rgba(255,255,255,0.05)",
    // 曲线
    origCurve: "rgba(200,200,200,0.55)",
    editCurve: "rgba(255,255,255,0.92)",
    selectionCurve: "rgba(100,200,255,0.95)",
    // 叠加文字 & 播放头（画布中央的操作提示文字，需保持可读：
    // 旧值 35% 不透明度在两套主题下都只剩 1.5-1.8:1）
    overlayTextColor: "rgba(235,240,248,0.45)",
    playheadLine: "rgba(255,255,255,0.25)",
};

/** 浅色主题配色。 */
const LIGHT_COLORS: PianoRollColors = {
    axisBorder: "rgba(0,0,0,0.10)",
    whiteKey: "#ffffff",
    blackKey: "#3a3a3a",
    blackKeyGradient: "rgba(0,0,0,0.25)",
    cLabel: "#2563eb",
    whiteKeyLabel: "rgba(80,80,80,0.65)",
    blackKeyLabel: "rgba(255,255,255,0.85)",
    cSeparator: "rgba(0,0,0,0.25)",
    keySeparator: "rgba(0,0,0,0.12)",
    tensionLabel: "rgba(0,0,0,0.55)",
    tensionLine: "rgba(0,0,0,0.10)",
    // 网格线
    pitchGridC: "rgba(0,0,0,0.12)",
    pitchGridOther: "rgba(0,0,0,0.06)",
    // 曲线
    origCurve: "rgba(132,104,26,0.80)",
    editCurve: "rgba(178,108,0,1)",
    selectionCurve: "rgba(0,116,200,1)",
    // 叠加文字 & 播放头（画布中央的操作提示文字，需保持可读）
    overlayTextColor: "rgba(30,36,48,0.60)",
    playheadLine: "rgba(0,0,0,0.20)",
};

/**
 * 取当前主题的配色表。
 *
 * @param isDark 是否深色主题。
 * @returns 配色表；两套主题返回**同一对象引用**（只读，勿改）。
 */
export function resolvePianoRollColors(isDark: boolean): PianoRollColors {
    return isDark ? DARK_COLORS : LIGHT_COLORS;
}
