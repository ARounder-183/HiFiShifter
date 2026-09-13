/**
 * 参数编辑器（Piano Roll）主题配色表
 *
 * 【主要内容】
 * 集中定义参数编辑器画布的全部颜色：琴键区、网格线、钢琴背景（黑键行背景带）、
 * 音阶高亮、曲线、叠加文字与播放头，按深色 / 浅色两套主题给出。
 *
 * 【作用】
 * 这些颜色此前**内联在** `render.ts` 的 `drawPianoRoll` 里。阶段 2 起同一份配色
 * 要被两条渲染路径消费（Canvas2D 细节层、GL 静态层），内联会立刻产生"两份配色
 * 各自演进"的风险——而颜色分叉的表现是"某个图层在两种模式下色值差一点点"，
 * 极难归因。因此提取为单一来源。
 *
 * 【与其他模块的关系】
 * - 上游：`render.ts` 的 `drawPianoRoll`（Canvas2D 路径，音阶高亮已迁走、只余非网格
 *   图层）与 `PianoRollPanel` 在构建 GL 场景层的网格输入时调用。
 * - 下游：`PianoRollPanel.buildGridSpec` 把本表的 CSS 颜色经 `parseRgbaColor` 转成
 *   数值 RGBA，交给 `kernel/scene/gridInstances` 构建 GL 实例（含黑键行背景带与
 *   音阶高亮强调线）。
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
    /**
     * 黑键行背景带（钢琴背景：黑键行与白键行区分开，其余行保持原背景）。
     *
     * 【为什么只标黑键行】这是 REAPER / Logic / Ableton 的既有惯例：只给黑键行加一条
     * 半透明带、白键行保持原背景，正好复刻钢琴键盘的黑白交替。反过来给白键行加带
     * 在浅色主题下会与底色糊在一起，反而削弱行间对比。
     *
     * 【为什么两套主题的**明暗方向相反**（只读常量值时尤其注意）】带子要"看得出行、
     * 又压不住网格线"：
     * - 浅色主题底色亮（`#edf0f5`），**压暗**有余量：`rgba(0,0,0,0.06)` → 实测 Δ14。
     * - 深色主题底色本来就接近黑（`#1f1f1f` = 31），压暗**没有余量**：
     *   `rgba(0,0,0,0.08)` 只能到 29，Δ2 —— 实测**肉眼几乎不可见**（对照：深色主题
     *   的弱网格线本身 Δ11，比带子还明显，等于背景带白做）。
     *   因此深色主题改用**提亮**：`rgba(255,255,255,0.04)` → Δ9，与网格线同量级、
     *   既不淹没线也能看清行。
     *
     * 特殊说明 1：alpha 必须克制——"网格线仍清晰可见"是验收标准。数值集中放在这里，
     * 后续调参只改一处。
     * 特殊说明 2：必须是 `rgba()` 写法。`parseRgbaColor` 只认 `rgb()/rgba()`，hex 会
     * 被解析成**不透明洋红**（故意的暴露设计，见 `normalizeCssColor` 说明）。
     */
    readonly blackKeyRowBand: string;
    /**
     * 音阶高亮：音阶音级上的强调线。
     *
     * 特殊说明：绘制顺序在 {@link blackKeyRowBand} **之上**——背景是纹理、高亮是
     * 语义，语义必须压住纹理。同样必须是 `rgba()` 写法。
     */
    readonly scaleHighlight: string;
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
    // 钢琴背景 / 音阶高亮（见接口处的取值说明：alpha 必须压不住网格线）
    // 深色主题**提亮**（底色接近黑，压暗没有余量，Δ2 等于不可见）；浅色主题压暗。
    blackKeyRowBand: "rgba(255,255,255,0.04)",
    scaleHighlight: "rgba(255,200,80,0.22)",
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
    // 钢琴背景 / 音阶高亮（浅色主题的琥珀加深，否则在白底上不可读）
    blackKeyRowBand: "rgba(0,0,0,0.06)",
    scaleHighlight: "rgba(200,120,20,0.22)",
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

/**
 * 把任意 CSS 颜色写法归一化为浏览器认可的 `rgb()/rgba()` 形式。
 *
 * 【为什么必须先归一化】`parseRgbaColor` 只认 `rgb()/rgba()` 两种写法，其他格式
 * （hex / 颜色关键字 / 主题变量）会被解析成**不透明洋红**——这是既有实现的故意
 * 设计，用来在真机上把"漏解析"暴露成刺眼的洋红。而本配色表里 `whiteKey`
 * （`#ffffff` / `#d7dade`）与 `blackKey`（`#3a3a3a` / `#2e3136`）正是 hex 写法，
 * 直接解析会让整个键盘变成洋红（曾实际发生）。
 *
 * 特殊说明 1：借浏览器做归一化（挂一个游离 `span` 读 `getComputedStyle`）。这条
 * 路径会触发样式重算，**不能进渲染热路径**——调用方必须在低频时机（镜像更新 /
 * 主题变化）解析并按值缓存。
 *
 * 特殊说明 2：无 DOM 环境（node 单测）原样返回，由调用方的有限性校验兜底。
 *
 * @param css CSS 颜色字符串（任意写法）。
 * @returns `rgb()/rgba()` 字符串；无 DOM 时原样返回。
 */
export function normalizeCssColor(css: string): string {
    if (typeof document === "undefined") return css;
    const probe = document.createElement("span");
    probe.style.color = css;
    probe.style.display = "none";
    document.body.appendChild(probe);
    const computed = getComputedStyle(probe).color;
    probe.remove();
    return computed.length > 0 ? computed : css;
}

/**
 * 检测曲线（后端推送的 per-clip 音高曲线）的循环调色板。
 *
 * 【为什么两套主题不同】浅色主题必须提高不透明度并加深，否则青绿在白底上
 * 对比度只有约 1.4:1、几乎隐形；深色主题则要避免过亮刺眼。
 *
 * 【为什么琥珀色被换成玫红】琥珀是**编辑包络线**的专属色相，检测曲线占用它
 * 会与主曲线混淆——这是既有实现刻意做的取舍，迁移时保留。
 *
 * @param isDark 是否深色主题。
 * @returns 调色板（按 clip 索引循环取用）。
 */
export function resolveDetectedCurveColors(isDark: boolean): readonly string[] {
    return isDark
        ? ["rgba(80, 220, 180, 0.56)", "rgba(255, 110, 197, 0.60)", "rgba(180, 120, 255, 0.56)", "rgba(60, 180, 255, 0.56)"]
        : ["rgba(0, 150, 118, 0.80)", "rgba(214, 44, 140, 0.75)", "rgba(124, 58, 237, 0.70)", "rgba(2, 132, 199, 0.80)"];
}

/**
 * 副参数曲线的循环调色板。
 *
 * 特殊说明：`"pitch"` 这个副参数不走本表，它固定用青蓝（见
 * `resolveSecondaryCurveColor`），以便与主参数曲线区分。
 *
 * @param isDark 是否深色主题。
 * @returns 调色板（按副参数索引循环取用）。
 */
export function resolveSecondaryCurveColors(isDark: boolean): readonly string[] {
    return isDark
        ? ["rgba(100, 200, 255, 0.62)", "rgba(255, 110, 197, 0.62)", "rgba(180, 120, 255, 0.62)", "rgba(60, 200, 160, 0.62)"]
        : ["rgba(0, 116, 200, 0.72)", "rgba(214, 44, 140, 0.72)", "rgba(124, 58, 237, 0.72)", "rgba(22, 163, 116, 0.75)"];
}

/**
 * 取某个副参数曲线的颜色。
 *
 * @param isDark 是否深色主题。
 * @param paramId 副参数 id。
 * @param index 在副参数列表中的序号（用于循环取色）。
 * @returns CSS 颜色。
 */
export function resolveSecondaryCurveColor(
    isDark: boolean,
    paramId: string,
    index: number,
): string {
    if (paramId === "pitch") return "rgba(100, 200, 255, 0.65)";
    const palette = resolveSecondaryCurveColors(isDark);
    return palette[index % palette.length];
}

/** 剪贴板预览曲线的颜色（青蓝，用虚线+降不透明度与选区高亮区分）。 */
export function resolveClipboardPreviewColor(isDark: boolean): string {
    return isDark ? "rgba(100, 200, 255, 0.55)" : "rgba(0, 116, 200, 0.60)";
}
