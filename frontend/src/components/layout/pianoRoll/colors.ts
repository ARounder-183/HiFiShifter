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
 * 【取值未必是字面色值：可能是 CSS 变量】{@link PLAYHEAD_COLOR_TOKEN} 是
 * `var(--qt-playhead)`，两套主题共用——播放头必须与时间轴标尺（同一个变量）同色，
 * 写死字面色值会让用户一改主题色两处就分叉（曾实际发生）。因此**消费方必须先
 * 归一化**：GL 走 `parseRgbaColor(normalizeCssColor(css))`，Canvas2D 的
 * `strokeStyle` 也必须过一遍 `normalizeCssColor`（它同样不解析 `var(...)`，
 * 且非法值是**静默忽略**而非报错）。
 *
 * 【与其他模块的关系】
 * - 上游：`render.ts` 的 `drawPianoRoll`（Canvas2D 路径，音阶高亮已迁走、只余非网格
 *   图层）与 `PianoRollPanel` 在构建 GL 场景层的网格输入时调用。
 * - 下游：`PianoRollPanel.buildGridSpec` 把本表的 CSS 颜色经 `parseRgbaColor` 转成
 *   数值 RGBA，交给 `kernel/scene/gridInstances` 构建 GL 实例（含黑键行背景带与
 *   音阶高亮强调线）；`PianoRollPanel.resolvePlayheadRgba` 另取 `playheadLine` 喂给
 *   GL 叠加层的播放头（经 `normalizeCssColor` 解析变量，带缓存）。
 * - 独立性：纯函数 + 常量，不依赖 DOM / React / WebGL（变量只是字符串，解析由消费方负责）。
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
     * 【方向恒定：两套主题都必须**压暗**】键盘列的黑键恒比白键暗（`blackKey`
     * `#2e3136` vs `whiteKey` `#d7dade`），背景带复刻的是同一个语义，因此方向不能
     * 随主题翻转。曾有一版深色主题改用**提亮**（理由是"底色接近黑、压暗没有余量"），
     * 结果黑键行比白键行更亮 —— 与键盘列**恰好相反**（用户报告："深色模式下背景的
     * 黑键和白键的深色区域是相反的（浅色模式正常）"）。压暗在深色底上余量确实小，
     * 但那是**选 alpha** 的问题，不是改方向能解决的。
     *
     * 【alpha 的取值区间：看得见，但不抢过网格线】
     * - 下限（不得白做）：Δ带 ≥ 6。深色底 `#1f1f1f` = 31，压暗余量只有 31，
     *   所以深色必须用明显更大的 alpha（0.25 → Δ7.75）才能达到与浅色同量级的可辨度；
     *   早期深色版的 `rgba(0,0,0,0.08)` 只有 Δ2，肉眼几乎不可见。
     * - 上限（网格线仍是最强对比）：Δ带 ≤ Δ最弱网格线。浅色两者同为 Δ14.3；深色
     *   弱网格线 `rgba(255,255,255,0.05)` 是 Δ11.2，带子 Δ7.75 仍在其下。
     *
     * 特殊说明 1：alpha 必须克制——"网格线仍清晰可见"是验收标准。数值集中放在这里，
     * 后续调参只改一处；方向性与上下限由 `colors.test.ts` 按主题逐个钉住。
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
    /**
     * 选区块的半透明填充。
     *
     * 【为什么必须留在配色表里】它此前散落在 `render.ts` 的绘制代码中（两条字面
     * `rgba(100,200,255,…)`），而选区块现在由 **GL 场景层**绘制——宿主需要一个
     * 数值 RGBA，若在那侧再写一份字面色值就会分叉。集中在这里，Canvas2D 与 GL
     * 两条路径共用同一个来源。
     *
     * 【alpha 必须很低（0.08）】选区块覆盖整个视口高度，曲线从中穿过。它比曲线
     * **更早**合成（在曲线之下），alpha 稍大就会把下方的网格冲淡。
     */
    readonly selectionBand: string;
    /** 选区块的边框（四条 1 CSS px 的边）。 */
    readonly selectionBorder: string;
    /** 画布中央的操作提示文字。 */
    readonly overlayTextColor: string;
    /**
     * 播放头竖线。
     *
     * 【必须是 CSS 变量，不能是字面色值】参数编辑器复用时间轴那套标尺组件
     * （`TimeRuler`），它的播放头取 `--qt-playhead`（DOM/CSS 变量，用户可在
     * 「外观设置」里改）。画布侧的播放头若自带一套字面色值，用户一改主题色两者就
     * **颜色不一致**（用户报告："播放线在底下和上方标尺的颜色不一致"）。
     *
     * 曾有一版两套主题各写各的（深色 `rgba(255,255,255,0.25)`、浅色
     * `rgba(0,0,0,0.20)`），两个都不是 `--qt-playhead` —— 与标尺的 `#f05a5a`
     * 毫无关系。统一为变量后，两处必然同色，且随用户的主题设置一起变。
     *
     * 特殊说明：GL 侧不认 `var(...)`，调用方必须先经 `normalizeCssColor` 归一化
     * （它会用浏览器把变量解析成 `rgb()/rgba()`），再交给 `parseRgbaColor`。
     */
    readonly playheadLine: string;
}

/**
 * 播放头颜色 token（唯一定义处）。
 *
 * 指向时间轴标尺所用的同一个变量；两套主题共用，避免"两处播放头各自漂移"。
 */
export const PLAYHEAD_COLOR_TOKEN = "var(--qt-playhead)";

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
    // 钢琴背景 / 音阶高亮（见接口处的取值说明：方向恒为压暗、alpha 有上下限）
    // 深色底接近黑（31），压暗余量小，故 alpha 明显大于浅色主题才能同样看得见
    // （0.08 → Δ2 等于白做；0.25 → Δ7.75，且仍在弱网格线的 Δ11.2 之下）。
    blackKeyRowBand: "rgba(0,0,0,0.25)",
    scaleHighlight: "rgba(255,200,80,0.22)",
    // 曲线
    origCurve: "rgba(200,200,200,0.55)",
    editCurve: "rgba(255,255,255,0.92)",
    selectionCurve: "rgba(100,200,255,0.95)",
    selectionBand: "rgba(100, 200, 255, 0.08)",
    selectionBorder: "rgba(100, 200, 255, 0.30)",
    // 叠加文字 & 播放头（画布中央的操作提示文字，需保持可读：
    // 旧值 35% 不透明度在两套主题下都只剩 1.5-1.8:1）
    overlayTextColor: "rgba(235,240,248,0.45)",
    // 与时间轴标尺同源（见 PLAYHEAD_COLOR_TOKEN 说明），不得写字面色值。
    playheadLine: PLAYHEAD_COLOR_TOKEN,
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
    // 选区块填充 / 边框：**两套主题同值**——迁移前就是硬编码的同一对字面量，
    // 这里只把它挪进配色表（不顺手"优化"浅色主题的取值，那会改变现有观感）。
    selectionBand: "rgba(100, 200, 255, 0.08)",
    selectionBorder: "rgba(100, 200, 255, 0.30)",
    // 叠加文字 & 播放头（画布中央的操作提示文字，需保持可读）
    overlayTextColor: "rgba(30,36,48,0.60)",
    // 与时间轴标尺同源（见 PLAYHEAD_COLOR_TOKEN 说明），不得写字面色值。
    playheadLine: PLAYHEAD_COLOR_TOKEN,
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
        ? [
              "rgba(80, 220, 180, 0.56)",
              "rgba(255, 110, 197, 0.60)",
              "rgba(180, 120, 255, 0.56)",
              "rgba(60, 180, 255, 0.56)",
          ]
        : [
              "rgba(0, 150, 118, 0.80)",
              "rgba(214, 44, 140, 0.75)",
              "rgba(124, 58, 237, 0.70)",
              "rgba(2, 132, 199, 0.80)",
          ];
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
        ? [
              "rgba(100, 200, 255, 0.62)",
              "rgba(255, 110, 197, 0.62)",
              "rgba(180, 120, 255, 0.62)",
              "rgba(60, 200, 160, 0.62)",
          ]
        : [
              "rgba(0, 116, 200, 0.72)",
              "rgba(214, 44, 140, 0.72)",
              "rgba(124, 58, 237, 0.72)",
              "rgba(22, 163, 116, 0.75)",
          ];
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
