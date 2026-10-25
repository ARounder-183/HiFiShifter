/**
 * 参数编辑器面板的模块级常量与纯函数。
 *
 * 【为什么单独成文件】这些是"面板内部工具"而不是"面板本身"：音名表、播放头配色的
 * 缓存、参数工具栏的排序权重、帧区间换算。它们不碰任何 hook 状态、也不渲染，留在
 * 面板文件里只会把那个组件推得更远。搬出来之后，面板文件剩下的每一行都确实与
 * "这个面板怎么跑起来"有关。
 *
 * 【边界】这里的函数必须保持**纯**（除 `playheadRgbaCache` 这个进程级缓存外）。
 * 需要读 store / ref / DOM 的东西不属于这里。
 */
import { resolveEffectivePitchAlgo } from "../../../../features/tracks/pitchAlgoOptions";
import { timeToFrame } from "../utils";
import { PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX } from "../kernel/scroll/verticalValueScroll";
import { normalizeCssColor, resolvePianoRollColors } from "../colors";
import { parseRgbaColor } from "../../timeline/runtime/timelineClipGlRenderer";

export const NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
export const PARAM_EDITOR_VERTICAL_SCROLL_RANGE_PX = PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX;

/** 「添加颤音」预览抓取的帧数上限（≈2s，够看清若干周期与渐入）。 */
export const VIBRATO_APPLY_PREVIEW_FRAMES = 400;

/**
 * **不参与「无选区时先全选」** 的操作（见 `handleEditOp` 开头的隐式全选）。
 *
 * 参数编辑器里的操作绝大多数以**参数选区**为作用域（复制/剪切、初始化、
 * 各类对话框编辑、音量↔动态互转、另存为音高参考、导出 MIDI…）：没有选区时它们
 * 原本静默什么都不做，用户得先自己"全选"再点一次。这些一律先全选再执行。
 *
 * 只有三类例外：
 * - **选区自身的命令**：`selectAll` / `deselect` —— 它们就是在操作选区，
 *   "先全选再执行"会把"取消选择"变成"全选"；
 * - **作用域来自剪辑选择**的命令：`selectClipParamRange` /
 *   `addClipsToParamSelection` / `removeClipsFromParamSelection` —— 它们按被选中的
 *   剪辑（或其起止时间）改写参数选区，与"当前有没有参数选区"无关；
 * - **`paste`** —— 粘贴的作用对象完全由**剪贴板**决定：无选区时按"播放光标作为
 *   复制起点 + 剪贴板自己的段布局"推导选区（见 `pasteTargetSelectionFromClipboard`）。
 *   在这里全选会把数据摊到整条曲线上（起点跑到工程开头、断层被拉长）。
 */
export const SELECTION_SCOPE_EXEMPT_OPS: ReadonlySet<string> = new Set([
    "selectAll",
    "deselect",
    "selectClipParamRange",
    "addClipsToParamSelection",
    "removeClipsFromParamSelection",
    "paste",
]);

/**
 * 音频块的时间范围（秒）→ 选区**帧边界**（半开 `[startBound, endBound)`）。
 *
 * 起点向下取整、终点向上取整（沿用改造前 `floor(起点) / ceil(终点)` 的约定）：
 * 块两端**部分覆盖**的帧也算在范围内，不会在边界处悄悄丢掉一帧。
 *
 * 【为什么需要这一层】音频块是秒制（`startSec` / `lengthSec`，与 BPM 无关），
 * 选区是帧制；两者只在"按块选参数范围"这类入口处交界，故集中在此一处换算。
 *
 * 注意结果走**数据路径**（`selectionFromFrames` / `addFrameRanges`），不经过
 * "切点"量化：块的起止是绝对时间，不该被"最近中点"再挪半帧。
 */
export function clipTimeRangeToFrameBounds(
    clip: { startSec: number; lengthSec: number },
    framePeriodMs: number,
): { startBound: number; endBound: number } {
    const fp = Math.max(1e-6, framePeriodMs);
    const startBound = Math.max(0, timeToFrame(clip.startSec, fp));
    const endBound = Math.max(
        startBound,
        Math.ceil(((clip.startSec + clip.lengthSec) * 1000) / fp),
    );
    return { startBound, endBound };
}

/**
 * 本面板在共享视口中的来源标识。
 *
 * 与时间轴的 `TIMELINE_SYNC_ORIGIN` 成对：双方发布时登记来源，订阅回调据此
 * 跳过"自己造成的广播"（否则时间轴会把自己的值再应用一遍并回退）。
 */
export const PIANO_ROLL_SYNC_ORIGIN = "pianoRoll";

/**
 * 把 `getFixedDashPattern` 的返回值收窄为元组。
 *
 * 特殊说明：该函数返回 `number[]`（历史签名，也被 Canvas2D 的 `setLineDash` 消费），
 * 而曲线图层描述符要求 `[dash, gap]` 二元组。这里做一次显式收窄而不是改它的签名——
 * 改签名会波及 Canvas2D 路径，而那条路径是已充分验证的。长度不符时返回 null
 * （视为实线），宁可退化也不产出非法图案。
 *
 * @param pattern `[dash, gap]`。
 * @returns 二元组；长度不符时为 null。
 */
export function toDashTuple(pattern: readonly number[]): [number, number] | null {
    return pattern.length >= 2 ? [pattern[0], pattern[1]] : null;
}

/**
 * 判断数值是否值得写入 DOM（跨过容差）。
 *
 * 【为什么要取反写法】`previous` 为 NaN（从未写入）时 `Math.abs(next - NaN) <= eps`
 * 恒为 false，若正着写会得到「无需写入」，表现为首帧完全不写。取反让 NaN 稳定走
 * 「需要写入」分支（与内核宿主同一写法）。
 *
 * @param next 本次要写入的值。
 * @param previous 上一次写入的值（NaN = 从未写入）。
 * @param epsilon 视为「无变化」的最大差值。
 * @returns 需要写入时为 true。
 */
export function shouldWriteNumber(next: number, previous: number, epsilon = 0.01): boolean {
    return !(Math.abs(next - previous) <= epsilon);
}

/** 播放头 RGBA 的解析缓存（键 = 解析后的 CSS 颜色字符串）。 */
const playheadRgbaCache = new Map<string, [number, number, number, number]>();

/**
 * 解析参数编辑器播放头的 GL 颜色（与上方标尺同源）。
 *
 * 流程：按主题取配色表里的 `playheadLine`（它是指向 `--qt-playhead` 的 CSS 变量，
 * 与时间轴标尺同一来源）→ 经 `normalizeCssColor` 借浏览器解析成 `rgb()/rgba()`
 * → `parseRgbaColor` 转 0..1 浮点。
 *
 * 【为什么按「解析后的字符串」做缓存】本函数在每帧的绘制提交路径上被调用，而
 * `normalizeCssColor` 会挂 DOM 探针读 `getComputedStyle`——裸调等于把样式重算拖进
 * 每一帧（与 `resolveThemeColor` 的缓存理由相同）。主题或自定义主题色一变，解析
 * 结果字符串就变，键随之变化、缓存自然失效；反之稳态下每帧命中缓存、零重算。
 *
 * 特殊说明 1：GL 不认 `var(...)`，所以**必须**先归一化；`parseRgbaColor` 只认
 * `rgb()/rgba()`，直接喂变量会被解析成不透明洋红。
 *
 * 特殊说明 2（**为什么不能省掉这一步、让宿主兜底**）宿主在缺省时用一个硬编码
 * `[0,0,0,0.2]`——那是迁移期的占位值，与 `--qt-playhead` 毫无关系。参数编辑器的
 * Canvas2D 播放头已被 `skipPlayhead: true` 永久跳过，GL 是唯一绘制者，因此"不喂"
 * 就等于"画布与标尺两个颜色"（用户报告："播放线在底下和上方标尺的颜色不一致"）。
 *
 * 特殊说明 3：`themeMode` 参与键（经配色表间接体现），但这里额外接收它只是为了
 * 让调用方语义清晰——缓存键用的是解析结果，本身已包含主题信息。
 *
 * @param themeMode 当前主题模式（决定取哪套配色表）。
 * @returns 归一化后的 RGBA（0..1 浮点）。
 */
export function resolvePlayheadRgba(themeMode: "dark" | "light"): [number, number, number, number] {
    const css = resolvePianoRollColors(themeMode === "dark").playheadLine;
    const normalized = normalizeCssColor(css);
    const cached = playheadRgbaCache.get(normalized);
    if (cached !== undefined) return cached;
    const parsed = parseRgbaColor(normalized);
    const rgba: [number, number, number, number] = parsed.every((v) => Number.isFinite(v))
        ? [parsed[0], parsed[1], parsed[2], parsed[3]]
        : // 解析失败时退回与标尺同色相的实色（`--qt-playhead` 的深色默认值），
          // 而不是透明/洋红：宁可颜色略有偏差，也不要播放头看不见。
          [0xf0 / 255, 0x5a / 255, 0x5a / 255, 1];
    playheadRgbaCache.set(normalized, rgba);
    return rgba;
}

/**
 * 参数编辑器工具栏的参数显示顺序排名（数值越小越靠左）。
 * - 「音高」为核心参数，固定在最左侧（在 JSX 中单独渲染，不在此排序）；
 * - 「音量/声像」是所有算法的共通参数，固定在最右侧；
 * - 「动态」紧挨音量（两者共用一个药丸呈现，排序只需保证相邻）；
 * - 中间参数随算法不同而变化。
 */
export function getParamToolbarRank(paramId: string, algo: string | undefined | null): number {
    // 先归一：参数**集**来自后端，后端对未识别的算法按默认算法（nsf-hifigan）
    // 处理（见 `PitchAnalysisAlgo::effective`）。这里若拿原始字符串分支，
    // 未识别的轨道会"取到 nsf-hifigan 的参数、却按 world 的顺序排"。
    switch (resolveEffectivePitchAlgo(algo)) {
        case "nsf_hifigan_onnx":
            // 音高、共振峰、气声音量、张力、音量、动态、声像
            switch (paramId) {
                case "formant_shift_cents":
                    return 10;
                case "breath_gain":
                    return 20;
                case "hifigan_tension":
                    return 30;
                case "volume":
                    return 90;
                case "dyn":
                    return 91;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
        case "vslib":
            // 音高、共振峰、气声强度、音量、动态、声像
            switch (paramId) {
                case "formant_shift_cents":
                    return 10;
                case "breathiness":
                    return 20;
                case "volume":
                    return 90;
                case "dyn":
                    return 91;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
        default:
            // world / none（以及 `resolveEffectivePitchAlgo` 归一后的其余取值）：
            // 仅保证音量/动态/声像在右侧，其余保持后端顺序
            switch (paramId) {
                case "volume":
                    return 90;
                case "dyn":
                    return 91;
                case "pan":
                    return 100;
                default:
                    return 50;
            }
    }
}

export function sameStringArray(a: string[], b: string[]) {
    if (a.length !== b.length) return false;
    return a.every((value, index) => value === b[index]);
}
