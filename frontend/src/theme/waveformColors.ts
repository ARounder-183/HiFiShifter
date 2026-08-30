/**
 * 波形渲染颜色配置
 *
 * 定义深色和浅色主题下的波形填充和描边颜色，
 * 并支持从自定义主题中读取覆盖值。
 */

import type { ThemeMode } from "./AppThemeProvider";
import { loadAppearance } from "./themeStorage";
import { loadCustomThemes } from "./themeStorage";

export interface WaveformColors {
    /** 波形填充颜色 */
    fill: string;
    /** 波形描边颜色 */
    stroke: string;
    /** MIDI 音高线颜色（timeline 上 MIDI clip 的音高预览） */
    midiPitch?: string;
}

/**
 * 深色主题的时间线波形颜色。
 *
 * 深色主题的 clip 色块整体压暗（timelineCanvasStyle 的 darkMode 方案），
 * 波形用**半透明浅色**（~0.80 白）：与浅色文字同方向，在暗色块上清晰可辨，
 * 且底色透出让波形带同色相的浅调（Ableton 深色主题式）。
 * 钢琴卷帘波形画在深色画布上，同为浅色系（见下方 pianoRoll 组）。
 */
const darkTimelineWaveformColors: WaveformColors = {
    fill: "#f2f6fb",
    stroke: "rgba(255, 255, 255, 0.80)",
    midiPitch: "#d8e8fa",
};

/**
 * 浅色主题的时间线波形颜色：中明度低饱和色块 + 深色波形
 * （透明度让底色透出，波形带同色相深调，对比 ≥4:1）
 */
const lightTimelineWaveformColors: WaveformColors = {
    fill: "#080b14",
    stroke: "rgba(0, 0, 0, 0.70)",
    midiPitch: "#002848",
};

// 钢琴卷帘底衬波形：两主题保持同一「蓝灰」色相族（对应参考轨道的语义），
// 深浅各自反向：深色主题 = 浅蓝灰波形，浅色主题 = 深蓝灰波形；亮度都压在
// 编辑包络线之下（背景层 ≈3:1，曲线 ≥3.5:1），只做参照不抢层级。
const darkPianoRollWaveformColors: WaveformColors = {
    fill: "rgba(116,136,160,0.28)",
    stroke: "rgba(168,186,208,0.55)",
};

// 浅色主题：明确的深色波形（旧值 0.48 透明度的中蓝渲染后偏淡，像水洗蓝，
// 不符合"浅色主题 = 深色波形"的方向）。
const lightPianoRollWaveformColors: WaveformColors = {
    fill: "rgba(70,100,135,0.22)",
    stroke: "rgba(45,70,100,0.60)",
};

/**
 * 根据主题模式获取波形颜色配置
 *
 * 优先使用自定义主题中的波形颜色（如果有激活的自定义主题且设置了波形颜色），
 * 否则回退到内置的主题默认波形颜色。
 *
 * @param mode - 主题模式 ('dark' | 'light')
 * @returns 波形颜色配置对象
 *
 * @example
 * const colors = getWaveformColors('dark');
 * // { fill: 'rgba(255,255,255,0.34)', stroke: 'rgba(255,255,255,0.92)' }
 */
export function getWaveformColors(
    mode: ThemeMode,
    surface: "timeline" | "piano-roll" = "timeline",
): WaveformColors {
    // 时间线波形颜色与 Clip 色块**强耦合**（亮色块配深波形），不能被自定义
    // 主题覆盖——否则色块改了波形没跟着改，对比度会失调。
    // 自定义主题只允许覆盖钢琴卷帘波形（画在深色画布上，与色块无关）。
    if (surface === "piano-roll") {
        try {
            const appearance = loadAppearance();
            if (appearance.activeCustomThemeId) {
                const themes = loadCustomThemes();
                const active = themes.find((t) => t.id === appearance.activeCustomThemeId);
                if (active?.waveformColors) {
                    return active.waveformColors;
                }
            }
        } catch {
            // fallthrough to default
        }
        return mode === "dark" ? darkPianoRollWaveformColors : lightPianoRollWaveformColors;
    }
    return mode === "dark" ? darkTimelineWaveformColors : lightTimelineWaveformColors;
}
