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
 * 时间线波形画在**饱和的轨道色块**上（timelineCanvasStyle 的色块方案），
 * 因此波形必须是深色且足够实：alpha 提高以保证在彩色块上能"看穿"色块。
 * 钢琴卷帘波形画在深色画布上，仍为浅色（见下方 pianoRoll 组）。两个
 * surface 的对比方向相反，不可混用。
 */
const darkTimelineWaveformColors: WaveformColors = {
    fill: "rgba(14, 18, 26, 0.78)",
    stroke: "rgba(8, 12, 20, 0.96)",
    midiPitch: "rgba(8, 60, 100, 0.92)",
};

/**
 * 浅色主题的时间线波形颜色（同为深色系：波形画在亮色块上，与主题无关）
 */
const lightTimelineWaveformColors: WaveformColors = {
    fill: "rgba(24, 28, 40, 0.72)",
    stroke: "rgba(16, 20, 30, 0.94)",
    midiPitch: "rgba(4, 72, 120, 0.9)",
};

const darkPianoRollWaveformColors: WaveformColors = {
    fill: "rgba(146,182,218,0.24)",
    stroke: "rgba(214,230,246,0.56)",
};

const lightPianoRollWaveformColors: WaveformColors = {
    fill: "rgba(88,118,152,0.20)",
    stroke: "rgba(58,86,120,0.48)",
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
    // 尝试从自定义主题读取波形颜色
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

    if (surface === "piano-roll") {
        return mode === "dark" ? darkPianoRollWaveformColors : lightPianoRollWaveformColors;
    }
    return mode === "dark" ? darkTimelineWaveformColors : lightTimelineWaveformColors;
}
