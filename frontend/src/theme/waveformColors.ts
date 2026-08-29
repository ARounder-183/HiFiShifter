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
 * 采用半透明深黑：不透明度 ~0.73 叠在色块上呈现"深色调波形"（REAPER 式），
 * 轨道色仍从波形中透出，既保证对比度又不至于把色块完全盖死（全不透明
 * 纯黑会吞掉整个色块，见用户实测反馈）。
 * 钢琴卷帘波形画在深色画布上，仍为浅色（见下方 pianoRoll 组）。两个
 * surface 的对比方向相反，不可混用。
 */
const darkTimelineWaveformColors: WaveformColors = {
    fill: "#04060c",
    stroke: "rgba(0, 0, 0, 0.73)",
    midiPitch: "#02203a",
};

/**
 * 浅色主题的时间线波形颜色（同为深色系：波形画在亮色块上，与主题无关）
 */
const lightTimelineWaveformColors: WaveformColors = {
    fill: "#080b14",
    stroke: "rgba(0, 0, 0, 0.73)",
    midiPitch: "#002848",
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
