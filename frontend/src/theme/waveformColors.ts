/**
 * 波形渲染颜色配置
 *
 * 定义深色和浅色主题下的波形填充和描边颜色
 */

import type { ThemeMode } from "./AppThemeProvider";

export interface WaveformColors {
    /** 波形填充颜色 */
    fill: string;
    /** 波形描边颜色 */
    stroke: string;
}

/**
 * 深色主题波形颜色
 */
const darkWaveformColors: WaveformColors = {
    fill: "rgba(255,255,255,0.34)",
    stroke: "rgba(255,255,255,0.92)",
};

/**
 * 浅色主题波形颜色
 */
const lightWaveformColors: WaveformColors = {
    fill: "rgba(0,0,0,0.28)",
    stroke: "rgba(0,0,0,0.82)",
};

/**
 * 根据主题模式获取波形颜色配置
 *
 * @param mode - 主题模式 ('dark' | 'light')
 * @returns 波形颜色配置对象
 *
 * @example
 * const colors = getWaveformColors('dark');
 * // { fill: 'rgba(255,255,255,0.2)', stroke: 'rgba(255,255,255,0.7)' }
 */
export function getWaveformColors(mode: ThemeMode): WaveformColors {
    return mode === "dark" ? darkWaveformColors : lightWaveformColors;
}
