/**
 * 时间标尺高度常量（供时间标尺与左右角框对齐使用）。
 * 独立成模块以保持组件文件的 Fast Refresh 友好。
 */
import { TEMPO_ROW_HEIGHT_PX } from "./TempoMapRulerRow";

/** 时间标尺基础高度（时间单位区域）。 */
export const RULER_BASE_HEIGHT_PX = 48;

/** Tempo Map 行总高度（分隔线 + 行）。 */
export const TEMPO_ROW_TOTAL_PX = TEMPO_ROW_HEIGHT_PX + 4;

/** 计算时间标尺总高度（Tempo Map 行可见时额外增高）。 */
export function timeRulerHeightPx(showTempoRow: boolean): number {
    return RULER_BASE_HEIGHT_PX + (showTempoRow ? TEMPO_ROW_TOTAL_PX : 0);
}
