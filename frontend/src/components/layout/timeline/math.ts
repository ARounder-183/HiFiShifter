import { formatGainDbValue as formatGainDbValueFromGainModel } from "../../../utils/gainDb";

export function clamp(value: number, minValue: number, maxValue: number): number {
    return Math.min(maxValue, Math.max(minValue, value));
}

export function gainToDb(gain: number): number {
    // 【不要再夹一个 1e-4 的下限】那会让 −80 dB 以下全部并成同一个值，而展示需要
    // 一路下探到 −150 dB（低于它才显示 −inf，见 utils/gainDb 的显示下界）。
    const g = Number(gain);
    if (!Number.isFinite(g) || g <= 0) return Number.NEGATIVE_INFINITY;
    return 20 * Math.log10(g);
}

export function dbToGain(db: number): number {
    return Math.pow(10, db / 20);
}

/**
 * 增益 dB 的展示文本。
 *
 * 【为什么只是转手】增益格式化必须**只有一套**规则（含 `-inf` 与 −150 dB 的边界）。
 * 这里曾有一份自己的实现（没有 −inf 分支、也没有下界概念），于是同一个值在时间轴、
 * 轨道推子与参数编辑器上写法不同。权威实现见 `utils/gainDb`。
 */
export function formatGainDbValue(db: number): string {
    return formatGainDbValueFromGainModel(db);
}

/**
 * 编辑态数值文本：保留 6 位小数精度并去除尾零与浮点噪声。
 *
 * ★ 编辑中的输入框禁止使用 2 位小数等展示级取整——例如由时长/BPM 换算出的
 *   倍率 1.2456 若被取整为 1.25，提交后的实际倍率就与用户输入的时长不再
 *   精确对应。展示态（非编辑）仍可用各自的展示级格式化取整。
 */
export function formatEditNumber(value: number): string {
    if (!Number.isFinite(value)) return "0";
    return String(Number(value.toFixed(6)));
}
