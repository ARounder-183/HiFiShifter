export function clamp(value: number, minValue: number, maxValue: number): number {
    return Math.min(maxValue, Math.max(minValue, value));
}

export function gainToDb(gain: number): number {
    const g = Math.max(1e-4, Number(gain) || 1);
    return 20 * Math.log10(g);
}

export function dbToGain(db: number): number {
    return Math.pow(10, db / 20);
}

export function formatGainDbValue(db: number): string {
    const rounded = Math.round(db * 10) / 10;
    if (Math.abs(rounded) < 0.05) return "0";
    const sign = rounded > 0 ? "+" : "";
    const value = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
    return `${sign}${value}`;
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