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
