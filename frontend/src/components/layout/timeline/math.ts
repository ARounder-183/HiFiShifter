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
