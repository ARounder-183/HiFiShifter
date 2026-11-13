/**
 * 参数曲线“翻转”能力判定。
 *
 * 只对关于 0 对称的参数开放：pan、tension、formant_shift_cents、
 * child pitch offset cents/degrees 等。音量/动态是非负幅值，翻转会产生
 * 无意义负值；pitch 以 60 为中心、不是 0 对称，也不在范围内。
 */
export function supportsParamFlip(
    range: { min: number; max: number } | undefined,
    defaultValue: number,
): boolean {
    if (!range) return false;
    const { min, max } = range;
    if (!Number.isFinite(min) || !Number.isFinite(max)) return false;
    return Math.abs(min + max) <= 1e-9 && Math.abs(defaultValue) <= 1e-9;
}

/** 翻转后的值：围绕 0 镜像；保留 0，避免 -0。 */
export function flipParamValue(value: number): number {
    if (!Number.isFinite(value) || value === 0) return 0;
    return -value;
}
