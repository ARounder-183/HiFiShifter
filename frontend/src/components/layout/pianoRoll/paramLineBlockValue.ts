/** Translation-equivariant representative of this exact curve window. */
export function paramLineBlockValue(
    startSec: number,
    endSec: number,
    samples: {
        readonly values: readonly number[];
        readonly startFrame: number;
        readonly stride: number;
        readonly framePeriodMs: number;
        readonly isPitch: boolean;
    },
): number | null {
    const stride = Math.max(1, samples.stride);
    const first = Math.max(
        0,
        Math.ceil(((startSec * 1000) / samples.framePeriodMs - samples.startFrame) / stride - 1e-9),
    );
    const last = Math.min(
        samples.values.length - 1,
        Math.ceil(((endSec * 1000) / samples.framePeriodMs - samples.startFrame) / stride - 1e-9) -
            1,
    );
    if (last < first) return null;
    const values: number[] = [];
    for (let index = first; index <= last; index++) {
        const value = samples.values[index];
        if (Number.isFinite(value) && (!samples.isPitch || value > 0)) values.push(value);
    }
    if (!values.length) return null;
    return median(values);
}

export function paramLineValueCenter(values: readonly number[], isPitch: boolean): number | null {
    return median(values.filter((value) => Number.isFinite(value) && (!isPitch || value > 0)));
}

function median(values: number[]): number | null {
    if (!values.length) return null;
    values.sort((a, b) => a - b);
    const middle = Math.floor(values.length / 2);
    return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}
