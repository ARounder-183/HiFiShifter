/** Map a GridSize string to beat fraction. Default = 1 beat (quarter note). */
export function gridStepBeats(grid: string): number {
    // Base values (in beats, where 1 beat = quarter note)
    const base: Record<string, number> = {
        "1/1": 4,
        "1/2": 2,
        "1/4": 1,
        "1/8": 0.5,
        "1/16": 0.25,
        "1/32": 0.125,
        "1/64": 0.0625,
    };

    // Dotted: 1.5× base
    if (grid.endsWith("d")) {
        const root = grid.slice(0, -1);
        const b = base[root];
        if (b !== undefined) return b * 1.5;
    }
    // Triplet: 2/3× base
    if (grid.endsWith("t")) {
        const root = grid.slice(0, -1);
        const b = base[root];
        if (b !== undefined) return (b * 2) / 3;
    }

    return base[grid] ?? 1;
}
