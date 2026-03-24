// Worker to downsample/aggregate waveform peaks to a target width.
// Messages:
// { id:number, type: 'process', min: Float32Array | number[], max: Float32Array | number[], targetWidth: number }
// Response:
// { id:number, min: Float32Array, max: Float32Array }

self.addEventListener('message', (ev) => {
    const data = ev.data as any;
    if (!data || data.type !== 'process') return;
    const id: number = data.id;
    try {
        const srcMin = new Float32Array(data.min);
        const srcMax = new Float32Array(data.max);
        const target = Math.max(1, Math.floor(Number(data.targetWidth) || 1));
        const n = srcMin.length;
        const outMin = new Float32Array(target);
        const outMax = new Float32Array(target);

        if (n === 0) {
            for (let i = 0; i < target; i++) {
                outMin[i] = 0;
                outMax[i] = 0;
            }
        } else if (n <= target) {
            // Upsample via linear interpolation
            for (let i = 0; i < target; i++) {
                const x = (i * (n - 1)) / Math.max(1, target - 1);
                const i0 = Math.floor(x);
                const i1 = Math.min(n - 1, i0 + 1);
                const f = x - i0;
                outMin[i] = srcMin[i0] * (1 - f) + srcMin[i1] * f;
                outMax[i] = srcMax[i0] * (1 - f) + srcMax[i1] * f;
            }
        } else {
            // Downsample by taking min/max over source buckets
            for (let i = 0; i < target; i++) {
                const s0 = Math.floor((i * n) / target);
                const s1 = Math.floor(((i + 1) * n) / target);
                const start = Math.max(0, Math.min(n - 1, s0));
                const end = Math.max(start + 1, Math.min(n, s1));
                let mn = Number.POSITIVE_INFINITY;
                let mx = Number.NEGATIVE_INFINITY;
                for (let j = start; j < end; j++) {
                    const a = srcMin[j] ?? 0;
                    const b = srcMax[j] ?? 0;
                    if (a < mn) mn = a;
                    if (b > mx) mx = b;
                }
                if (!Number.isFinite(mn)) mn = 0;
                if (!Number.isFinite(mx)) mx = 0;
                outMin[i] = mn;
                outMax[i] = mx;
            }
        }

        // Transfer buffers back to main thread
        (self as any).postMessage({ id, min: outMin, max: outMax }, [outMin.buffer, outMax.buffer]);
    } catch (e) {
        (self as any).postMessage({ id, error: String(e) });
    }
});
