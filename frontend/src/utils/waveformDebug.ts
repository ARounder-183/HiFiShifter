/**
 * waveformDebug — 波形渲染诊断工具
 *
 * 通过 localStorage 开关启用:
 *   localStorage.setItem('hifishifter.debugWaveform', '1')
 * 关闭:
 *   localStorage.removeItem('hifishifter.debugWaveform')
 *
 * 每 2 秒输出一次汇总，包含:
 *   - 帧统计（总帧数、慢帧数、平均帧时间）
 *   - 数据缺失统计（getInterleavedSlice 返回 null 的次数 —— 闪烁主因）
 *   - Buffer 池统计（命中率、当前池大小）
 *   - invalidate 调用次数 vs 实际绘制次数
 */
const INTERVAL_MS = 2000;

let _enabled = false;
function isEnabled(): boolean {
    return (
        typeof window !== "undefined" &&
        window.localStorage?.getItem("hifishifter.debugWaveform") === "1"
    );
}

// ── 帧统计 ──────────────────────────────────────────
let _frameTotal = 0;
let _frameSlow = 0; // > 16ms
let _frameTimeSum = 0;
let _frameTimeMax = 0;

// ── 绘制触发源统计 ──────────────────────────────────
let _invalidateBusCalls = 0; // 来自 timelineViewportBus（滚动）
let _invalidateMipmapCalls = 0; // 来自 mipmap "done" 事件（数据加载）
let _invalidatePropCalls = 0; // 来自 React props 变化
let _drawCount = 0; // 实际执行 drawRef 的次数

// ── 数据缺失统计（闪烁主因） ─────────────────────────
let _missNullCount = 0; // getInterleavedSlice 返回 null
let _missShortCount = 0; // interleaved.length < 4
let _hitCount = 0; // 正常命中

// ── Buffer 池统计 ───────────────────────────────────
interface PoolStats {
    acquire: number;
    miss: number; // 池未命中 → new allocation
    release: number;
    discard: number; // 池满丢弃
    currentSize: () => number;
}
const _poolStats: Record<string, PoolStats> = {};

// ── 定时器 ──────────────────────────────────────────
let _timer: ReturnType<typeof setInterval> | null = null;

function ensureTimer(): void {
    if (_timer != null) return;
    _timer = setInterval(() => {
        if (!isEnabled()) {
            if (_timer != null) { clearInterval(_timer); _timer = null; }
            reset();
            return;
        }
        dump();
    }, INTERVAL_MS);
}

function reset(): void {
    _frameTotal = 0; _frameSlow = 0; _frameTimeSum = 0; _frameTimeMax = 0;
    _invalidateBusCalls = 0; _invalidateMipmapCalls = 0; _invalidatePropCalls = 0; _drawCount = 0;
    _missNullCount = 0; _missShortCount = 0; _hitCount = 0;
    for (const s of Object.values(_poolStats)) {
        s.acquire = 0; s.miss = 0; s.release = 0; s.discard = 0;
    }
}

function dump(): void {
    const avg = _frameTotal > 0 ? (_frameTimeSum / _frameTotal).toFixed(1) : "0";
    const missTotal = _missNullCount + _missShortCount;
    const missPct = (_hitCount + missTotal) > 0
        ? ((missTotal / (_hitCount + missTotal)) * 100).toFixed(1)
        : "0";

    console.log(
        `%c[WaveformDiag]──────── ${new Date().toLocaleTimeString()} ────────`,
        "font-weight:bold;color:#0af",
    );
    console.log(
        `  frames: ${_frameTotal} total | ${_frameSlow} slow(>16ms) | avg=${avg}ms | max=${_frameTimeMax.toFixed(1)}ms | draws=${_drawCount}`,
    );
    console.log(
        `  invalidate src: bus=${_invalidateBusCalls} mipmap=${_invalidateMipmapCalls} props=${_invalidatePropCalls}`,
    );
    console.log(
        `  data: hit=${_hitCount} | miss(null)=${_missNullCount} miss(short)=${_missShortCount} | missRate=${missPct}%`,
        missTotal > _hitCount * 0.1 ? "color:red;font-weight:bold" : "",
    );

    for (const [name, s] of Object.entries(_poolStats)) {
        const total = s.acquire || 1;
        const hitRate = ((1 - s.miss / total) * 100).toFixed(0);
        console.log(
            `  pool[${name}]: acq=${s.acquire} miss=${s.miss}(${hitRate}% hit) | rel=${s.release} discard=${s.discard} | sz=${s.currentSize()}`,
        );
    }

    if (_mipmapSizeFn) {
        console.log(`  mipmapCache: ${_mipmapSizeFn()} files`);
    }
}

// ── 公开 API ────────────────────────────────────────

export function wfDiag_frameStart(): void {
    if (!_enabled) { _enabled = isEnabled(); if (!_enabled) return; }
    ensureTimer();
}

export function wfDiag_frameEnd(totalMs: number): void {
    if (!_enabled) return;
    _frameTotal++;
    _frameTimeSum += totalMs;
    if (totalMs > _frameTimeMax) _frameTimeMax = totalMs;
    if (totalMs > 16) _frameSlow++;
    _drawCount++;
}

export function wfDiag_invalidateBus(): void {
    if (!_enabled) { _enabled = isEnabled(); if (!_enabled) return; }
    _invalidateBusCalls++;
}

export function wfDiag_invalidateMipmap(): void {
    if (!_enabled) { _enabled = isEnabled(); if (!_enabled) return; }
    _invalidateMipmapCalls++;
}

export function wfDiag_invalidateProps(): void {
    if (!_enabled) { _enabled = isEnabled(); if (!_enabled) return; }
    _invalidatePropCalls++;
}

export function wfDiag_dataHit(): void {
    if (!_enabled) return;
    _hitCount++;
}

export function wfDiag_dataMissNull(): void {
    if (!_enabled) return;
    _missNullCount++;
}

export function wfDiag_dataMissShort(): void {
    if (!_enabled) return;
    _missShortCount++;
}

// ── Buffer 池跟踪 ───────────────────────────────────

export function wfDiag_poolRegister(
    name: string,
    currentSize: () => number,
): void {
    if (!_poolStats[name]) {
        _poolStats[name] = { acquire: 0, miss: 0, release: 0, discard: 0, currentSize };
    }
}

export function wfDiag_poolAcquire(name: string, hit: boolean): void {
    if (!_enabled) { _enabled = isEnabled(); if (!_enabled) return; }
    const s = _poolStats[name];
    if (!s) return;
    s.acquire++;
    if (!hit) s.miss++;
}

export function wfDiag_poolRelease(name: string, accepted: boolean): void {
    if (!_enabled) return;
    const s = _poolStats[name];
    if (!s) return;
    s.release++;
    if (!accepted) s.discard++;
}

// ── Mipmap 缓存统计 ─────────────────────────────────
let _mipmapSizeFn: (() => number) | null = null;
export function wfDiag_setMipmapSizeFn(fn: () => number): void {
    _mipmapSizeFn = fn;
}

export function wfDiag_reset(): void {
    reset();
}
