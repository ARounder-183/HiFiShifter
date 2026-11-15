import { createDebugLogger } from "./debugLog";

/**
 * waveformDebug — 波形缓冲池诊断工具
 *
 * 由「帮助 → 日志详细程度」的「高」档驱动（不再读 localStorage 独立开关）。
 * 每 2 秒输出一次汇总，包含:
 *   - Buffer 池统计（命中率、当前池大小）
 *   - mipmap 缓存规模
 *
 * 【与 dev/frameProfiler.ts 的分工】帧统计（总帧数/慢帧/平均帧时间）已由
 * `__hfsFrameProfiler` 接管，本模块不再重复；曾经的数据缺失统计
 * （getInterleavedSlice 返回 null 的计数）随其消费方重构而失去挂点，已删除 ——
 * 需要时在取数路径重新接线即可，不要留下无人调用的死计数器。
 */
const INTERVAL_MS = 2000;
const waveformDebugLogger = createDebugLogger("waveform");

let _enabled = false;
/**
 * 档位探测节流：这些 wfDiag_* 钩子挂在缓冲池热路径上，逐次查策略会拖慢
 * 热路径。策略变化最多 2 秒后生效，与汇总输出节奏一致；「无」档下恒关闭。
 */
let _lastFlagCheck = 0;
function isEnabled(): boolean {
    const now = Date.now();
    if (now - _lastFlagCheck < INTERVAL_MS) return _enabled;
    _lastFlagCheck = now;
    _enabled = waveformDebugLogger.enabled;
    return _enabled;
}

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
            if (_timer != null) {
                clearInterval(_timer);
                _timer = null;
            }
            // 关闭时同步复位内部使能标记：否则后续调用继续累计计数却永不
            // 输出，重新打开开关时会先看到一段错位的旧计数。
            _enabled = false;
            reset();
            return;
        }
        dump();
    }, INTERVAL_MS);
}

function reset(): void {
    for (const s of Object.values(_poolStats)) {
        s.acquire = 0;
        s.miss = 0;
        s.release = 0;
        s.discard = 0;
    }
}

function dump(): void {
    waveformDebugLogger.log(
        `%c[WaveformDiag]──────── ${new Date().toLocaleTimeString()} ────────`,
        "font-weight:bold;color:#0af",
    );

    for (const [name, s] of Object.entries(_poolStats)) {
        const total = s.acquire || 1;
        const hitRate = ((1 - s.miss / total) * 100).toFixed(0);
        waveformDebugLogger.log(
            `  pool[${name}]: acq=${s.acquire} miss=${s.miss}(${hitRate}% hit) | rel=${s.release} discard=${s.discard} | sz=${s.currentSize()}`,
        );
    }

    if (_mipmapSizeFn) {
        waveformDebugLogger.log(`  mipmapCache: ${_mipmapSizeFn()} files`);
    }
}

// ── 公开 API ────────────────────────────────────────

// ── Buffer 池跟踪 ───────────────────────────────────

export function wfDiag_poolRegister(name: string, currentSize: () => number): void {
    if (!_poolStats[name]) {
        _poolStats[name] = { acquire: 0, miss: 0, release: 0, discard: 0, currentSize };
    }
}

export function wfDiag_poolAcquire(name: string, hit: boolean): void {
    if (!_enabled) {
        _enabled = isEnabled();
        if (!_enabled) return;
    }
    // 汇总输出的唯一启动点：第一个进入热路径的钩子顺手把定时器立起来；
    // 「无」档下 isEnabled() 为 false，定时器永不起。
    ensureTimer();
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
