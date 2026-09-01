/**
 * dev-only：帧率与分图层耗时探针。
 *
 * 【为什么需要它】用户环境没有 Chrome DevTools 的 Performance 面板，而渲染
 * 链路里**真正剩余的大头（React 提交、浏览器 layout/paint）没有离线基准
 * 可测**——此前两次基于推理的优化一次押错方向（波形 per-clip 开销）、一次
 * 直接导致缩放抖动（flushSync）。没有数据就继续动手，等于继续盲打。
 *
 * 【主要内容】
 * - rAF 循环测帧间隔（FPS / p50 / p95）；
 * - `recordLayer(name, ms)`：由帧提交器按图层上报绘制耗时；
 * - `recordReact(ms)`：由 React `<Profiler>` 上报提交耗时；
 * - 自渲染一个小浮层（右上角），每 500ms 刷新一次聚合快照。
 *
 * 【开销】每层每帧 2 次 `performance.now()` + 尾部一次聚合，量级在微秒，
 * 相对被测对象可忽略。但仍是 dev-only：由 `main.tsx` 在
 * `import.meta.env.DEV` 下启动，生产构建不打包。
 *
 * 【与其他模块的关系】
 * - 数据生产方：`timelineFrameCommitter.commit()`（图层耗时）、
 *   `TimelinePanel` 的 `<Profiler>`（React 提交耗时）。
 * - 通过 `globalThis.__hfsFrameProfiler` 挂钩，**不与生产模块建立静态
 *   import 依赖**——生产模块只做一次属性查找，未启用时为零成本。
 * - 开关：localStorage "hifishifter.frameProfiler" === "1"，PERF 面板有按钮。
 */

/** 每项保留的采样数（约 180 帧 ≈ 3 秒 @60fps）。 */
const SAMPLE_LIMIT = 180;

const layerSamples = new Map<string, number[]>();
let frameSamples: number[] = [];
let reactSamples: number[] = [];
let commitSamples: number[] = [];
let lastFrameTime = 0;
let rafId = 0;
let overlayEl: HTMLDivElement | null = null;
let overlayTimer = 0;
let running = false;

function pushSample(samples: number[], value: number): void {
    samples.push(value);
    if (samples.length > SAMPLE_LIMIT) samples.shift();
}

/** 取分位数（线性插值）；空数组返回 0。 */
function percentile(samples: number[], p: number): number {
    if (samples.length === 0) return 0;
    const sorted = [...samples].sort((a, b) => a - b);
    const idx = Math.min(
        sorted.length - 1,
        Math.max(0, Math.round((p / 100) * (sorted.length - 1))),
    );
    return sorted[idx];
}

function fmt(ms: number): string {
    return ms >= 100 ? ms.toFixed(0) : ms.toFixed(1);
}

/** 聚合快照：各图层 / 帧间隔 / React 提交的分位数。 */
function snapshot(): {
    fps: number;
    frameP50: number;
    frameP95: number;
    commitP50: number;
    commitP95: number;
    reactP50: number;
    reactP95: number;
    layers: Array<{ name: string; p50: number; p95: number }>;
} {
    const layers: Array<{ name: string; p50: number; p95: number }> = [];
    for (const [name, samples] of layerSamples) {
        if (samples.length === 0) continue;
        layers.push({
            name,
            p50: percentile(samples, 50),
            p95: percentile(samples, 95),
        });
    }
    // 按 p95 与 p50 的差降序：抖动最大的图层排最前面。
    layers.sort((a, b) => b.p95 - b.p50 - (a.p95 - a.p50));
    const frameP50 = percentile(frameSamples, 50);
    return {
        fps: frameP50 > 0 ? 1000 / frameP50 : 0,
        frameP50,
        frameP95: percentile(frameSamples, 95),
        commitP50: percentile(commitSamples, 50),
        commitP95: percentile(commitSamples, 95),
        reactP50: percentile(reactSamples, 50),
        reactP95: percentile(reactSamples, 95),
        layers,
    };
}

function renderOverlay(): void {
    if (!overlayEl) return;
    const snap = snapshot();
    const lines = [
        `FPS ${snap.fps.toFixed(0)}  (frame p50 ${fmt(snap.frameP50)} / p95 ${fmt(snap.frameP95)}ms)`,
    ];
    if (snap.commitP50 > 0) {
        lines.push(`commit p50 ${fmt(snap.commitP50)} / p95 ${fmt(snap.commitP95)}ms`);
    }
    for (const layer of snap.layers) {
        lines.push(`  ${layer.name.padEnd(12, " ")} p50 ${fmt(layer.p50)}  p95 ${fmt(layer.p95)}`);
    }
    if (snap.reactP50 > 0) {
        lines.push(`react p50 ${fmt(snap.reactP50)} / p95 ${fmt(snap.reactP95)}ms`);
    }
    overlayEl.textContent = lines.join("\n");
}

/** 全局挂钩：生产模块经它上报，避免静态 import 依赖。 */
const hook = {
    /** 上报一个图层的绘制耗时（毫秒）。 */
    recordLayer(name: string, ms: number): void {
        let samples = layerSamples.get(name);
        if (samples === undefined) {
            samples = [];
            layerSamples.set(name, samples);
        }
        pushSample(samples, ms);
    },
    /** 上报一次帧提交的总耗时（毫秒）。 */
    recordCommit(ms: number): void {
        pushSample(commitSamples, ms);
    },
    /** 上报一次 React 提交的耗时（毫秒，来自 `<Profiler>` 的 actualDuration）。 */
    recordReact(ms: number): void {
        pushSample(reactSamples, ms);
    },
    /** 清空统计（切换场景时用）。 */
    reset(): void {
        layerSamples.clear();
        frameSamples = [];
        reactSamples = [];
        commitSamples = [];
    },
};

/**
 * 启动探针：rAF 测帧间隔 + 渲染浮层。
 *
 * 流程：
 * 1. 建浮层 DOM（右上角，pointer-events 关闭，不干扰操作）；
 * 2. rAF 循环累计帧间隔；
 * 3. 每 500ms 刷新一次浮层文本。
 */
export function startFrameProfiler(): void {
    if (running) return;
    running = true;
    (globalThis as unknown as Record<string, unknown>).__hfsFrameProfiler = hook;

    overlayEl = document.createElement("div");
    overlayEl.setAttribute("data-hs-frame-profiler", "1");
    Object.assign(overlayEl.style, {
        position: "fixed",
        right: "12px",
        top: "12px",
        zIndex: "9999",
        padding: "8px 10px",
        borderRadius: "6px",
        background: "rgba(17, 24, 39, 0.92)",
        color: "#86efac",
        font: "12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace",
        whiteSpace: "pre",
        pointerEvents: "none",
        boxShadow: "0 4px 16px rgba(0, 0, 0, 0.35)",
    } satisfies Partial<CSSStyleDeclaration>);
    overlayEl.textContent = "profiler starting…";
    document.body.appendChild(overlayEl);

    const loop = (time: number): void => {
        if (lastFrameTime > 0) {
            const delta = time - lastFrameTime;
            // 忽略超过 1s 的间隔（标签页切走、断点暂停）。
            if (delta < 1000) pushSample(frameSamples, delta);
        }
        lastFrameTime = time;
        rafId = requestAnimationFrame(loop);
    };
    rafId = requestAnimationFrame(loop);

    overlayTimer = window.setInterval(renderOverlay, 500);
}

/** 停止探针并移除浮层。 */
export function stopFrameProfiler(): void {
    running = false;
    cancelAnimationFrame(rafId);
    window.clearInterval(overlayTimer);
    overlayEl?.remove();
    overlayEl = null;
    lastFrameTime = 0;
    delete (globalThis as unknown as Record<string, unknown>).__hfsFrameProfiler;
}
