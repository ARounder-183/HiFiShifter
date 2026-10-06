/**
 * 标尺刻度的**视口覆盖率**回归 —— 直接测量用户可见量。
 *
 * 【为什么单独立一份】"水平缩放时标尺刻度与文本在某段之内消失"被修过三轮都没
 * 修好，根因是三轮的判据都选错了：它们都在优化"相邻标签**间距**之比"，而用户看到
 * 的是"视口里**还有没有**刻度与文本"。间距是**内部量**，视口是否有刻度是**边缘量**
 * —— 间距可以完美（1.99×）而视口内标签数为 0，因此那种扫描对本题在结构上是盲的。
 *
 * 本文件复刻完整渲染管线（生成 → 切片 → 平移 → 视口裁剪），只断言用户可见量：
 * 1. 真实视口内的带标签刻度**不得被切片丢掉**（主回归锁）；
 * 2. 视口不得为空（缩放大到视口内不足一拍时跳过）；
 * 3. 视口边缘到最近标签的距离有界（不得出现空白段）。
 *
 * 【根因】切片缓冲 `tickWindowBufferPx` 的下界只吸收了提交滞后（`LAG`），
 * 漏掉了锚点量化（`STEP`），而真实视口相对锚点最多右移 `STEP + LAG`。窄视口
 * （`vw ≤ 640`，此时 `vw * 0.5` 取不到）下切片右边界因此落在真实视口**之内**，
 * 把视口右端的刻度整段切掉。
 */

import { describe, expect, it } from "vitest";

import { buildTimelineTicks } from "./buildTimelineTicks.js";
import { createTickAxis } from "./tickAxis.js";
import { TICK_WINDOW_LAG_PX, TICK_WINDOW_STEP_PX, tickWindowRangePx } from "./tickWindow.js";
import type { TempoMap } from "../../../../utils/tempoMap.ts";

const MIN_LABEL_SPACING_PX = 110;
/** 滞后上界 = 锚点量化 + 提交死区。 */
const MAX_LAG_PX = TICK_WINDOW_STEP_PX + TICK_WINDOW_LAG_PX;

/** 密集 Tempo Map（与 labels / renderPipeline 两份测试同构）。 */
function denseTempoMap(): TempoMap {
    const bpms = [80, 159, 96, 128, 80, 159, 96, 128];
    return {
        points: bpms.map((bpm, i) => ({
            id: `d${i}`,
            positionSec: i * 7.3,
            bpm,
            timeSignature: { numerator: 4, denominator: 4 },
            scale: null,
        })),
    };
}

interface Rendered {
    /** 屏幕坐标（相对视口左缘）下、位于视口内的带标签刻度，升序。 */
    xs: number[];
    /** 位于真实视口内、却被切片边界丢掉的带标签刻度数。 */
    lost: number;
}

/**
 * 复刻生产管线：
 * 1. `createTickAxis` + `buildTimelineTicks` —— React 侧生成（窗口宽 `vw + STEP`）；
 * 2. `tickWindowRangePx` + 二分 —— `TimeRulerMarks` 的切片；
 * 3. 减去真值平移 —— 内核写的内容层 transform。
 *
 * 注意 `reactScroll` 与 `trueScroll` 是两个**不同**的量：前者是 React 量化提交后
 * 的位置（决定生成窗口），后者是内核实时位置（决定屏幕上刻度的落点）。二者的差
 * 就是提交滞后。
 */
function renderPipeline(args: {
    pxPerSec: number;
    reactScroll: number;
    trueScroll: number;
    viewportWidth: number;
    tempoMap: TempoMap | null;
}): Rendered {
    const { axis, anchorPx } = createTickAxis({
        pxPerSec: args.pxPerSec,
        scrollLeftPx: args.reactScroll,
        viewportWidthPx: args.viewportWidth,
    });
    const ticks = buildTimelineTicks({
        axis,
        bpm: 120,
        beatsPerBar: 4,
        grid: "1/4",
        primaryUnit: "barBeats",
        secondaryUnit: "clock",
        minLabelSpacingPx: MIN_LABEL_SPACING_PX,
        minGridSpacingPx: 8,
        swingPercent: 0,
        tempoMap: args.tempoMap,
    });
    const labeled = ticks.filter((tick) => tick.showLabel);

    // ── 2. 切片（与 `TimeRulerMarks` 同一入口与公式）──
    const { bufferPx } = tickWindowRangePx(args.viewportWidth);
    const leftPx = Math.max(0, anchorPx - bufferPx);
    const rightPx = anchorPx + args.viewportWidth + bufferPx;
    const lowerBound = (target: number): number => {
        let lo = 0;
        let hi = labeled.length;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (labeled[mid].contentPx < target) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    };
    const start = Math.max(0, lowerBound(leftPx) - 1);
    const end = Math.min(labeled.length, lowerBound(rightPx) + 1);
    const sliced = labeled.slice(start, end);

    // ── 3. 真实视口与屏幕坐标 ──
    const lo = args.trueScroll;
    const hi = args.trueScroll + args.viewportWidth;
    const inTrueViewport = labeled.filter((tick) => tick.contentPx >= lo && tick.contentPx <= hi);
    const kept = new Set(sliced);
    const lost = inTrueViewport.filter((tick) => !kept.has(tick)).length;

    const xs = sliced
        .map((tick) => tick.contentPx - args.trueScroll)
        .filter((x) => x >= 0 && x <= args.viewportWidth)
        .sort((a, b) => a - b);
    return { xs, lost };
}

/** 缩放到"视口内不足一拍"时标签天然稀疏，属正常，跳过。 */
function tooZoomedForLabels(pxPerSec: number, viewportWidth: number): boolean {
    // 工程 BPM 120 ⇒ 一拍 = 0.5s ⇒ 一拍宽 = pxPerSec / 2。
    return pxPerSec * 0.5 > viewportWidth;
}

const VIEWPORT_WIDTHS = [320, 400, 500, 640, 700, 800, 1024, 1200, 1500, 2560];
const LAGS = [0, 64, 128, 200, 255, 256, 300, 400, MAX_LAG_PX - 1];
const TRUE_SCROLLS = [0, 8000, 40000];

describe("标尺刻度的视口覆盖率", () => {
    /**
     * 【本轮的主回归锁】真实视口内的每一个带标签刻度都必须穿过切片边界。
     *
     * 旧实现下窄视口最多单档丢失 3 个标签（视口右端整段空白）；修正缓冲下界后
     * 54000 次检查 0 失败。
     *
     * 【为什么必须显式给超时】这一条是 10 视口 × 200 缩放 × 9 滞后 × 3 滚动 =
     * **54000 次** `renderPipeline` 的穷举（本文件其余几条只有几百到一千次）。
     * 独占运行时约 1.6s，而 vitest 默认超时是 5s —— 全量并行（350 个文件、
     * 19 个 worker）时它会被挤到 5s 以上，然后报成一个看不出所以然的
     * `STACK_TRACE_ERROR`：那是 vitest **内部给超时用的哨兵错误**（`withTimeout`
     * 把它当作 timeoutError 抛出），不是断言失败，所以消息里既没有期望值也没有
     * 实际值，只有一句 "STACK_TRACE_ERROR"。
     *
     * 实测：整仓连跑 4 次，不给超时会有 3 次挂在这一条上（给了 4 次全过）。
     *
     * 穷举本身就是这条用例的价值（它就是标尺切片的回归锁），所以不给它瘦身，
     * 而是给一个与成本相称的预算：约 30 倍余量，真卡死时仍然会失败。
     */
    it("★ 真实视口内的带标签刻度不得被切片丢掉", () => {
        let checks = 0;
        let fails = 0;
        let worst = "";
        for (const viewportWidth of VIEWPORT_WIDTHS) {
            for (let i = 0; i < 200; i += 1) {
                const pxPerSec = 4 * Math.pow(2000 / 4, i / 199);
                for (const lag of LAGS) {
                    for (const trueScroll of TRUE_SCROLLS) {
                        const r = renderPipeline({
                            pxPerSec,
                            reactScroll: Math.max(0, trueScroll - lag),
                            trueScroll,
                            viewportWidth,
                            tempoMap: null,
                        });
                        checks += 1;
                        if (r.lost > 0) {
                            fails += 1;
                            worst = `vw=${viewportWidth} pxPerSec=${pxPerSec.toFixed(1)} lag=${lag} scroll=${trueScroll} lost=${r.lost}`;
                        }
                    }
                }
            }
        }
        expect(checks).toBeGreaterThan(10000);
        expect(fails, `被切片丢掉的刻度：${worst}`).toBe(0);
    }, 60_000);

    it("★ 视口内至少有一个标签（视口宽于一拍时）", () => {
        let checked = 0;
        for (const viewportWidth of VIEWPORT_WIDTHS) {
            for (let i = 0; i < 200; i += 1) {
                const pxPerSec = 4 * Math.pow(2000 / 4, i / 199);
                if (tooZoomedForLabels(pxPerSec, viewportWidth)) continue;
                for (const lag of [0, 255, MAX_LAG_PX - 1]) {
                    const trueScroll = 8000;
                    const r = renderPipeline({
                        pxPerSec,
                        reactScroll: Math.max(0, trueScroll - lag),
                        trueScroll,
                        viewportWidth,
                        tempoMap: null,
                    });
                    checked += 1;
                    expect(
                        r.xs.length,
                        `视口无标签：vw=${viewportWidth} pxPerSec=${pxPerSec.toFixed(1)} lag=${lag}`,
                    ).toBeGreaterThan(0);
                }
            }
        }
        expect(checked).toBeGreaterThan(1000);
    });

    it("★ 视口边缘的空白不超过内部标签间距（不得出现边缘空洞）", () => {
        // 上界必须**自洽**：标签栅格间距随缩放增大（`minLabelSpacingPx` 的 2 的幂倍，
        // 且高缩放下最小 stride 是一拍），固定的像素上界会在高缩放下误报。这里用
        // "边缘空白 ≤ 内部最大间距 + 让位阈值"：边缘空白一旦超过内部间距，就说明
        // 视口边缘被切掉了一段本该存在的栅格。
        const YIELD_PX = 26;
        let worst = { excess: 0, detail: "" };
        let checked = 0;
        for (const viewportWidth of VIEWPORT_WIDTHS) {
            for (let i = 0; i < 200; i += 1) {
                const pxPerSec = 4 * Math.pow(2000 / 4, i / 199);
                for (const lag of [0, 255, MAX_LAG_PX - 1]) {
                    const trueScroll = 8000;
                    const r = renderPipeline({
                        pxPerSec,
                        reactScroll: Math.max(0, trueScroll - lag),
                        trueScroll,
                        viewportWidth,
                        tempoMap: null,
                    });
                    // 需要至少 3 个标签，内部间距才足以代表栅格步长。
                    if (r.xs.length < 3) continue;
                    const gaps = r.xs.slice(1).map((x, k) => x - r.xs[k]);
                    const maxInterior = Math.max(...gaps);
                    const edge = Math.max(r.xs[0], viewportWidth - r.xs[r.xs.length - 1]);
                    const excess = edge - (maxInterior + YIELD_PX);
                    checked += 1;
                    if (excess > worst.excess) {
                        worst = {
                            excess,
                            detail: `vw=${viewportWidth} pxPerSec=${pxPerSec.toFixed(1)} lag=${lag} edge=${edge.toFixed(0)} maxInterior=${maxInterior.toFixed(0)}`,
                        };
                    }
                }
            }
        }
        expect(checked).toBeGreaterThan(1000);
        expect(worst.excess, `视口边缘出现空洞：${worst.detail}`).toBeLessThanOrEqual(0);
    });

    it("★ 密集 Tempo Map 下同样不得丢刻度", () => {
        const tempoMap = denseTempoMap();
        let fails = 0;
        let worst = "";
        for (const viewportWidth of [400, 640, 800, 1500]) {
            for (let i = 0; i < 80; i += 1) {
                const pxPerSec = 4 * Math.pow(2000 / 4, i / 79);
                for (const lag of [0, 255, MAX_LAG_PX - 1]) {
                    for (const trueScroll of [0, 8000]) {
                        const r = renderPipeline({
                            pxPerSec,
                            reactScroll: Math.max(0, trueScroll - lag),
                            trueScroll,
                            viewportWidth,
                            tempoMap,
                        });
                        if (r.lost > 0) {
                            fails += 1;
                            worst = `vw=${viewportWidth} pxPerSec=${pxPerSec.toFixed(1)} lag=${lag} scroll=${trueScroll} lost=${r.lost}`;
                        }
                    }
                }
            }
        }
        expect(fails, `Tempo Map 下被切片丢掉的刻度：${worst}`).toBe(0);
    });
});
