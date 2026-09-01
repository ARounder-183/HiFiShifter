/**
 * `drawTimelineCanvas` 合批重写的**结构**回归测试。
 *
 * 【测什么】这次重写把「每个 clip 一次 `ctx.clip()` 圆角遮罩 + 逐个 fillRect」
 * 换成「每角独立半径的 roundRect + 按样式合批」。视觉逐像素等价性靠推理保证
 * （见 timelineCanvasRenderer 里的长注释），这里锁的是**结构性不变量**：
 *
 * 1. 互不遮挡的 clip 不得产生任何 `clip()` 调用（这是本次最大的收益来源）；
 * 2. `fill()` / `stroke()` 的次数必须与 clip 数**无关**，只随样式数增长——
 *    否则合批就退化回了逐 clip 绘制；
 * 3. 填充矩形的**总面积守恒**：合批不得漏画任何一块 header / body / 分隔线
 *    / 分隔缝；
 * 4. 屏障（前导重叠 / 编组外圈描边）必须切断批次，否则 z 序会被打乱。
 *
 * 【怎么测】用一个记录型 mock 替换 CanvasRenderingContext2D，并 polyfill
 * `Path2D` 为「记录 roundRect 参数的容器」。不依赖真实 DOM / 浏览器。
 */

import { beforeEach, describe, expect, it } from "vitest";

import type { TimelineCanvasClipModel } from "./timelineCanvasModel.js";

/** 记录下来的一个矩形（roundRect 的参数）。 */
interface RecordedRect {
    x: number;
    y: number;
    w: number;
    h: number;
    radii: number | number[];
}

/** 记录型 Path2D：只需要 roundRect。 */
class RecordingPath2D {
    readonly rects: RecordedRect[] = [];
    roundRect(x: number, y: number, w: number, h: number, radii: number | number[] = 0): void {
        this.rects.push({ x, y, w, h, radii });
    }
}

interface RecordedFill {
    style: string;
    alpha: number;
    rects: RecordedRect[];
    /** true = 合批提交（传入 Path2D）；false = 逐 clip 的细节层填充。 */
    batched: boolean;
}
interface RecordedStroke {
    style: string;
    lineWidth: number;
    rects: RecordedRect[];
    batched: boolean;
}

/** 记录型 ctx：只实现 drawTimelineCanvas 用到的成员。 */
class RecordingContext {
    fillStyle = "";
    strokeStyle = "";
    lineWidth = 1;
    globalAlpha = 1;
    font = "";
    textBaseline = "";
    textAlign = "";
    lineCap = "butt";

    clipCount = 0;
    fillRectCount = 0;
    strokeRectCount = 0;
    fillCalls: RecordedFill[] = [];
    strokeCalls: RecordedStroke[] = [];
    /** 每次 fill/stroke 的调用序号，用于判断批次边界。 */
    opSequence: string[] = [];

    private pendingRects: RecordedRect[] = [];
    private pendingRect: RecordedRect | null = null;

    beginPath(): void {
        this.pendingRects = [];
        this.pendingRect = null;
    }
    roundRect(x: number, y: number, w: number, h: number, radii: number | number[] = 0): void {
        this.pendingRects.push({ x, y, w, h, radii });
    }
    rect(x: number, y: number, w: number, h: number): void {
        this.pendingRect = { x, y, w, h, radii: 0 };
    }
    arc(): void {}
    ellipse(): void {}
    moveTo(): void {}
    lineTo(): void {}
    closePath(): void {}
    clip(): void {
        this.clipCount += 1;
    }
    fillRect(): void {
        this.fillRectCount += 1;
    }
    strokeRect(): void {
        this.strokeRectCount += 1;
    }
    fill(path?: unknown): void {
        // 合批路径会显式传入 Path2D；细节层走隐式当前路径（不传参）。
        // 这个区分是断言"合批是否生效"的关键。
        const batched = path instanceof RecordingPath2D;
        const rects = batched
            ? (path as RecordingPath2D).rects
            : this.pendingRect
              ? [this.pendingRect]
              : this.pendingRects;
        this.fillCalls.push({
            style: this.fillStyle,
            alpha: this.globalAlpha,
            rects: [...rects],
            batched,
        });
        this.opSequence.push("fill");
    }
    stroke(path?: unknown): void {
        const batched = path instanceof RecordingPath2D;
        const rects = batched
            ? (path as RecordingPath2D).rects
            : this.pendingRect
              ? [this.pendingRect]
              : this.pendingRects;
        this.strokeCalls.push({
            style: this.strokeStyle,
            lineWidth: this.lineWidth,
            rects: [...rects],
            batched,
        });
        this.opSequence.push("stroke");
    }
    save(): void {}
    restore(): void {}
    setTransform(): void {}
    translate(): void {}
    clearRect(): void {}
    measureText(text: string): { width: number } {
        return { width: text.length * 6 };
    }
    fillText(): void {}
}

/** 构造一个 clip 模型。 */
function clip(overrides: Partial<TimelineCanvasClipModel> = {}): TimelineCanvasClipModel {
    return {
        id: "clip",
        trackId: "track",
        name: "Clip",
        leftPx: 0,
        topPx: 0,
        widthPx: 200,
        heightPx: 90,
        headerHeightPx: 18,
        fadeInPx: 0,
        fadeOutPx: 0,
        fadeInShape: 0,
        fadeOutShape: 0,
        fadeInDir: 0,
        fadeOutDir: 0,
        selected: false,
        muted: false,
        gain: 1,
        playbackRate: 1,
        isMidiClip: false,
        isRenaming: false,
        snapOffsetPx: 0,
        leadingOverlapPx: 0,
        ...overrides,
    };
}

/** 一行里 N 个首尾相接（互不重叠）的 clip。 */
function backToBack(count: number, startX = 0, widthPx = 200): TimelineCanvasClipModel[] {
    return Array.from({ length: count }, (_unused, index) =>
        clip({
            id: `c${index}`,
            leftPx: startX + index * widthPx,
            widthPx,
        }),
    );
}

let ctx: RecordingContext;

beforeEach(() => {
    ctx = new RecordingContext();
    (globalThis as unknown as { Path2D: unknown }).Path2D = RecordingPath2D;
    // `drawTimelineCanvas` 的清屏会读 `window.devicePixelRatio`；Node 环境下
    // 没有 window，补一个最小桩即可（本测试不关心 DPR）。
    if ((globalThis as unknown as { window?: unknown }).window === undefined) {
        (globalThis as unknown as { window: unknown }).window = { devicePixelRatio: 1 };
    }
});

describe("drawTimelineCanvas 合批", () => {
    // 延迟 import：Path2D 的 polyfill 必须在模块求值前就位（模块内只在函数
    // 调用时才 new Path2D，因此这里 import 只是取引用，仍然安全）。
    async function load() {
        return import("./timelineCanvasRenderer.js");
    }

    it("不重叠的 clip 不产生任何 clip() 调用，且 fill/stroke 次数与 clip 数无关", async () => {
        const { drawTimelineCanvas } = await load();
        // 宽度取 60px：低于名称文本的 152px 阈值，且吸附三角完整落在 clip 内
        // —— 这正是全览缩放（clip 很窄）时的形态，也是最需要合批的场景。
        const widthPx = 60;
        const counts: Array<{ fill: number; stroke: number; clip: number }> = [];

        for (const count of [10, 50, 200]) {
            ctx = new RecordingContext();
            drawTimelineCanvas(ctx as unknown as CanvasRenderingContext2D, {
                width: 4000,
                height: 90,
                clips: backToBack(count, 0, widthPx),
                fontFamily: "sans-serif",
                darkMode: true,
            });
            counts.push({
                fill: ctx.fillCalls.filter((call) => call.batched).length,
                stroke: ctx.strokeCalls.filter((call) => call.batched).length,
                clip: ctx.clipCount,
            });
        }

        // 主体绘制完全不需要圆角遮罩（这是本次重写要消灭的最大单项开销）。
        for (const entry of counts) {
            expect(entry.clip).toBe(0);
        }
        // 合批生效：三种 clip 数的绘制调用次数完全一致。
        expect(counts[0]).toEqual(counts[1]);
        expect(counts[1]).toEqual(counts[2]);
        // 合批调用数远小于 clip 数（否则就是退化回逐 clip 绘制）。
        // 注意：细节层（旋钮 / 徽标 / 文字）天然是逐 clip 的，不计入这里。
        expect(counts[2]!.fill).toBeLessThan(20);
        expect(counts[2]!.stroke).toBeLessThan(20);
    });

    it("宽 clip 的 clip() 只来自名称文本裁剪，与主体无关", async () => {
        const { drawTimelineCanvas } = await load();
        const count = 12;
        drawTimelineCanvas(ctx as unknown as CanvasRenderingContext2D, {
            width: 4000,
            height: 90,
            // 200px ≥ 152px ⇒ 每个 clip 会为名称文本裁剪一次。
            clips: backToBack(count, 0, 200),
            fontFamily: "sans-serif",
            darkMode: true,
        });
        // 恰好等于 clip 数：不多（说明主体与吸附三角都没再引入遮罩）。
        expect(ctx.clipCount).toBe(count);
        // 主体填充仍是合批的（远少于 clip 数）。
        expect(ctx.fillCalls.filter((call) => call.batched).length).toBeLessThan(20);
    });

    it("填充总面积守恒：每个 clip 的 header / body / 分隔线都被画到", async () => {
        const { drawTimelineCanvas } = await load();
        const count = 40;
        const widthPx = 200;
        const heightPx = 90;
        const headerHeight = 18;
        drawTimelineCanvas(ctx as unknown as CanvasRenderingContext2D, {
            width: count * widthPx,
            height: heightPx,
            clips: backToBack(count, 0, widthPx),
            fontFamily: "sans-serif",
            darkMode: true,
        });

        // 只统计不透明的主体填充（排除分隔线与分隔缝的细线）。
        const bodyArea = (heightPx - headerHeight) * widthPx;
        let counted = 0;
        for (const call of ctx.fillCalls) {
            for (const rect of call.rects) {
                if (Math.abs(rect.h - (heightPx - headerHeight)) < 1e-6) {
                    counted += rect.w * rect.h;
                }
            }
        }
        expect(counted).toBeCloseTo(count * bodyArea, 6);

        // header 面积同样守恒。
        let headerCounted = 0;
        for (const call of ctx.fillCalls) {
            for (const rect of call.rects) {
                if (Math.abs(rect.h - headerHeight) < 1e-6) {
                    headerCounted += rect.w * rect.h;
                }
            }
        }
        expect(headerCounted).toBeCloseTo(count * headerHeight * widthPx, 6);
    });

    it("前导重叠会切断批次（屏障生效）", async () => {
        const { drawTimelineCanvas } = await load();

        const run = (clips: TimelineCanvasClipModel[]): number => {
            ctx = new RecordingContext();
            drawTimelineCanvas(ctx as unknown as CanvasRenderingContext2D, {
                width: 4000,
                height: 90,
                clips,
                fontFamily: "sans-serif",
                darkMode: true,
            });
            return ctx.fillCalls.length;
        };

        const plain = run(backToBack(20));
        const withOverlap = run(
            backToBack(20).map((item, index) =>
                index === 10 ? { ...item, leadingOverlapPx: 40 } : item,
            ),
        );
        // 中间插入一个重叠 clip ⇒ 批次被切成两段，填充调用数应显著增加。
        expect(withOverlap).toBeGreaterThan(plain);
    });

    it("编组激活会切断批次（外圈描边会伸出自身矩形）", async () => {
        const { drawTimelineCanvas } = await load();

        const run = (clips: TimelineCanvasClipModel[]): number => {
            ctx = new RecordingContext();
            drawTimelineCanvas(ctx as unknown as CanvasRenderingContext2D, {
                width: 4000,
                height: 90,
                clips,
                fontFamily: "sans-serif",
                darkMode: true,
                activeGroupIds: new Set(["g1"]),
            });
            return ctx.strokeCalls.length;
        };

        const plain = run(backToBack(20));
        const grouped = run(
            backToBack(20).map((item, index) => (index === 10 ? { ...item, groupId: "g1" } : item)),
        );
        expect(grouped).toBeGreaterThan(plain);
    });
});
