/**
 * 参数编辑器内核宿主 · 构造 / 销毁 / 钳制 / 值域往返单测。
 *
 * 【为什么可以在 node 环境测】阶段 1 的宿主**不碰 WebGL**（绘制仍在面板的
 * Canvas2D 上，经回调注入），只用到很少的 DOM API，因此可以用「记录型桩」替掉
 * 容器与滚动条元素，在无 jsdom 的 node 环境下验证生命周期与数值语义。
 *
 * 【本测试要钉住的核心不变量】
 * 1. `dispose()` 真正释放宿主**取得的**资源：滚动订阅、帧调度、事件监听。
 *    宿主注册的监听（滚动条 thumb 的按下、拖拽的 window 移动 / 抬起）必须逐条
 *    摘除——宿主模式最常见的缺陷就是漏摘 window 监听，卸载后仍持有回调并写 DOM。
 *    断言前先校验「确实注册过」，避免零监听时变成**空断言**。
 * 2. 横向上限 = 内容宽、竖向上限 = 1600（浏览器实测的旧实现基线，见计划 Task 4）；
 * 3. 值域 ↔ 像素往返无损；
 * 4. 重复 `dispose()` 不抛错。
 */
import { describe, expect, it } from "vitest";

import { PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX } from "../scroll/verticalValueScroll";
import { createPianoRollKernelHost } from "./pianoRollKernelHost";

/**
 * 记录型 DOM 桩：统计每个事件类型上 add / remove 的次数、保留处理器以便手动派发，
 * 并带一个可写的 `style`。
 *
 * 特殊说明：保留处理器是必要的——宿主的用户手势回调（拖 thumb）只能通过**真的**
 * 派发 pointerdown / pointermove 来验证，仅统计 add 次数无法覆盖该路径。
 * 同时保留"摘除后不再被调用"的语义：`removeEventListener` 会把处理器移出表。
 */
function makeTarget() {
    const counts = new Map<string, { added: number; removed: number }>();
    const handlers = new Map<string, (event: never) => void>();
    return {
        style: {} as CSSStyleDeclaration,
        counts,
        /** 当前仍注册着的处理器（按事件类型）。测试可直接调用以模拟派发。 */
        handlers,
        addEventListener(type: string, handler: (event: never) => void) {
            const entry = counts.get(type) ?? { added: 0, removed: 0 };
            entry.added += 1;
            counts.set(type, entry);
            handlers.set(type, handler);
        },
        removeEventListener(type: string) {
            const entry = counts.get(type) ?? { added: 0, removed: 0 };
            entry.removed += 1;
            counts.set(type, entry);
            handlers.delete(type);
        },
        /**
         * 全部类型都配平（加过几条就摘掉几条）。
         *
         * 特殊说明：断言前应先校验 `totalAdded() > 0`，否则零监听时会成为空断言。
         */
        balanced(): boolean {
            for (const entry of counts.values()) {
                if (entry.added !== entry.removed) return false;
            }
            return true;
        },
        totalAdded(): number {
            let sum = 0;
            for (const entry of counts.values()) sum += entry.added;
            return sum;
        },
    };
}

/**
 * 造一个测试用宿主（容器 / thumb 全是记录型桩）。
 *
 * @param options 可覆盖项；`offsetPx` 用于覆盖「同步偏移」场景（见坐标契约用例）。
 * @returns 宿主句柄、桩，以及 `flush()`（把排队的帧跑掉，使帧提交可确定性验证）。
 */
function makeHost(options: { offsetPx?: number } = {}) {
    const offsetPx = options.offsetPx ?? 0;
    const container = makeTarget();
    const vThumb = makeTarget();
    const hThumb = makeTarget();
    const rulerContent = { style: {} as CSSStyleDeclaration };
    const paintedAxes: number[] = [];
    const userScrolls: number[] = [];
    let pending: FrameRequestCallback | null = null;
    let handle = 0;
    let scrollLeftCommits = 0;

    // 宿主把拖拽的移动 / 抬起挂在 window 上。node 环境没有 window，因此装一个桩并
    // 用 try/finally 保证无论构造成功与否都还原全局，避免污染同进程的其它测试。
    const windowStub = makeTarget();
    const previousWindow = (globalThis as { window?: unknown }).window;
    (globalThis as { window?: unknown }).window = windowStub;

    let host: ReturnType<typeof createPianoRollKernelHost>;
    try {
        host = createPianoRollKernelHost({
            // node 无 DOM：桩只需满足宿主实际用到的成员（量测 + 事件 + style）。
            container: Object.assign(container, {
                clientWidth: 1864,
                clientHeight: 823,
                scrollLeft: 0,
                scrollTop: 0,
            }) as never,
            hScrollbarThumb: hThumb as never,
            vScrollbarThumb: vThumb as never,
            data: () => ({
                projectSec: 100,
                valueDomain: { min: 0, max: 100, span: 50 },
            }),
            initialPxPerSec: 91.25,
            horizontalOffsetPx: () => offsetPx,
            sync: {
                rulerContent: rulerContent as never,
            },
            onFrame: (axis) => {
                paintedAxes.push(axis.scrollLeftPx);
            },
            onScrollLeftCommit: () => {
                scrollLeftCommits += 1;
            },
            onUserScrollLeft: (drawingScrollLeft) => {
                userScrolls.push(drawingScrollLeft);
            },
            // 注入帧调度：手动 flush，避免依赖 node 里不存在的 rAF。
            requestFrame: (cb) => {
                pending = cb;
                handle += 1;
                return handle;
            },
            cancelFrame: () => {
                pending = null;
            },
        });
    } finally {
        if (previousWindow === undefined) {
            Reflect.deleteProperty(globalThis, "window");
        } else {
            (globalThis as { window?: unknown }).window = previousWindow;
        }
    }

    return {
        host,
        container,
        vThumb,
        hThumb,
        /** window 桩：拖拽的 pointermove / pointerup 从这里派发。 */
        windowHandlers: windowStub.handlers,
        rulerContent,
        paintedAxes,
        userScrolls,
        scrollLeftCommits: () => scrollLeftCommits,
        hasPendingFrame: () => pending !== null,
        /** 跑掉当前排队的帧（上限 8 次，防止自驱动的无限循环）。 */
        flush() {
            for (let i = 0; i < 8 && pending !== null; i += 1) {
                const cb = pending;
                pending = null;
                cb(0);
            }
        },
    };
}

describe("createPianoRollKernelHost", () => {
    it("dispose 后不再调度帧、不再写 DOM（资源真被释放）", () => {
        const t = makeHost();
        t.host.setScrollLeft(300);
        t.flush();
        const commitsBeforeDispose = t.scrollLeftCommits();
        expect(commitsBeforeDispose).toBeGreaterThan(0);

        t.host.dispose();
        // dispose 本身不得留下待执行的帧。
        expect(t.hasPendingFrame()).toBe(false);

        // dispose 后的写入不得再产生帧提交 / DOM 写入。
        t.host.setScrollLeft(900);
        t.flush();
        expect(t.scrollLeftCommits()).toBe(commitsBeforeDispose);
        expect(t.hasPendingFrame()).toBe(false);
    });

    it("dispose 后监听登记配平，且确实注册过滚动条监听（非空断言）", () => {
        const t = makeHost();
        // 先证明断言有真实约束力：宿主确实在 thumb 上注册过 pointerdown。
        expect(t.hThumb.totalAdded()).toBeGreaterThan(0);
        expect(t.vThumb.totalAdded()).toBeGreaterThan(0);
        t.host.dispose();
        expect(t.container.balanced()).toBe(true);
        expect(t.vThumb.balanced()).toBe(true);
        expect(t.hThumb.balanced()).toBe(true);
    });

    it("重复 dispose 是安全的空操作", () => {
        const t = makeHost();
        t.host.dispose();
        expect(() => t.host.dispose()).not.toThrow();
        expect(t.container.balanced()).toBe(true);
    });

    it("横向上限 = 内容宽（旧实现实测 9125）", () => {
        const t = makeHost();
        t.host.setScrollLeft(999999);
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(100 * 91.25, 6);
        t.host.dispose();
    });

    it("竖向上限 = 1600（旧实现实测值），与视口高无关", () => {
        const t = makeHost();
        t.host.setScrollTop(999999);
        expect(t.host.getViewport().scrollTop).toBeCloseTo(PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX, 6);
        t.host.dispose();
    });

    it("值域中心往返无损（内核像素 ↔ 值域）", () => {
        const t = makeHost();
        t.host.setValueCenter(62.5);
        expect(t.host.getValueCenter()).toBeCloseTo(62.5, 6);
        t.host.dispose();
    });

    it("帧提交会写 DOM、回调面板并量化提交", () => {
        const t = makeHost();
        t.host.setScrollLeft(300);
        t.flush();
        expect(t.scrollLeftCommits()).toBeGreaterThan(0);
        // 面板回调收到的是同一份投影（绘制坐标）。
        expect(t.paintedAxes.at(-1)).toBeCloseTo(300, 6);
        // 标尺内容层按绘制坐标反向平移。
        expect(t.rulerContent.style.transform).toBe("translateX(-300px)");
        t.host.dispose();
    });
});

/**
 * 坐标契约：内核持有**原生坐标**（域 `[0, 内容宽 + 偏移]`），对外一律暴露
 * **绘制坐标**（域 `[−偏移, 内容宽]`）。
 *
 * 【为什么单独成组】这是本阶段最容易出错、也最难在类型上发现的一处：两套坐标系
 * 只差一个偏移量，写错时功能"看起来正常"，但同步模式下网格会与时间轴错位、
 * 或同步留白（负的绘制坐标）无法表示。以下期望值取自浏览器实测的旧实现行为。
 */
describe("createPianoRollKernelHost · 坐标契约（偏移 200）", () => {
    const OFFSET = 200;
    /** 内容宽 = 100s × 91.25px/s。 */
    const CONTENT_W = 100 * 91.25;
    const VIEWPORT_W = 1864;

    it("静置：原生 0 → 绘制 -偏移（= 旧实现同步留白实测值）", () => {
        const t = makeHost({ offsetPx: OFFSET });
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(-OFFSET, 6);
        t.host.dispose();
    });

    it("绘制域 = [-偏移, 内容宽]（两端都可表示）", () => {
        const t = makeHost({ offsetPx: OFFSET });
        t.host.setScrollLeft(999999);
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(CONTENT_W, 6);
        t.host.setScrollLeft(-999999);
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(-OFFSET, 6);
        t.host.dispose();
    });

    it("投影与视口同一口径（都是绘制坐标）", () => {
        const t = makeHost({ offsetPx: OFFSET });
        t.host.setScrollLeft(1200);
        t.flush();
        expect(t.host.getAxis().scrollLeftPx).toBeCloseTo(1200, 6);
        expect(t.paintedAxes.at(-1)).toBeCloseTo(1200, 6);
        t.host.dispose();
    });

    it("滚动条内容尺寸 = 原生 scrollWidth（= 内容宽 + 偏移 + 视口宽）", () => {
        const t = makeHost({ offsetPx: OFFSET });
        const nativeScrollWidth = CONTENT_W + OFFSET + VIEWPORT_W;
        expect(t.host.getScrollbarGeometries().horizontal.thumbLengthPx).toBeCloseTo(
            (VIEWPORT_W * VIEWPORT_W) / nativeScrollWidth,
            6,
        );
        t.host.dispose();
    });

    it("偏移为 0 时绘制域退化为 [0, 内容宽]（未开启同步的既有调用方不受影响）", () => {
        const t = makeHost();
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(0, 6);
        t.host.setScrollLeft(999999);
        expect(t.host.getViewport().scrollLeft).toBeCloseTo(CONTENT_W, 6);
        t.host.dispose();
    });
});

/**
 * 用户手势上报：`onUserScrollLeft` 只在宿主**亲自解析**的手势后触发。
 *
 * 【为什么单列一组】它与 `onScrollLeftCommit` 长得很像但语义完全不同：前者代表
 * "用户想滚到这儿"（面板据此同步时间轴），后者只是"位置变了"（含宿主的镜像回写）。
 * 混用会让镜像回写被当成用户滚动，或让滚动条拖拽漏掉同步——两者都曾实际发生。
 */
describe("createPianoRollKernelHost · onUserScrollLeft", () => {
    it("命令式写入（setScrollLeft）不触发用户手势回调", () => {
        const t = makeHost();
        t.host.setScrollLeft(500);
        t.flush();
        expect(t.userScrolls).toEqual([]);
        t.host.dispose();
    });

    it("拖横向 thumb 触发回调，且报的是**绘制坐标**（已减去偏移）", () => {
        const t = makeHost({ offsetPx: 200 });
        const down = t.hThumb.handlers.get("pointerdown");
        expect(down).toBeDefined();
        // 模拟按下（记录起点）后移动指针。
        down?.({
            preventDefault() {},
            stopPropagation() {},
            clientX: 100,
            pointerId: 1,
            currentTarget: { setPointerCapture() {} },
        } as never);
        const move = t.windowHandlers.get("pointermove");
        expect(move).toBeDefined();
        move?.({ clientX: 180 } as never);
        expect(t.userScrolls.length).toBeGreaterThan(0);
        // 偏移 200：回调给出的绘制坐标应比内核内部的原生坐标小 200。
        const reported = t.userScrolls.at(-1) as number;
        expect(reported).toBeCloseTo(t.host.getViewport().scrollLeft, 6);
        t.host.dispose();
    });
});
