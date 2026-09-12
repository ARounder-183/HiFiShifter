/**
 * 参数编辑器内核宿主 · 构造 / 销毁 / 钳制 / 值域往返单测。
 *
 * 【为什么可以在 node 环境测】阶段 1 的宿主**不碰 WebGL**（绘制仍在面板的
 * Canvas2D 上，经回调注入），只用到很少的 DOM API，因此可以用「记录型桩」替掉
 * 容器与滚动条元素，在无 jsdom 的 node 环境下验证生命周期与数值语义。
 *
 * 【本测试要钉住的核心不变量】
 * 1. `dispose()` 真正释放宿主**取得的**资源：滚动订阅、帧调度、DOM 写入能力。
 *    注意 Task 4 的宿主**不注册任何输入监听**（滚轮 / 键盘 / 滚动条拖拽属 Task 7），
 *    因此这里不去断言「摘掉了监听」——那在零监听时是**空断言**。改为断言
 *    「dispose 后不再有任何帧被调度、不再写 DOM」，这在零监听下依然有真实约束力。
 *    监听登记的 add/remove 配平仍保留校验，Task 7 接入后它会自动开始生效。
 * 2. 横向上限 = 内容宽、竖向上限 = 1600（浏览器实测的旧实现基线，见计划 Task 4）；
 * 3. 值域 ↔ 像素往返无损；
 * 4. 重复 `dispose()` 不抛错。
 */
import { describe, expect, it } from "vitest";

import { PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX } from "../scroll/verticalValueScroll";
import { createPianoRollKernelHost } from "./pianoRollKernelHost";

/** 记录型 DOM 桩：统计每个事件类型上 add / remove 的次数，并带一个可写的 `style`。 */
function makeTarget() {
    const counts = new Map<string, { added: number; removed: number }>();
    return {
        style: {} as CSSStyleDeclaration,
        counts,
        addEventListener(type: string) {
            const entry = counts.get(type) ?? { added: 0, removed: 0 };
            entry.added += 1;
            counts.set(type, entry);
        },
        removeEventListener(type: string) {
            const entry = counts.get(type) ?? { added: 0, removed: 0 };
            entry.removed += 1;
            counts.set(type, entry);
        },
        /**
         * 全部类型都配平（加过几条就摘掉几条）。
         *
         * 特殊说明：Task 4 的宿主不注册监听，故当前恒为 true；Task 7 接入输入后
         * 本断言开始具备真实约束力（防漏摘 window 监听）。
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
 * @returns 宿主句柄、桩，以及 `flush()`（把排队的帧跑掉，使帧提交可确定性验证）。
 */
function makeHost() {
    const container = makeTarget();
    const vThumb = makeTarget();
    const hThumb = makeTarget();
    const rulerContent = { style: {} as CSSStyleDeclaration };
    const paintedAxes: number[] = [];
    let pending: FrameRequestCallback | null = null;
    let handle = 0;
    let scrollLeftCommits = 0;

    const host = createPianoRollKernelHost({
        // node 无 DOM：桩只需满足宿主实际用到的成员（量测 + 事件 + style）。
        container: Object.assign(container, {
            clientWidth: 1864,
            clientHeight: 823,
        }) as never,
        hScrollbarThumb: hThumb as never,
        vScrollbarThumb: vThumb as never,
        data: () => ({
            projectSec: 100,
            valueDomain: { min: 0, max: 100, span: 50 },
        }),
        initialPxPerSec: 91.25,
        sync: {
            rulerContent: rulerContent as never,
        },
        onFrame: (axis) => {
            paintedAxes.push(axis.scrollLeftPx);
        },
        onScrollLeftCommit: () => {
            scrollLeftCommits += 1;
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

    return {
        host,
        container,
        vThumb,
        hThumb,
        rulerContent,
        paintedAxes,
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

    it("dispose 后监听登记仍配平（Task 7 接入输入后本断言开始生效）", () => {
        const t = makeHost();
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
        expect(t.host.getViewport().scrollTop).toBeCloseTo(
            PIANO_ROLL_VERTICAL_SCROLL_RANGE_PX,
            6,
        );
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
