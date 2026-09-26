/**
 * 视口总线重放判定的回归。
 *
 * 【为什么必须有这份测试】重放是**合成**输入，它出错时不会报错、也不会稳定复现 ——
 * 只会表现为"用绘制工具画 `音量` / `动态` 时按下那一点在、拖拽轨迹完全不画"
 * （见 viewportReplay.ts 的模块头说明）。两个坑都只能靠单测钉住：
 *
 * - 载荷的按键状态若沿用"最近一次事件"的，按下后停顿一拍就会注入 `buttons === 0`，
 *   手势把它读成"用户松手了"并当场收尾；
 * - 若不区分"视口变了"与"只是强制重绘"，绘制音量 / 动态时每帧一次的波形重绘都会
 *   注入一次重放。
 */
import { describe, expect, it } from "vitest";

import {
    isSameViewportProjection,
    resolveViewportReplay,
    type PointerMoveSnapshot,
    type ViewportProjection,
} from "./viewportReplay";

/**
 * 各手势用来判定"键还按着吗"的掩码（见 `usePianoRollInteractions` 的移动处理器）。
 * 载荷必须让它们全部成立 —— 否则该手势会把重放当成"已松手"。
 */
const LEFT_BUTTON_MASK = 1;
const RIGHT_BUTTON_MASK = 2;

function projection(scrollLeftPx: number, pxPerSec = 150, viewportWidthPx = 1200) {
    return { scrollLeftPx, pxPerSec, viewportWidthPx } satisfies ViewportProjection;
}

/** 一次"悬停"移动：按下之前最后收到的那种事件（`buttons === 0`）。 */
function hoverMove(): PointerMoveSnapshot {
    return {
        clientX: 400,
        clientY: 220,
        pointerId: 1,
        pointerType: "mouse",
        isPrimary: true,
        shiftKey: false,
        ctrlKey: false,
        altKey: false,
        metaKey: false,
        buttons: 0,
    };
}

/** 一次真实的拖拽移动（`buttons === 1`）。 */
function dragMove(): PointerMoveSnapshot {
    return { ...hoverMove(), clientX: 512, buttons: LEFT_BUTTON_MASK };
}

/** 一次"投影变了 + 左键拖拽中"的正常重放输入。 */
function replayableInput(overrides: Partial<Parameters<typeof resolveViewportReplay>[0]> = {}) {
    return {
        projection: projection(320),
        previousProjection: projection(300),
        dragging: true,
        pressedButtons: LEFT_BUTTON_MASK,
        lastMove: dragMove(),
        ...overrides,
    };
}

describe("isSameViewportProjection", () => {
    it("三项全等才算相同", () => {
        const a = projection(300);
        expect(isSameViewportProjection(a, projection(300))).toBe(true);
        expect(isSameViewportProjection(a, projection(301))).toBe(false);
        expect(isSameViewportProjection(a, projection(300, 151))).toBe(false);
        expect(isSameViewportProjection(a, projection(300, 150, 1201))).toBe(false);
    });

    /** 没有基准时视为"变了"：宁可多放一次，也不漏掉一次真实视口变化。 */
    it("没有基准（null）视为不相同", () => {
        expect(isSameViewportProjection(null, projection(300))).toBe(false);
    });
});

describe("resolveViewportReplay：★ 载荷的按键状态取「此刻」", () => {
    /**
     * 这是本 bug 的核心回归：按下之后、第一次真实移动之前，最近一次事件是**悬停**
     * （`buttons === 0`）。载荷必须取"此刻按着左键"，否则手势当场收尾。
     */
    it("★ 最近一次事件是悬停（buttons = 0）时，载荷仍声明左键按着", () => {
        const decision = resolveViewportReplay(
            replayableInput({ lastMove: hoverMove(), pressedButtons: LEFT_BUTTON_MASK }),
        );
        expect(decision.replay).toBe(true);
        if (!decision.replay) return;
        expect(decision.payload.buttons).toBe(LEFT_BUTTON_MASK);
        // 绘制 / 直线 / 颤音 / 平移等手势的判据必须成立。
        expect((decision.payload.buttons & LEFT_BUTTON_MASK) === LEFT_BUTTON_MASK).toBe(true);
    });

    it("载荷携带位置与设备信息，且 button 为 -1（移动事件语义）", () => {
        const decision = resolveViewportReplay(replayableInput({ lastMove: dragMove() }));
        expect(decision.replay).toBe(true);
        if (!decision.replay) return;
        expect(decision.payload.clientX).toBe(512);
        expect(decision.payload.clientY).toBe(220);
        expect(decision.payload.pointerId).toBe(1);
        expect(decision.payload.pointerType).toBe("mouse");
        expect(decision.payload.isPrimary).toBe(true);
        expect(decision.payload.button).toBe(-1);
        expect(decision.payload.bubbles).toBe(true);
        expect(decision.payload.cancelable).toBe(true);
    });

    it("多键同按时掩码按位成立（左键 + 右键）", () => {
        const decision = resolveViewportReplay(
            replayableInput({ pressedButtons: LEFT_BUTTON_MASK | RIGHT_BUTTON_MASK }),
        );
        expect(decision.replay).toBe(true);
        if (!decision.replay) return;
        const buttons = decision.payload.buttons;
        expect((buttons & LEFT_BUTTON_MASK) === LEFT_BUTTON_MASK).toBe(true);
        expect((buttons & RIGHT_BUTTON_MASK) === RIGHT_BUTTON_MASK).toBe(true);
    });

    it("右键拖拽路径的掩码同样成立", () => {
        const decision = resolveViewportReplay(
            replayableInput({ pressedButtons: RIGHT_BUTTON_MASK, lastMove: hoverMove() }),
        );
        expect(decision.replay).toBe(true);
        if (!decision.replay) return;
        expect((decision.payload.buttons & RIGHT_BUTTON_MASK) === RIGHT_BUTTON_MASK).toBe(true);
    });
});

describe("resolveViewportReplay：强制重绘不算视口变化", () => {
    /**
     * 绘制音量 / 动态时，每次 `pointermove` 都会请求一次波形重绘，而
     * `pianoRollViewportBus.invalidate()` 会 paint 全部订阅者（包括本重放）。
     * 投影没变就不该重放 —— 这是自激环（invalidate → 重放 → 移动 → invalidate）
     * 的断点，也是本 bug 触发条件的消除点。
     */
    it("★ 投影未变 → 不重放（哪怕正拖着且最近一次是悬停事件）", () => {
        const same = projection(300);
        const decision = resolveViewportReplay(
            replayableInput({
                projection: same,
                previousProjection: same,
                lastMove: hoverMove(),
            }),
        );
        expect(decision.replay).toBe(false);
        if (decision.replay) return;
        expect(decision.reason).toBe("projection-unchanged");
    });

    it("投影变了才重放（真实滚动 / 缩放）", () => {
        const decision = resolveViewportReplay(
            replayableInput({ projection: projection(301), previousProjection: projection(300) }),
        );
        expect(decision.replay).toBe(true);
    });

    it("首次 paint（无基准）视为变化 → 重放", () => {
        const decision = resolveViewportReplay(replayableInput({ previousProjection: null }));
        expect(decision.replay).toBe(true);
    });
});

describe("resolveViewportReplay：不重放的其余情形", () => {
    it("未在拖拽 → 不重放", () => {
        const decision = resolveViewportReplay(
            replayableInput({ dragging: false, pressedButtons: 0 }),
        );
        expect(decision.replay).toBe(false);
        if (decision.replay) return;
        expect(decision.reason).toBe("not-dragging");
    });

    it("拖拽中但没有可重放的位置 → 不重放", () => {
        const decision = resolveViewportReplay(replayableInput({ lastMove: null }));
        expect(decision.replay).toBe(false);
        if (decision.replay) return;
        expect(decision.reason).toBe("no-last-move");
    });

    /**
     * 自相矛盾的输入（拖拽中却没有任何按键按下）**优先**被拦下：即使投影没变也要
     * 报出这个原因，否则这条诊断会被"投影未变"掩盖，永远看不见。
     */
    it("★ 拖拽中按键掩码为 0 → 拦下并报出该原因（优先于投影未变）", () => {
        const same = projection(300);
        const decision = resolveViewportReplay(
            replayableInput({
                projection: same,
                previousProjection: same,
                pressedButtons: 0,
            }),
        );
        expect(decision.replay).toBe(false);
        if (decision.replay) return;
        expect(decision.reason).toBe("pressed-buttons-lost");
    });

    /** 空闲时的强制重绘不得报出"按键丢失"（否则诊断会变成噪声）。 */
    it("未拖拽且掩码为 0 时不报按键丢失", () => {
        const same = projection(300);
        const decision = resolveViewportReplay(
            replayableInput({
                projection: same,
                previousProjection: same,
                dragging: false,
                pressedButtons: 0,
                lastMove: null,
            }),
        );
        expect(decision.replay).toBe(false);
        if (decision.replay) return;
        expect(decision.reason).not.toBe("pressed-buttons-lost");
    });
});
