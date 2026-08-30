/**
 * gestureFocusGuard 的注册/触发/幂等回归。
 *
 * node 测试环境没有全局 window/document：测试先用最小桩补上（监听函数
 * 记录到表里），再动态 import 模块（模块在首次注册时挂监听），之后直接
 * 驱动桩里的 blur/visibilitychange 回调验证：
 * 1. blur 触发一次收尾；
 * 2. finish 自注销后不再触发（幂等）；
 * 3. 多手势全部触发且单点异常被隔离；
 * 4. visibilitychange → hidden 收尾、回 visible 不收尾。
 */
import { test } from "vitest";

type ListenerMap = Record<string, Array<() => void>>;

function makeTarget() {
    const listeners: ListenerMap = {};
    const target = {
        visibilityState: "visible",
        addEventListener: (type: string, fn: () => void) => {
            (listeners[type] ??= []).push(fn);
        },
        removeEventListener: (type: string, fn: () => void) => {
            listeners[type] = (listeners[type] ?? []).filter((f) => f !== fn);
        },
    };
    return { target, listeners };
}

test("components/layout/timeline/gestureFocusGuard.ts scripted checks", async () => {
    // node 环境补齐全局桩，再导入模块。
    const win = makeTarget();
    const doc = makeTarget();
    (globalThis as { window?: unknown }).window = win.target;
    (globalThis as { document?: unknown }).document = doc.target;
    const { registerDragAbort } = await import("./gestureFocusGuard");

    const fireBlur = () => {
        for (const fn of [...(win.listeners["blur"] ?? [])]) fn();
    };
    const fireVisibility = (state: "visible" | "hidden") => {
        doc.target.visibilityState = state;
        for (const fn of [...(doc.listeners["visibilitychange"] ?? [])]) fn();
    };

    // ── 1. blur 触发一次收尾；finish 自注销后不再触发 ─────────────
    let calls = 0;
    const unregister = registerDragAbort(() => {
        calls += 1;
        unregister(); // 模拟 finish 第一步自注销
    });
    fireBlur();
    if (calls !== 1) {
        throw new Error(`blur should invoke the abort exactly once, got ${calls}`);
    }
    fireBlur();
    if (calls !== 1) {
        throw new Error(`abort must be dead after unregister, got ${calls}`);
    }

    // ── 2. 多手势：全部触发且异常隔离 ────────────────────────────
    const order: string[] = [];
    const unregA = registerDragAbort(() => {
        order.push("a");
        unregA();
    });
    const unregB = registerDragAbort(() => {
        order.push("b-throw");
        unregB();
        throw new Error("boom");
    });
    const unregC = registerDragAbort(() => {
        order.push("c");
        unregC();
    });
    fireBlur();
    if (order.join(",") !== "a,b-throw,c") {
        throw new Error(`all aborts should run despite one throwing: ${order.join(",")}`);
    }

    // ── 3. visibilitychange hidden 收尾、回 visible 不收尾 ────────
    // 断言走函数参数（unknown）：绕过 TS 对闭包自增变量的字面量收窄。
    const assertEq = (actual: unknown, expected: number, label: string): void => {
        if (actual !== expected) {
            throw new Error(`${label}: expected ${expected}, got ${String(actual)}`);
        }
    };
    let hiddenCalls = 0;
    const unreg2 = registerDragAbort(() => {
        hiddenCalls += 1; // 本段不模拟自注销：验证"未注销可重复触发、注销后静默"
    });
    fireVisibility("hidden");
    assertEq(hiddenCalls, 1, "hidden should invoke the abort");
    fireVisibility("visible");
    assertEq(hiddenCalls, 1, "visible must not invoke the abort");
    fireVisibility("hidden");
    assertEq(hiddenCalls, 2, "re-hidden should invoke again while still registered");
    unreg2();
    fireVisibility("hidden");
    assertEq(hiddenCalls, 2, "abort must be dead after unregister");
    fireVisibility("visible");
}, 30_000);