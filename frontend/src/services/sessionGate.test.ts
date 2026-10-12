/**
 * 会话闸门的契约。
 *
 * 【要钉死什么】
 * 1. 终态置位后**幂等** —— 第一条终态携带的协议码决定界面给哪句指引，后到的连带
 *    失败不得覆盖它；
 * 2. 快照**引用稳定** —— `useSyncExternalStore` 靠引用相等判断"变了没有"，每次新建
 *    对象会让 React 认为一直在变并无限重渲染；
 * 3. 订阅/退订成对，退订后不再收到通知（订阅者出错也不得影响闸门状态）。
 */
import { afterEach, expect, test, vi } from "vitest";

import {
    ERROR_CODE,
    getSessionGateSnapshot,
    isBackendTerminal,
    markBackendTerminal,
    resetSessionGateForTests,
    sessionTerminal,
    subscribeSessionGate,
} from "./sessionGate";

afterEach(() => {
    resetSessionGateForTests();
});

test("the gate starts open", () => {
    expect(isBackendTerminal()).toBe(false);
    expect(sessionTerminal()).toBeNull();
    expect(getSessionGateSnapshot()).toBeNull();
});

test("marking the gate is idempotent and keeps the first code", () => {
    markBackendTerminal(
        "plug-in instance is not bound to an ARA document",
        ERROR_CODE.ARA_NOT_BOUND,
    );
    expect(isBackendTerminal()).toBe(true);
    expect(sessionTerminal()?.code).toBe(ERROR_CODE.ARA_NOT_BOUND);

    // 后到的连带失败不得覆盖第一条：它决定界面给哪句指引。
    markBackendTerminal("FX processor closed", ERROR_CODE.FX_PROCESSOR_CLOSED);
    expect(sessionTerminal()?.code).toBe(ERROR_CODE.ARA_NOT_BOUND);
    expect(sessionTerminal()?.message).toContain("not bound");
});

test("a missing code is stored as null rather than guessed", () => {
    // 旧后端不带 `errorCode`：只能展示文本，**不能**按码分支。
    markBackendTerminal("document closed");
    expect(sessionTerminal()?.code).toBeNull();
});

test("the snapshot reference is stable across reads", () => {
    markBackendTerminal("document closed", ERROR_CODE.DOCUMENT_CLOSED);
    // 【为什么这是硬要求】每次新建对象会让 `useSyncExternalStore` 认为快照一直在变。
    expect(getSessionGateSnapshot()).toBe(getSessionGateSnapshot());
});

test("subscribers are notified once and unsubscribing stops notifications", () => {
    const listener = vi.fn();
    const off = subscribeSessionGate(listener);
    markBackendTerminal("document closed", ERROR_CODE.DOCUMENT_CLOSED);
    expect(listener).toHaveBeenCalledTimes(1);
    // 幂等：第二次标记不再通知。
    markBackendTerminal("document closed again", ERROR_CODE.DOCUMENT_CLOSED);
    expect(listener).toHaveBeenCalledTimes(1);

    off();
    resetSessionGateForTests();
    markBackendTerminal("document closed", ERROR_CODE.DOCUMENT_CLOSED);
    expect(listener).toHaveBeenCalledTimes(1);
});

test("a throwing subscriber does not break the gate", () => {
    const bad = vi.fn(() => {
        throw new Error("subscriber blew up");
    });
    const good = vi.fn();
    subscribeSessionGate(bad);
    subscribeSessionGate(good);
    expect(() => markBackendTerminal("document closed", ERROR_CODE.DOCUMENT_CLOSED)).not.toThrow();
    expect(good).toHaveBeenCalledTimes(1);
    expect(isBackendTerminal()).toBe(true);
});
