// @vitest-environment jsdom
/**
 * DPR 订阅入口自检。
 *
 * 【主要内容】验证 `subscribeDevicePixelRatio` 的三条契约：
 * 1. **换绑**：`(resolution: N dppx)` 只在 dpr 离开 N 时触发一次，回调后必须用新
 *    dpr 重新订阅 —— 否则第二次缩放永远收不到通知（这是最容易写错的一点）；
 * 2. **旧 WebKit 回退**：`MediaQueryList` 没有 `addEventListener` 时走
 *    `addListener`，不得抛错；
 * 3. **退订**：退订后回调不再触发。
 *
 * 【与其他模块的关系】仅覆盖 `useDevicePixelRatio.ts` 的命令式订阅；React 形态
 * 只是它加一层 state，不单独测。不依赖真实 matchMedia（自建替身）。
 */

import { afterEach, describe, expect, it } from "vitest";

import { subscribeDevicePixelRatio } from "./useDevicePixelRatio";

/** 一个可控的 matchMedia 替身：记录订阅过的 query 并允许手动触发变化。 */
function installFakeMatchMedia(mode: "addEventListener" | "addListener") {
    const queries: string[] = [];
    const handlers = new Map<string, () => void>();
    /** 已注册过的全部回调（退订后仍保留引用，用于验证 disposed 守卫）。 */
    const captured: Array<() => void> = [];

    const fakeMatchMedia = (query: string) => {
        queries.push(query);
        const index = queries.length - 1;
        const list: Record<string, unknown> = { media: query, matches: false };
        const register = (handler: () => void) => {
            handlers.set(query, handler);
            captured[index] = handler;
        };
        if (mode === "addEventListener") {
            list.addEventListener = (_type: string, handler: () => void) => register(handler);
            list.removeEventListener = () => {
                handlers.delete(query);
            };
        } else {
            list.addListener = (handler: () => void) => register(handler);
            list.removeListener = () => {
                handlers.delete(query);
            };
        }
        return list as unknown as MediaQueryList;
    };

    const original = Object.getOwnPropertyDescriptor(window, "matchMedia");
    Object.defineProperty(window, "matchMedia", {
        configurable: true,
        writable: true,
        value: fakeMatchMedia,
    });

    return {
        queries,
        /** 触发最近一次订阅的回调（模拟 dpr 离开该 query）。 */
        fire(queryIndex: number) {
            const query = queries[queryIndex];
            const handler = handlers.get(query);
            if (handler === undefined) throw new Error(`no live subscription for ${query}`);
            handler();
        },
        /** 拿到曾注册过的回调（即便已退订），用于验证 disposed 守卫。 */
        captured(queryIndex: number): () => void {
            const handler = captured[queryIndex];
            if (handler === undefined) throw new Error(`no captured handler at ${queryIndex}`);
            return handler;
        },
        restore() {
            if (original) Object.defineProperty(window, "matchMedia", original);
            else Reflect.deleteProperty(window, "matchMedia");
        },
    };
}

function setDevicePixelRatio(value: number): void {
    Object.defineProperty(window, "devicePixelRatio", {
        configurable: true,
        writable: true,
        value,
    });
}

afterEach(() => {
    setDevicePixelRatio(1);
});

describe("subscribeDevicePixelRatio", () => {
    it("dpr 变化后换绑：第二次变化仍能收到通知", () => {
        const fake = installFakeMatchMedia("addEventListener");
        try {
            setDevicePixelRatio(1.5);
            const seen: number[] = [];
            const unsubscribe = subscribeDevicePixelRatio((dpr) => seen.push(dpr));

            expect(fake.queries).toHaveLength(1);
            expect(fake.queries[0]).toContain("1.5");

            // 第一次变化：1.5 → 2
            setDevicePixelRatio(2);
            fake.fire(0);
            expect(seen).toEqual([2]);
            // 必须已用新 dpr 重新订阅（旧 query 已永久失配）。
            expect(fake.queries).toHaveLength(2);
            expect(fake.queries[1]).toContain("2");

            // 第二次变化：2 → 1 —— 换绑没做对的话这里收不到。
            setDevicePixelRatio(1);
            fake.fire(1);
            expect(seen).toEqual([2, 1]);

            unsubscribe();
        } finally {
            fake.restore();
        }
    });

    it("MediaQueryList 只有 addListener 时走回退路径，不抛错", () => {
        const fake = installFakeMatchMedia("addListener");
        try {
            setDevicePixelRatio(1.25);
            const seen: number[] = [];
            const unsubscribe = subscribeDevicePixelRatio((dpr) => seen.push(dpr));

            expect(fake.queries).toHaveLength(1);
            setDevicePixelRatio(1.75);
            fake.fire(0);
            expect(seen).toEqual([1.75]);

            unsubscribe();
        } finally {
            fake.restore();
        }
    });

    it("退订后不再回调", () => {
        const fake = installFakeMatchMedia("addEventListener");
        try {
            setDevicePixelRatio(1);
            const seen: number[] = [];
            const unsubscribe = subscribeDevicePixelRatio((dpr) => seen.push(dpr));
            const stale = fake.captured(0);

            unsubscribe();
            // 退订必须解绑：替身里 handler 已被移除。
            expect(() => fake.fire(0)).toThrow();

            // 即便持有过期回调引用再触发，也不得回调（disposed 守卫）。
            setDevicePixelRatio(3);
            stale();
            expect(seen).toEqual([]);
        } finally {
            fake.restore();
        }
    });

    it("环境没有 matchMedia 时返回空退订函数，不抛错", () => {
        const original = Object.getOwnPropertyDescriptor(window, "matchMedia");
        Reflect.deleteProperty(window, "matchMedia");
        try {
            expect(() => subscribeDevicePixelRatio(() => undefined)()).not.toThrow();
        } finally {
            if (original) Object.defineProperty(window, "matchMedia", original);
        }
    });
});
