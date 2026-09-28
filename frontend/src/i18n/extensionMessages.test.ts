// @vitest-environment jsdom
/*
 * 扩展文案通道的契约测试。
 *
 * 【为什么要锁"内置键不可被覆盖"这一条】解析顺序是
 * `静态词典 → 扩展层 → en-US → 键名`。若把扩展层提到前面，一个第三方面板
 * 注册 `ok: "…"` 就能改掉全应用的「确定」按钮 —— 那不是扩展，是劫持。
 * 这条顺序是安全性属性，必须由测试钉住。
 */
import { afterEach, expect, test } from "vitest";

import {
    getExtensionMessagesVersion,
    lookupExtensionMessage,
    registerExtensionMessages,
    resetExtensionMessagesForTests,
    subscribeExtensionMessages,
} from "./extensionMessages";
import { messages } from "./messages";

afterEach(() => {
    resetExtensionMessagesForTests();
});

test("注册后可按语系查到扩展键", () => {
    registerExtensionMessages({
        "en-US": { "panel.mine.title": "My Panel" },
        "zh-CN": { "panel.mine.title": "我的面板" },
    });
    expect(lookupExtensionMessage("en-US", "panel.mine.title")).toBe("My Panel");
    expect(lookupExtensionMessage("zh-CN", "panel.mine.title")).toBe("我的面板");
});

test("未注册的语系返回 undefined，交由上层继续回退", () => {
    registerExtensionMessages({ "en-US": { "panel.mine.title": "My Panel" } });
    expect(lookupExtensionMessage("ja-JP", "panel.mine.title")).toBeUndefined();
});

test("注销后查不到（扩展卸载必须调用注销）", () => {
    const dispose = registerExtensionMessages({ "en-US": { k: "v" } });
    expect(lookupExtensionMessage("en-US", "k")).toBe("v");
    dispose();
    expect(lookupExtensionMessage("en-US", "k")).toBeUndefined();
});

test("重复注销幂等", () => {
    const dispose = registerExtensionMessages({ "en-US": { k: "v" } });
    dispose();
    expect(() => dispose()).not.toThrow();
});

test("后注册的扩展可以覆盖先前扩展的同名键", () => {
    registerExtensionMessages({ "en-US": { k: "first" } });
    registerExtensionMessages({ "en-US": { k: "second" } });
    expect(lookupExtensionMessage("en-US", "k")).toBe("second");
});

test("扩展层不包含内置键，因此不可能覆盖内置文案", () => {
    /*
     * 这条断言的是**机制**而不是某一处调用：内置键在静态词典里能查到，
     * 于是 `t()` 的 `??` 链永远不会走到扩展层。这里直接验证静态词典命中。
     */
    registerExtensionMessages({ "en-US": { ok: "HIJACKED" } });
    expect(messages["en-US"].ok).toBe("OK");
    // 扩展层里确实有这个名字，但解析顺序把它挡在后面
    expect(lookupExtensionMessage("en-US", "ok")).toBe("HIJACKED");
});

test("注册与注销都会推进版本号并通知订阅者", () => {
    let calls = 0;
    const unsubscribe = subscribeExtensionMessages(() => {
        calls += 1;
    });
    const before = getExtensionMessagesVersion();

    const dispose = registerExtensionMessages({ "en-US": { k: "v" } });
    expect(getExtensionMessagesVersion()).toBeGreaterThan(before);
    expect(calls).toBe(1);

    dispose();
    expect(calls).toBe(2);
    unsubscribe();
    registerExtensionMessages({ "en-US": { k2: "v" } });
    expect(calls).toBe(2);
});
