// @vitest-environment jsdom
/*
 * 宿主专属快捷键的让路判定。
 *
 * 【为什么这份测试是这条功能的**唯一**契约】`classifyHostKeyRelay` 决定"哪些按键可以
 * 还给宿主"。判宽一格，`Delete` 就会被送到宿主那里删宿主的片段；判窄一格，用户按 Space
 * 仍然只看到「当前宿主没有提供这个接口」。两种情况都不会崩、不会报错，只会在真机上
 * 表现成"怪怪的" —— 因此每一条规则都必须有一条断言钉住。
 */
import { beforeEach, describe, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({ relay: vi.fn(), publish: vi.fn(), focus: vi.fn() }));

vi.mock("./api/hostKeys", () => ({
    hostKeysApi: {
        relay: mocks.relay,
        publishEscapeSet: mocks.publish,
        focusHost: mocks.focus,
    },
}));

import {
    classifyHostKeyRelay,
    HOST_KEY_FOCUS_RESULT_EVENT,
    handKeyboardToHost,
    type HostKeyFocusOutcome,
    collectHostKeyEscapeStrokes,
    hostKeyRelayEnv,
    isBareStroke,
    isRelayableStroke,
    publishHostKeyEscapeSet,
    relayHostKey,
    resetHostKeyRelayLatchForTests,
    RELAYABLE_KEYS,
    subscribeHostKeyRelayNotice,
    type HostKeyRelayEnv,
    type HostKeyStroke,
} from "./hostKeyRelay";
import { DEFAULT_KEYBINDINGS } from "../features/keybindings/defaultKeybindings";
import type { KeybindingMap } from "../features/keybindings/types";

/** 插件里、没有 ARA 播放请求的宿主 —— 本功能唯一的目标场景。 */
const PLUGIN_NO_TRANSPORT: HostKeyRelayEnv = {
    plugin: true,
    transportControl: false,
    unsupported: false,
};

function stroke(key: string, mods: Partial<HostKeyStroke> = {}): HostKeyStroke {
    return { key, ctrl: false, shift: false, alt: false, meta: false, ...mods };
}

beforeEach(() => {
    mocks.relay.mockReset();
    mocks.publish.mockReset();
    mocks.publish.mockResolvedValue({ ok: true, count: 0 });
    resetHostKeyRelayLatchForTests();
});

describe("让路判定", () => {
    test("插件里、无 ARA 播放请求时，播放键让给宿主", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("space"), PLUGIN_NO_TRANSPORT),
        ).toEqual({ action: "playback.toggle", stroke: stroke("space") });
        expect(
            classifyHostKeyRelay("playback.stop", null, stroke("enter"), PLUGIN_NO_TRANSPORT),
        ).toEqual({ action: "playback.stop", stroke: stroke("enter") });
    });

    test("规则 1：独立 App 里没有宿主可让", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("space"), {
                ...PLUGIN_NO_TRANSPORT,
                plugin: false,
            }),
        ).toBeNull();
    });

    test("规则 2：插件已能控制宿主播放时不让（否则双触发）", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("space"), {
                ...PLUGIN_NO_TRANSPORT,
                transportControl: true,
            }),
        ).toBeNull();
    });

    test("规则 3：只让 transport.control 这一族 —— 剪辑类被拒时绝不可外泄", () => {
        // 【为什么这条最重要】这些动作在插件里也会被拒（宿主没有写接口），但它们
        // 一旦被投给宿主，就是"删宿主的片段""分割宿主的 item"。
        for (const action of [
            "clip.delete",
            "clip.split",
            "clip.copy",
            "edit.undo",
            "project.importMedia",
            "track.delete",
        ]) {
            expect(
                classifyHostKeyRelay(action, "timeline", stroke("delete"), PLUGIN_NO_TRANSPORT),
                action,
            ).toBeNull();
        }
    });

    test("规则 4：带修饰的键不让", () => {
        for (const mods of [{ ctrl: true }, { shift: true }, { alt: true }, { meta: true }]) {
            expect(
                classifyHostKeyRelay(
                    "playback.toggle",
                    null,
                    stroke("space", mods),
                    PLUGIN_NO_TRANSPORT,
                ),
                JSON.stringify(mods),
            ).toBeNull();
        }
    });

    test("规则 5：本会话已明确让不成时不再尝试", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("space"), {
                ...PLUGIN_NO_TRANSPORT,
                unsupported: true,
            }),
        ).toBeNull();
    });

    test("规则 6：只让具名、与键盘布局无关的键", () => {
        for (const key of ["f5", "a", "1", "f1", "contextmenu"]) {
            expect(
                classifyHostKeyRelay("playback.toggle", null, stroke(key), PLUGIN_NO_TRANSPORT),
                key,
            ).toBeNull();
        }
        for (const key of RELAYABLE_KEYS) {
            expect(
                classifyHostKeyRelay("playback.toggle", null, stroke(key), PLUGIN_NO_TRANSPORT),
                key,
            ).not.toBeNull();
        }
    });
});

describe("键位分类", () => {
    test("裸键与可让路键是两个不同的问题", () => {
        expect(isBareStroke(stroke("space"))).toBe(true);
        expect(isBareStroke(stroke("space", { shift: true }))).toBe(false);
        // 裸键但不在白名单里：仍然不让（布局相关的键）。
        expect(isBareStroke(stroke("a"))).toBe(true);
        expect(isRelayableStroke(stroke("a"))).toBe(false);
        expect(isRelayableStroke(stroke("space"))).toBe(true);
    });
});

describe("宿主专属按键集合", () => {
    test("默认键位下只收播放这一族，且去重、排序稳定", () => {
        const env = PLUGIN_NO_TRANSPORT;
        const strokes = collectHostKeyEscapeStrokes(DEFAULT_KEYBINDINGS, env);
        expect(strokes.map((item) => item.key)).toEqual(["enter", "space"]);
        // 稳定顺序：同一输入两次必须逐项相同（日志与测试都依赖它）。
        expect(collectHostKeyEscapeStrokes(DEFAULT_KEYBINDINGS, env)).toEqual(strokes);
    });

    test("用户把播放键改成一个布局相关的键时，它不进集合（也不该被让）", () => {
        const keybindings: KeybindingMap = {
            ...DEFAULT_KEYBINDINGS,
            "playback.toggle": [{ key: "f5" }],
            "playback.stop": [{ key: "__none__" }],
        };
        expect(collectHostKeyEscapeStrokes(keybindings, PLUGIN_NO_TRANSPORT)).toEqual([]);
    });

    test("替补绑定同样进集合（它们本来就该让）", () => {
        const keybindings: KeybindingMap = {
            ...DEFAULT_KEYBINDINGS,
            "playback.toggle": [{ key: "space" }, { key: "enter" }],
            "playback.stop": [{ key: "space" }],
        };
        expect(
            collectHostKeyEscapeStrokes(keybindings, PLUGIN_NO_TRANSPORT).map((item) => item.key),
        ).toEqual(["enter", "space"]);
    });

    test("插件能控制宿主播放时集合为空（此时一条都不该让）", () => {
        expect(
            collectHostKeyEscapeStrokes(DEFAULT_KEYBINDINGS, {
                ...PLUGIN_NO_TRANSPORT,
                transportControl: true,
            }),
        ).toEqual([]);
    });

    test("发布时把完整的键位描述发出去（原生要按它判等）", () => {
        publishHostKeyEscapeSet([stroke("space"), stroke("enter", { shift: true })]);
        expect(mocks.publish).toHaveBeenCalledWith([
            { key: "space", ctrl: false, shift: false, alt: false, meta: false },
            { key: "enter", ctrl: false, shift: true, alt: false, meta: false },
        ]);
    });
});

describe("发往原生", () => {
    test("已接手 ⇒ accepted", async () => {
        mocks.relay.mockResolvedValue({ accepted: true, reason: null });
        expect(await relayHostKey({ action: "playback.toggle", stroke: stroke("space") })).toBe(
            "accepted",
        );
        expect(mocks.relay).toHaveBeenCalledWith("space", "playback.toggle", {
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        });
    });

    test("正常不接手（宿主自己会处理 / 上一拍未走完 / 限流）都不算失败", async () => {
        for (const reason of ["host-native", "in-flight", "rate-limited"]) {
            resetHostKeyRelayLatchForTests();
            mocks.relay.mockResolvedValue({ accepted: false, reason });
            expect(
                await relayHostKey({ action: "playback.toggle", stroke: stroke("space") }),
                reason,
            ).toBe("accepted");
            // 这几种情况不该让会话闩锁生效 —— 下一次按键仍要尝试。
            expect(hostKeyRelayEnv().unsupported, reason).toBe(false);
        }
    });

    test("让不成立 ⇒ unsupported，并且本会话不再尝试", async () => {
        for (const reason of ["unsupported-key", "modified-key"]) {
            resetHostKeyRelayLatchForTests();
            mocks.relay.mockResolvedValue({ accepted: false, reason });
            expect(
                await relayHostKey({ action: "playback.toggle", stroke: stroke("space") }),
                reason,
            ).toBe("unsupported");
            expect(hostKeyRelayEnv().unsupported, reason).toBe(true);
        }
    });

    test("旧后端没有这条命令 ⇒ 同样算让不成立（回落旧文案）", async () => {
        mocks.relay.mockRejectedValue(new Error("method not wired yet"));
        expect(await relayHostKey({ action: "playback.toggle", stroke: stroke("space") })).toBe(
            "unsupported",
        );
        expect(hostKeyRelayEnv().unsupported).toBe(true);
    });
});

describe("通知订阅", () => {
    test("独立 App 里没有宿主消息口：订阅是空操作，退订也不炸", () => {
        delete (window as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__;
        const seen: string[] = [];
        const off = subscribeHostKeyRelayNotice((notice) => seen.push(notice.reason));
        expect(() => off()).not.toThrow();
        expect(seen).toEqual([]);
    });
});

describe("把键盘交给宿主", () => {
    /*
     * 【为什么这条要有测试】这个功能此前把结果**整个吞掉**：成功与失败在界面上完全
     * 同形。用户报障时只能描述成"似乎根本不起作用"。结果必须被广播出去，而且失败
     * 必须带着原因 —— 两种失败（找不到宿主窗口 / 系统拒绝换焦点）指向完全不同的
     * 下一步。
     */
    test("成功 ⇒ ok，且广播一条成功结果", async () => {
        mocks.focus.mockResolvedValue({ ok: true });
        const seen: HostKeyFocusOutcome[] = [];
        const listener = (event: Event) => {
            seen.push((event as CustomEvent<HostKeyFocusOutcome>).detail);
        };
        window.addEventListener(HOST_KEY_FOCUS_RESULT_EVENT, listener);
        const outcome = await handKeyboardToHost();
        window.removeEventListener(HOST_KEY_FOCUS_RESULT_EVENT, listener);
        expect(outcome).toEqual({ ok: true, reason: null });
        expect(seen).toEqual([{ ok: true, reason: null }]);
    });

    test("失败 ⇒ ok=false、原因原样带出", async () => {
        for (const reason of ["no-host-window", "post-failed"]) {
            mocks.focus.mockResolvedValue({ ok: false, reason });
            const seen: HostKeyFocusOutcome[] = [];
            const listener = (event: Event) => {
                seen.push((event as CustomEvent<HostKeyFocusOutcome>).detail);
            };
            window.addEventListener(HOST_KEY_FOCUS_RESULT_EVENT, listener);
            const outcome = await handKeyboardToHost();
            window.removeEventListener(HOST_KEY_FOCUS_RESULT_EVENT, listener);
            expect(outcome, reason).toEqual({ ok: false, reason });
            expect(seen, reason).toEqual([{ ok: false, reason }]);
        }
    });

    test("旧后端没有这条命令 ⇒ unavailable（而不是假装成功）", async () => {
        mocks.focus.mockRejectedValue(new Error("method not wired yet"));
        expect(await handKeyboardToHost()).toEqual({ ok: false, reason: "unavailable" });
    });
});
