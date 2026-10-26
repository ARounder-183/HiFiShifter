// @vitest-environment jsdom
/*
 * 宿主专属快捷键的让路判定。
 *
 * 【为什么这份测试是这条功能的**唯一**契约】`classifyHostKeyRelay` 与
 * `classifyUnboundHostKey` 决定"哪些按键可以还给宿主"。判宽一格，`Delete` 就会被送到
 * 宿主那里删宿主的片段；判窄一格，用户按 Space 仍然只看到「当前宿主没有提供这个接口」。
 * 两种情况都不会崩、不会报错，只会在真机上表现成"怪怪的" —— 因此每一条规则都必须有
 * 一条断言钉住。
 */
import { beforeEach, describe, expect, test, vi } from "vitest";

const mocks = vi.hoisted(() => ({ relay: vi.fn(), publish: vi.fn(), focus: vi.fn() }));

vi.mock("./api/hostKeys", () => ({
    hostKeysApi: {
        relay: mocks.relay,
        publishOwnedStrokes: mocks.publish,
        focusHost: mocks.focus,
    },
}));

import {
    classifyHostKeyRelay,
    classifyUnboundHostKey,
    collectOwnedStrokes,
    handKeyboardToHost,
    hostKeyRelayEnv,
    isBareStroke,
    isRelayableStroke,
    NON_RELAYABLE_CODES,
    publishOwnedStrokes,
    relayHostKey,
    resetHostKeyRelayLatchForTests,
    RELAYABLE_CODES,
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

function stroke(code: string, mods: Partial<HostKeyStroke> = {}): HostKeyStroke {
    return { code, ctrl: false, shift: false, alt: false, meta: false, ...mods };
}

beforeEach(() => {
    mocks.relay.mockReset();
    mocks.publish.mockReset();
    mocks.publish.mockResolvedValue({ ok: true, count: 0 });
    resetHostKeyRelayLatchForTests();
});

describe("入口 A：插件认识这个键，但宿主那一族", () => {
    test("插件里、无 ARA 播放请求时，播放键让给宿主", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("Space"), PLUGIN_NO_TRANSPORT),
        ).toEqual({ action: "playback.toggle", stroke: stroke("Space") });
        expect(
            classifyHostKeyRelay("playback.stop", null, stroke("Enter"), PLUGIN_NO_TRANSPORT),
        ).toEqual({ action: "playback.stop", stroke: stroke("Enter") });
    });

    test("节拍器与录音同属宿主那一族（宿主有自己的节拍器与录音键）", () => {
        /*
         * 【为什么这两条要用 F5/F6 而不是默认的 K / Ctrl+R】两族判定是**与关系**：
         * 动作属于宿主那一族（规则 3）**且**这个物理键在可让表里（规则 6）。
         * 默认的 `K` 是字母（打字领地）、`Ctrl+R` 带修饰（无法忠实重放），所以它们
         * 在默认键位下并不会被让 —— 这是对的，不是漏。用户把节拍器改绑到 F5 之后
         * 才真正走这条路。
         */
        expect(
            classifyHostKeyRelay("playback.metronome", null, stroke("F5"), PLUGIN_NO_TRANSPORT),
        ).not.toBeNull();
        expect(
            classifyHostKeyRelay("recording.toggle", null, stroke("F6"), PLUGIN_NO_TRANSPORT),
        ).not.toBeNull();
        // 默认键位下：字母与带修饰的组合都不让。
        expect(
            classifyHostKeyRelay("playback.metronome", null, stroke("KeyK"), PLUGIN_NO_TRANSPORT),
        ).toBeNull();
        expect(
            classifyHostKeyRelay(
                "recording.toggle",
                null,
                stroke("KeyR", { ctrl: true }),
                PLUGIN_NO_TRANSPORT,
            ),
        ).toBeNull();
    });

    test("规则 1：独立 App 里没有宿主可让", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("Space"), {
                ...PLUGIN_NO_TRANSPORT,
                plugin: false,
            }),
        ).toBeNull();
    });

    test("规则 2：插件已能控制宿主播放时不让（否则双触发）", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("Space"), {
                ...PLUGIN_NO_TRANSPORT,
                transportControl: true,
            }),
        ).toBeNull();
    });

    test("规则 3：只让宿主那一族 —— 剪辑类被拒时绝不可外泄", () => {
        // 【为什么这条最重要】这些动作在插件里也会被拒（宿主没有写接口），但它们
        // 一旦被投给宿主，就是"删宿主的片段""分割宿主的 item""撤销宿主的编辑"。
        for (const action of [
            "clip.delete",
            "clip.split",
            "clip.copy",
            "edit.undo",
            "project.importMedia",
            "track.delete",
            "edit.pasteTracks",
        ]) {
            expect(
                classifyHostKeyRelay(action, "timeline", stroke("Delete"), PLUGIN_NO_TRANSPORT),
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
                    stroke("Space", mods),
                    PLUGIN_NO_TRANSPORT,
                ),
                JSON.stringify(mods),
            ).toBeNull();
        }
    });

    test("规则 5：本会话已明确让不成时不再尝试", () => {
        expect(
            classifyHostKeyRelay("playback.toggle", null, stroke("Space"), {
                ...PLUGIN_NO_TRANSPORT,
                unsupported: true,
            }),
        ).toBeNull();
    });

    test("规则 6：只让具名、与键盘布局无关的键", () => {
        for (const code of ["KeyA", "Digit1", "Semicolon", "F25", ""]) {
            expect(
                classifyHostKeyRelay("playback.toggle", null, stroke(code), PLUGIN_NO_TRANSPORT),
                code,
            ).toBeNull();
        }
        for (const code of RELAYABLE_CODES) {
            expect(
                classifyHostKeyRelay("playback.toggle", null, stroke(code), PLUGIN_NO_TRANSPORT),
                code,
            ).not.toBeNull();
        }
    });
});

describe("入口 B：一条绑定都没命中", () => {
    test("具名裸键让给宿主 —— 这是「各种常用快捷键都不生效」的正面答案", () => {
        for (const code of ["Home", "End", "PageUp", "PageDown", "F2", "F5", "Insert"]) {
            expect(classifyUnboundHostKey(stroke(code), PLUGIN_NO_TRANSPORT), code).not.toBeNull();
        }
    });

    test("可打印字符不让：那会把插件窗口变成透明的键盘转发器", () => {
        for (const code of ["KeyY", "Semicolon", "Digit7", "Backquote"]) {
            expect(classifyUnboundHostKey(stroke(code), PLUGIN_NO_TRANSPORT), code).toBeNull();
        }
    });

    test("Escape / Tab 不让：宿主里它们几乎总是关窗口与换焦点", () => {
        for (const code of NON_RELAYABLE_CODES) {
            expect(classifyUnboundHostKey(stroke(code), PLUGIN_NO_TRANSPORT), code).toBeNull();
        }
    });

    test("带修饰、独立 App、已闩锁 —— 三种情况都不让", () => {
        expect(
            classifyUnboundHostKey(stroke("Home", { ctrl: true }), PLUGIN_NO_TRANSPORT),
        ).toBeNull();
        expect(
            classifyUnboundHostKey(stroke("Home"), { ...PLUGIN_NO_TRANSPORT, plugin: false }),
        ).toBeNull();
        expect(
            classifyUnboundHostKey(stroke("Home"), { ...PLUGIN_NO_TRANSPORT, unsupported: true }),
        ).toBeNull();
    });

    test("插件能控制宿主播放时仍然让（未绑定的键与传输能力无关）", () => {
        expect(
            classifyUnboundHostKey(stroke("Home"), {
                ...PLUGIN_NO_TRANSPORT,
                transportControl: true,
            }),
        ).not.toBeNull();
    });
});

describe("键位分类", () => {
    test("裸键与可让路键是两个不同的问题", () => {
        expect(isBareStroke(stroke("Space"))).toBe(true);
        expect(isBareStroke(stroke("Space", { shift: true }))).toBe(false);
        // 裸键但不在白名单里：仍然不让（布局相关的键）。
        expect(isBareStroke(stroke("KeyA"))).toBe(true);
        expect(isRelayableStroke(stroke("KeyA"))).toBe(false);
        expect(isRelayableStroke(stroke("Space"))).toBe(true);
    });
});

describe("插件占用的按键集合", () => {
    test("默认键位下收全部具名绑定，且去重、排序稳定", () => {
        const strokes = collectOwnedStrokes(DEFAULT_KEYBINDINGS, PLUGIN_NO_TRANSPORT);
        const codes = strokes.map((item) => item.code);
        // 播放 / 停止 / 节拍器 / 工具键都在里面。
        for (const code of ["Space", "Enter", "Tab", "F6", "F7", "F8", "F9", "F10"]) {
            expect(codes, code).toContain(code);
        }
        // 字母键不在可让表里，因此不进集合（它们本来也不可能被让出去）。
        expect(codes).not.toContain("KeyS");
        // 稳定顺序：同一输入两次必须逐项相同（日志与测试都依赖它）。
        expect(collectOwnedStrokes(DEFAULT_KEYBINDINGS, PLUGIN_NO_TRANSPORT)).toEqual(strokes);
    });

    test("用户改键之后集合跟着变", () => {
        const keybindings: KeybindingMap = {
            ...DEFAULT_KEYBINDINGS,
            "playback.toggle": [{ key: "f2" }],
        };
        const codes = collectOwnedStrokes(keybindings, PLUGIN_NO_TRANSPORT).map(
            (item) => item.code,
        );
        expect(codes).toContain("F2");
        // Space 仍被占用 —— 快速搜索的试听键也是它。这正是"按物理键记账"的意义：
        // 集合回答的是"插件占用了哪些键"，不是"哪个动作用了它"。
        expect(codes).toContain("Space");
        // 但整体必须与默认表不同（改键确实传导到了集合）。
        expect(codes).not.toEqual(
            collectOwnedStrokes(DEFAULT_KEYBINDINGS, PLUGIN_NO_TRANSPORT).map((item) => item.code),
        );
    });

    test("修饰位跟着绑定走（Ctrl+Shift+H 是一份不同的占用）", () => {
        const codes = collectOwnedStrokes(DEFAULT_KEYBINDINGS, PLUGIN_NO_TRANSPORT).filter(
            (item) => item.code === "KeyH",
        );
        // 键位表里的字母不进集合；这里只确认"修饰位不丢"这条契约仍然成立。
        expect(codes).toEqual([]);
        const strokes = collectOwnedStrokes(
            { ...DEFAULT_KEYBINDINGS, "playback.toggle": [{ key: "space", ctrl: true }] },
            PLUGIN_NO_TRANSPORT,
        );
        expect(strokes).toContainEqual({
            code: "Space",
            ctrl: true,
            shift: false,
            alt: false,
            meta: false,
        });
    });

    test("独立 App 里不发（没有宿主要防）", () => {
        expect(
            collectOwnedStrokes(DEFAULT_KEYBINDINGS, { ...PLUGIN_NO_TRANSPORT, plugin: false }),
        ).toEqual([]);
    });

    test("发布时把完整的键位描述发出去（原生要按它判等）", () => {
        publishOwnedStrokes([stroke("Space"), stroke("Enter", { shift: true })]);
        expect(mocks.publish).toHaveBeenCalledWith([
            { code: "Space", ctrl: false, shift: false, alt: false, meta: false },
            { code: "Enter", ctrl: false, shift: true, alt: false, meta: false },
        ]);
    });
});

describe("发往原生", () => {
    test("已接手 ⇒ accepted", async () => {
        mocks.relay.mockResolvedValue({ accepted: true, reason: null });
        expect(await relayHostKey({ action: "playback.toggle", stroke: stroke("Space") })).toBe(
            "accepted",
        );
        expect(mocks.relay).toHaveBeenCalledWith("Space", "playback.toggle", {
            ctrl: false,
            shift: false,
            alt: false,
            meta: false,
        });
    });

    test("未绑定的键把 action 发成 null（原生据此跳过探针）", async () => {
        mocks.relay.mockResolvedValue({ accepted: true, reason: null });
        await relayHostKey({ action: null, stroke: stroke("F2") });
        expect(mocks.relay).toHaveBeenCalledWith("F2", null, {
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
                await relayHostKey({ action: "playback.toggle", stroke: stroke("Space") }),
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
                await relayHostKey({ action: "playback.toggle", stroke: stroke("Space") }),
                reason,
            ).toBe("unsupported");
            expect(hostKeyRelayEnv().unsupported, reason).toBe(true);
        }
    });

    test("旧后端没有这条命令 ⇒ 同样算让不成立（回落旧文案）", async () => {
        mocks.relay.mockRejectedValue(new Error("method not wired yet"));
        expect(await relayHostKey({ action: "playback.toggle", stroke: stroke("Space") })).toBe(
            "unsupported",
        );
        expect(hostKeyRelayEnv().unsupported).toBe(true);
    });
});

describe("把键盘交给宿主", () => {
    /*
     * 【为什么这条要有测试】这个功能此前把结果**整个吞掉**：成功与失败在界面上完全
     * 同形。用户报障时只能描述成"似乎根本不起作用"。因此结果必须被**返回**出去 ——
     * 调用方（快捷键分支）据此决定要不要说一句。
     */
    test("成功 ⇒ ok，reason 为 null", async () => {
        mocks.focus.mockResolvedValue({ ok: true });
        expect(await handKeyboardToHost()).toEqual({ ok: true, reason: null });
    });

    test("失败 ⇒ ok=false、原因原样带出", async () => {
        for (const reason of ["no-host-window", "post-failed"]) {
            mocks.focus.mockResolvedValue({ ok: false, reason });
            expect(await handKeyboardToHost(), reason).toEqual({ ok: false, reason });
        }
    });

    test("旧后端没有这条命令 ⇒ unavailable（而不是假装成功）", async () => {
        mocks.focus.mockRejectedValue(new Error("method not wired yet"));
        expect(await handKeyboardToHost()).toEqual({ ok: false, reason: "unavailable" });
    });
});
