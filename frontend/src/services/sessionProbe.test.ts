// @vitest-environment jsdom
/**
 * 探活的**终态判定**契约。
 *
 * 【为什么这条必须有测试】`ARA_NOT_BOUND` 被当成终态时，整页会替换成"插件会话不可用"。
 * 对 ARA 形态那是对的（宿主从不走 ARA 路径，关窗重开一万次也一样）；但 Capture 形态
 * **本来就不需要 ARA 文档** —— 把它也判成终态，等于让所有不支持 ARA 的宿主（Cubase 也
 * 在内）永远打不开界面，而那正是 Capture 存在的理由。这条断言把两种形态分开钉住。
 */
import { afterEach, beforeEach, describe, expect, test } from "vitest";

import { sessionUnavailableFrom } from "./sessionProbe";

/** `invokeErrorKind` 读的就是这个字段（见 `services/invoke.ts`）。 */
const terminal = { kind: "terminal", message: "ARA_NOT_BOUND" };
const transient = { kind: "transient", message: "editor not connected yet" };

function bootstrap(hostKind: string): void {
    (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__ = {
        version: 1,
        hostKind,
    };
}

beforeEach(() => {
    delete (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__;
});

afterEach(() => {
    delete (window as unknown as { __HFS_PLUGIN_BOOTSTRAP__?: unknown }).__HFS_PLUGIN_BOOTSTRAP__;
});

describe("探活的终态判定", () => {
    test("ARA 形态：终态 ⇒ 整页提示；瞬时失败不算", () => {
        bootstrap("ara_only");
        expect(sessionUnavailableFrom(terminal)).toBe(true);
        expect(sessionUnavailableFrom(transient)).toBe(false);
    });

    test("Capture 形态：终态也不整页替换（它不需要 ARA 文档）", () => {
        bootstrap("capture");
        expect(
            sessionUnavailableFrom(terminal),
            "Capture 的宿主正是会报 ARA_NOT_BOUND 的那些；判成不可用会让界面永远打不开",
        ).toBe(false);
        expect(sessionUnavailableFrom(transient)).toBe(false);
    });

    test("REAPER 形态：与非 Capture 形态一致", () => {
        bootstrap("reaper");
        expect(sessionUnavailableFrom(terminal)).toBe(true);
    });

    test("没有 bootstrap（独立 App）：终态判定不适用，但函数本身不抛", () => {
        expect(sessionUnavailableFrom(transient)).toBe(false);
        expect(sessionUnavailableFrom(terminal)).toBe(true);
    });
});
