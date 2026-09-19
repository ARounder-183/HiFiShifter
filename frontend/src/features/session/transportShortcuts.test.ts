/**
 * 传输快捷键语义单测（`./transportShortcuts`）。
 *
 * 【本测试要锁住的回归】提交 77553e61 为了消除"重复触发播放"删掉了
 * `playback.stop` 空闲分支的 `playOriginal()`，于是默认 `Enter`
 * （标签「播放 / 停止」、手册写"播放/停止"）在未播放时完全无反应。
 *
 * 这张裁决表是修复后行为的唯一事实源：空闲时必须 `play`，播放中
 * `toggle` 必须 `pause`、`stop` 必须 `stop`。
 */
import { describe, expect, it } from "vitest";

import {
    resolveTransportShortcutCommand,
    type TransportShortcutActionId,
} from "./transportShortcuts";

describe("resolveTransportShortcutCommand", () => {
    it("【回归】空闲时 Enter（playback.stop）必须起播，而不是 no-op", () => {
        expect(resolveTransportShortcutCommand("playback.stop", false)).toBe("play");
    });

    it("空闲时 Space（playback.toggle）起播", () => {
        expect(resolveTransportShortcutCommand("playback.toggle", false)).toBe("play");
    });

    it("播放中 Space = 暂停（光标留在当前位置）", () => {
        expect(resolveTransportShortcutCommand("playback.toggle", true)).toBe("pause");
    });

    it("播放中 Enter = 停止（光标回到本次起播位置）", () => {
        expect(resolveTransportShortcutCommand("playback.stop", true)).toBe("stop");
    });

    it("暂停与停止是两个不同动作（光标落点不同）", () => {
        const ids: TransportShortcutActionId[] = ["playback.toggle", "playback.stop"];
        const playing = ids.map((id) => resolveTransportShortcutCommand(id, true));
        expect(playing[0]).not.toBe(playing[1]);
    });
});
