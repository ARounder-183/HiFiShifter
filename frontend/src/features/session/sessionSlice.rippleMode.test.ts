/**
 * 波纹编辑的"开关 × 方向"状态：初始值、写入白名单、两根轴互不污染、水合（含旧编码）。
 *
 * 【为什么要单独钉】旧版是一个三值 `rippleMode`，左键循环。改成两根轴之后，最容易
 * 在改动里重新耦合的两件事是：开关顺手改方向、方向顺手关开关。这里把四条边界逐条
 * 钉住，并顺带守住"旧配置不会被读成关着"这条迁移底线。
 */
import { expect, test } from "vitest";

import reducer, { setRippleScope, toggleRippleEnabled } from "./sessionSlice.ts";
import { loadUiSettings } from "./thunks/runtimeThunks.ts";
import type { UiSettings } from "../../services/api/settings.ts";

test("ripple: 初始值、写入白名单、两根轴正交", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    expect(base.rippleEnabled).toBe(false);
    expect(base.rippleScope).toBe("track");

    // 方向写入白名单：非法值静默忽略。
    expect(reducer(base, setRippleScope("all")).rippleScope).toBe("all");
    expect(reducer(base, setRippleScope("off" as never)).rippleScope).toBe("track");

    // 开关不碰方向，方向不碰开关。
    const on = reducer(base, toggleRippleEnabled());
    expect(on.rippleEnabled).toBe(true);
    expect(on.rippleScope).toBe("track");
    const allButOff = reducer(reducer(on, setRippleScope("all")), toggleRippleEnabled());
    expect(allButOff.rippleEnabled).toBe(false);
    expect(allButOff.rippleScope).toBe("all");
});

test("ripple: 水合认新字段，也认旧的三值编码", () => {
    const base = reducer(undefined, { type: "@@INIT" });
    const hydrate = (settings: UiSettings) =>
        reducer(base, loadUiSettings.fulfilled(settings, "req", undefined));

    // 新配置：开关与方向各取各的，关闭时方向仍然记得。
    const modern = hydrate({ rippleEnabled: false, rippleMode: "all" } as UiSettings);
    expect(modern.rippleEnabled).toBe(false);
    expect(modern.rippleScope).toBe("all");

    // 旧配置：没有 rippleEnabled，只能从三值编码推断"当时是开着的"。
    const legacyTrack = hydrate({ rippleMode: "track" } as UiSettings);
    expect(legacyTrack.rippleEnabled).toBe(true);
    expect(legacyTrack.rippleScope).toBe("track");

    const legacyAll = hydrate({ rippleMode: "all" } as UiSettings);
    expect(legacyAll.rippleEnabled).toBe(true);
    expect(legacyAll.rippleScope).toBe("all");

    const legacyOff = hydrate({ rippleMode: "off" } as UiSettings);
    expect(legacyOff.rippleEnabled).toBe(false);
    expect(legacyOff.rippleScope).toBe("track");

    // 完全没有这一项（mock / 插件形态）→ 出厂状态。
    const empty = hydrate({} as UiSettings);
    expect(empty.rippleEnabled).toBe(false);
    expect(empty.rippleScope).toBe("track");
});
