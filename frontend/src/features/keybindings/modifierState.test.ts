// 虚拟修饰键 → 修饰键视图：桌面恒等，移动端并集。
import { afterEach, describe, expect, it } from "vitest";
import {
    clearModifierState,
    modifierView,
    toggleModifierLock,
    toggleModifierOnce,
    virtualModifierFlags,
} from "./modifierState";
import type { ActionId } from "./types";

const SLIP = "modifier.clipSlipEdit" as ActionId;

function event(
    overrides: Partial<Record<"ctrlKey" | "shiftKey" | "altKey" | "metaKey", boolean>> = {},
) {
    return {
        ctrlKey: false,
        shiftKey: false,
        altKey: false,
        metaKey: false,
        ...overrides,
    };
}

afterEach(() => clearModifierState());

describe("modifierView", () => {
    it("★ 没有锁存时原样返回入参（同一个对象引用）—— 桌面零回归的锚点", () => {
        const e = event({ shiftKey: true });
        expect(modifierView(e)).toBe(e);
    });

    it("★ 锁存后 ctrl 为真，其余三个修饰键仍取物理值（并集，不是覆盖）", () => {
        toggleModifierOnce(SLIP);
        const view = modifierView(event({ shiftKey: true, altKey: true }));
        expect(view.ctrlKey).toBe(true);
        expect(view.shiftKey).toBe(true);
        expect(view.altKey).toBe(true);
        expect(view.metaKey).toBe(false);
        // 不能返回原对象：那会把事件对象「改成」按住 Ctrl。
        expect(view).not.toBe(event());
    });

    it("锁定态同样生效", () => {
        toggleModifierLock(SLIP);
        expect(modifierView(event()).ctrlKey).toBe(true);
    });

    it("解锁后回到恒等", () => {
        toggleModifierLock(SLIP);
        toggleModifierLock(SLIP);
        const e = event();
        expect(modifierView(e)).toBe(e);
    });
});

describe("virtualModifierFlags", () => {
    it("有任何锁存即为 ctrl（统一映射，不区分物理键）", () => {
        expect(virtualModifierFlags().ctrl).toBe(false);
        toggleModifierOnce(SLIP);
        expect(virtualModifierFlags()).toEqual({
            ctrl: true,
            shift: false,
            alt: false,
            meta: false,
        });
    });
});
