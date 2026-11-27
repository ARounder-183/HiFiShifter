// @vitest-environment jsdom
import { afterEach, describe, expect, test } from "vitest";

import {
    handleMenuDismissPointerDown,
    menuDismissLayerCount,
    pushMenuDismissLayer,
    resetMenuDismissStack,
} from "./menuDismissStack";

/** 造一个"表面"元素。 */
function surface(): HTMLDivElement {
    const element = document.createElement("div");
    document.body.append(element);
    return element;
}

afterEach(() => {
    resetMenuDismissStack();
    document.body.innerHTML = "";
});

describe("menuDismissStack", () => {
    test("点在里层菜单里 ⇒ 外层不关、里层不关", () => {
        const outer = surface();
        const inner = surface();
        const innerItem = document.createElement("span");
        inner.append(innerItem);
        const closed: string[] = [];
        pushMenuDismissLayer(outer, () => closed.push("outer"));
        pushMenuDismissLayer(inner, () => closed.push("inner"));

        handleMenuDismissPointerDown(innerItem);
        expect(closed).toEqual([]);
    });

    test("点在两层之外 ⇒ 全关（由深到浅）", () => {
        const outer = surface();
        const inner = surface();
        const closed: string[] = [];
        pushMenuDismissLayer(outer, () => closed.push("outer"));
        pushMenuDismissLayer(inner, () => closed.push("inner"));

        handleMenuDismissPointerDown(document.body);
        expect(closed).toEqual(["inner", "outer"]);
    });

    test("点在外层里、里层外 ⇒ 只关里层", () => {
        const outer = surface();
        const inner = surface();
        const outerItem = document.createElement("span");
        outer.append(outerItem);
        const closed: string[] = [];
        pushMenuDismissLayer(outer, () => closed.push("outer"));
        pushMenuDismissLayer(inner, () => closed.push("inner"));

        handleMenuDismissPointerDown(outerItem);
        expect(closed).toEqual(["inner"]);
    });

    test("注销后不再参与（卸载的菜单不该被再关一次）", () => {
        const outer = surface();
        const closed: string[] = [];
        const release = pushMenuDismissLayer(outer, () => closed.push("outer"));
        expect(menuDismissLayerCount()).toBe(1);
        release();
        expect(menuDismissLayerCount()).toBe(0);
        handleMenuDismissPointerDown(document.body);
        expect(closed).toEqual([]);
    });

    test("dismiss 期间就地注销不会漏掉其它层", () => {
        const first = surface();
        const second = surface();
        const closed: string[] = [];
        const releaseFirst = pushMenuDismissLayer(first, () => {
            closed.push("first");
            releaseFirst();
        });
        pushMenuDismissLayer(second, () => {
            closed.push("second");
        });
        handleMenuDismissPointerDown(document.body);
        expect(closed).toEqual(["second", "first"]);
        void releaseFirst;
    });
});
