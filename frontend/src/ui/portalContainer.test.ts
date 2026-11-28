// @vitest-environment jsdom
import { afterEach, expect, test } from "vitest";

import { menuPortalContainer } from "./portalContainer";

afterEach(() => {
    document.body.innerHTML = "";
});

test("优先挂到 .radix-themes，拿到完整 Radix 令牌", () => {
    const theme = document.createElement("div");
    theme.className = "radix-themes";
    document.body.append(theme);
    expect(menuPortalContainer()).toBe(theme);
});

test("无主题作用域时退回 body", () => {
    expect(menuPortalContainer()).toBe(document.body);
});
