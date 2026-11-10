// 刷新策略定向合同：同scope不清掉刚画的线，切参数/轨道不能显示错作用域数据。
import { expect, test } from "vitest";
import { shouldAcceptParamPayload, shouldClearParamRefresh } from "./paramRefreshPolicy";

test("同scope刷新保留可见曲线", () => {
    expect(shouldClearParamRefresh(false)).toBe(false);
});
test("切参数或轨道必须清掉旧scope曲线", () => {
    expect(shouldClearParamRefresh(true)).toBe(true);
});

test("同窗口空回包保留旧曲线", () => {
    expect(
        shouldAcceptParamPayload({
            hasValues: false,
            requestedKey: "v2|root|pitch|0|100|1",
            previousKey: "v2|root|pitch|0|100|1",
            previousHasValues: true,
        }),
    ).toBe(false);
});

test("不同窗口空回包必须接受，避免旧窗口曲线留在新坐标上", () => {
    expect(
        shouldAcceptParamPayload({
            hasValues: false,
            requestedKey: "v2|root|pitch|200|100|1",
            previousKey: "v2|root|pitch|0|100|1",
            previousHasValues: true,
        }),
    ).toBe(true);
});

test("首次加载与已有值回包都接受", () => {
    expect(
        shouldAcceptParamPayload({
            hasValues: false,
            requestedKey: "v2|root|pitch|0|100|1",
            previousKey: null,
            previousHasValues: false,
        }),
    ).toBe(true);
    expect(
        shouldAcceptParamPayload({
            hasValues: true,
            requestedKey: "v2|root|pitch|0|100|1",
            previousKey: "v2|root|pitch|0|100|1",
            previousHasValues: true,
        }),
    ).toBe(true);
});
