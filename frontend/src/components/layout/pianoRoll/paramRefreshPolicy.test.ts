// 刷新策略定向合同：同scope不清掉刚画的线，切参数/轨道不能显示错作用域数据。
import { expect, test } from "vitest";
import { shouldClearParamRefresh } from "./paramRefreshPolicy";

test("同scope刷新保留可见曲线", () => {
    expect(shouldClearParamRefresh(false)).toBe(false);
});
test("切参数或轨道必须清掉旧scope曲线", () => {
    expect(shouldClearParamRefresh(true)).toBe(true);
});
