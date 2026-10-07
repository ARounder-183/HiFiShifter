// 刷新策略定向合同：同scope不清掉刚画的线，切参数/轨道不能显示错作用域数据。
import { expect, test } from "vitest";
import { shouldClearParamRefresh } from "./paramRefreshPolicy";

test("插件提交/渲染通知在同scope保留可见曲线", () => {
    expect(shouldClearParamRefresh(false, true)).toBe(false);
});
test("插件切参数或轨道必须清掉旧scope曲线", () => {
    expect(shouldClearParamRefresh(true, true)).toBe(true);
});
test("独立App保持原有刷新规则", () => {
    expect(shouldClearParamRefresh(false, false)).toBe(true);
    expect(shouldClearParamRefresh(true, false)).toBe(true);
});
