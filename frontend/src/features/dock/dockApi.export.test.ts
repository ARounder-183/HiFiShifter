/*
 * 布局导出的往返契约。
 *
 * 【为什么必须有】导出是"可分享的 JSON"，导入必须能还原用户显式设置过的布局级
 * 偏好。`normalizeDockLayout` 会读取 `tabPosition`，导出漏掉它会让"标签行在上"
 * 在导出→导入后静默回到默认的下方。
 */
import { expect, test } from "vitest";

import { exportLayoutJsonFromLayout } from "./dockApi";
import { createDefaultDockLayout, normalizeDockLayout } from "./dockSchema";

test("导出 JSON 保留 tabPosition，导入后不丢失", () => {
    const layout = { ...createDefaultDockLayout(), tabPosition: "top" as const };
    const json = exportLayoutJsonFromLayout(layout);
    expect(JSON.parse(json).tabPosition).toBe("top");
    expect(normalizeDockLayout(JSON.parse(json)).tabPosition).toBe("top");
});
