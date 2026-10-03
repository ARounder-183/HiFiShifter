/**
 * 文件浏览器拖拽期间的悬停副作用抑制标记（`./fileBrowserDragStore`）。
 *
 * 【要钉住的性质】标记只在"越过阈值"后置真、在所有结束路径置假；幂等，
 * 且不会在用例之间泄漏状态。
 */

import { beforeEach, expect, test } from "vitest";

import {
    isFileBrowserDragActive,
    resetFileBrowserDragStoreForTests,
    setFileBrowserDragActive,
} from "./fileBrowserDragStore";

beforeEach(() => {
    resetFileBrowserDragStoreForTests();
});

test("默认不激活（未拖拽时浮标照常工作）", () => {
    expect(isFileBrowserDragActive()).toBe(false);
});

test("置真后为真，置假后为假", () => {
    setFileBrowserDragActive(true);
    expect(isFileBrowserDragActive()).toBe(true);
    setFileBrowserDragActive(false);
    expect(isFileBrowserDragActive()).toBe(false);
});

test("重复置假是幂等的（多个结束路径各调一次也不会出错）", () => {
    setFileBrowserDragActive(true);
    setFileBrowserDragActive(false);
    setFileBrowserDragActive(false);
    expect(isFileBrowserDragActive()).toBe(false);
});
