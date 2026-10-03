/**
 * 导入取消闸门（`./importCancellation`）。
 *
 * 【要钉住的性质】代次是全局的：一次"否决"（撤销 / 重做 / 历史跳转 / 打开工程）
 * 让**所有**在途循环的下一轮都能发现自己已过期；而在否决之后**新开始**的导入
 * 不应被误判为已取消。
 */

import { beforeEach, expect, test } from "vitest";

import {
    cancelActiveImports,
    currentImportGeneration,
    isImportCancelled,
    resetImportCancellationForTests,
} from "./importCancellation";

beforeEach(() => {
    resetImportCancellationForTests();
});

test("未否决时，本代次的导入不算被取消", () => {
    const gen = currentImportGeneration();
    expect(isImportCancelled(gen)).toBe(false);
});

test("否决之后，先前记下的代次即过期", () => {
    const gen = currentImportGeneration();
    cancelActiveImports();
    expect(isImportCancelled(gen)).toBe(true);
});

test("否决之后新开始的导入不受影响（记下的是新代次）", () => {
    const stale = currentImportGeneration();
    cancelActiveImports();
    const fresh = currentImportGeneration();
    expect(isImportCancelled(fresh)).toBe(false);
    expect(isImportCancelled(stale)).toBe(true);
});

test("连续两次否决：更早的代次同样过期，最近开始的仍有效", () => {
    const first = currentImportGeneration();
    cancelActiveImports();
    const second = currentImportGeneration();
    cancelActiveImports();
    const third = currentImportGeneration();
    expect(isImportCancelled(first)).toBe(true);
    expect(isImportCancelled(second)).toBe(true);
    expect(isImportCancelled(third)).toBe(false);
});
