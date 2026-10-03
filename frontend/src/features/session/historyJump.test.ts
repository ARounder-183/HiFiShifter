// @vitest-environment jsdom
/**
 * 「历史位置被整体改变」的统一入口（`./historyJump`）。
 *
 * 【要钉住的性质】它必须同时做两件事：否决在途导入、广播事件让 UI 复位
 * 旧时间线的瞬时状态（音高分析进度）。少任何一件，都会出现"撤销后循环继续灌
 * clip"或"状态栏卡在正在分析音高"。
 */

import { beforeEach, expect, test } from "vitest";

import { HISTORY_JUMP_EVENT, notifyHistoryJump } from "./historyJump";
import {
    currentImportGeneration,
    isImportCancelled,
    registerImportRun,
    resetImportCancellationForTests,
} from "./thunks/importCancellation";

beforeEach(() => {
    resetImportCancellationForTests();
});

test("否决在途导入", async () => {
    const gen = currentImportGeneration();
    expect(isImportCancelled(gen)).toBe(false);
    await notifyHistoryJump();
    expect(isImportCancelled(gen)).toBe(true);
});

test("广播历史跳转事件（供 UI 复位旧时间线的瞬时状态）", async () => {
    let fired = 0;
    const listener = () => {
        fired += 1;
    };
    window.addEventListener(HISTORY_JUMP_EVENT, listener);
    await notifyHistoryJump();
    window.removeEventListener(HISTORY_JUMP_EVENT, listener);
    expect(fired).toBe(1);
});

test("【关键】等在途导入收尾之后才返回（避免命令落在跳转之后）", async () => {
    const run = registerImportRun();
    let settled = false;
    const pending = notifyHistoryJump().then(() => {
        settled = true;
    });

    // 导入尚未收尾：跳转必须还没发生。
    await Promise.resolve();
    await Promise.resolve();
    expect(settled).toBe(false);

    run.finish();
    await pending;
    expect(settled).toBe(true);
});

test("没有在途导入时立即返回", async () => {
    await expect(notifyHistoryJump()).resolves.toBeUndefined();
});
