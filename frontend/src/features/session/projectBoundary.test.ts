/**
 * 可滚域求解契约。
 *
 * 【为什么要单测】时间轴、内核、参数编辑器三处必须共用同一个可滚时长；这里锁定的
 * 是“额外实时来源（例如正在进行的 Capture）只能把可滚域**延长**，绝不能缩短权威
 * projectSec 或 clip 末端”这一不变量。
 */
import { expect, test } from "vitest";

import { resolveScrollableProjectSec } from "./projectBoundary";
import type { ClipInfo } from "./sessionTypes";

/** 只填本函数关心的两个字段；其余字段与本契约无关。 */
function clip(startSec: number, lengthSec: number): ClipInfo {
    return { startSec, lengthSec } as ClipInfo;
}

test("额外实时长度可以延伸可滚域", () => {
    expect(resolveScrollableProjectSec(10, [clip(0, 5)], 25)).toBe(25);
});

test("额外长度小于权威值时不缩短", () => {
    expect(resolveScrollableProjectSec(30, [clip(0, 5)], 12)).toBe(30);
});

test("额外长度小于 clip 末端时不缩短", () => {
    expect(resolveScrollableProjectSec(10, [clip(0, 20)], 12)).toBe(20);
});

test("非法额外长度按 0 处理", () => {
    expect(resolveScrollableProjectSec(10, [clip(0, 5)], Number.NaN)).toBe(10);
    expect(resolveScrollableProjectSec(10, [clip(0, 5)], -5)).toBe(10);
});
