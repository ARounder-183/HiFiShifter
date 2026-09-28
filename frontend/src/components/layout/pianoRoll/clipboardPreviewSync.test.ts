/*
 * 剪贴板预览「对齐槽位」策略的测试。
 *
 * 【为什么要有】从记事本暂存块恢复一份**参数线**载荷后，预览必须立刻显示会贴成
 * 什么样（预览画的正是"粘贴会落下的数据"）；此前收到 `hifi:clipboardReplaced`
 * 只清空缓存，槽位里躺着的可粘贴数据因此永远画不出来。
 *
 * 这里锁定四条：读到就采用、读不到就清空、过期结果不得覆盖新结果、取消后不再落地。
 * 后两条是纯竞态规则 —— 面板组件无法单测，所以这段逻辑必须留在可测的模块里。
 */

import { expect, test } from "vitest";

import { createClipboardPreviewSync } from "./clipboardPreviewSync";
import type { ParamClipboardData } from "./paramClipboardMapping";

function clip(param: string, values: number[]): ParamClipboardData {
    return { param, framePeriodMs: 20, segments: [{ startFrame: 0, values }] };
}

/** 手动控制的读取器：每次调用返回一个可自行 resolve 的 promise。 */
function deferredReader() {
    const pending: Array<(data: ParamClipboardData | null) => void> = [];
    const read = () =>
        new Promise<ParamClipboardData | null>((resolve) => {
            pending.push(resolve);
        });
    return { read, pending };
}

test("槽位里是参数线载荷时采用它（恢复后预览能显示）", async () => {
    const data = clip("pitch", [1, 2, 3]);
    const sync = createClipboardPreviewSync(async () => data);
    const applied: Array<ParamClipboardData | null> = [];

    await sync.sync((next) => applied.push(next));

    expect(applied).toEqual([data]);
});

test("槽位里没有参数线数据时清空预览，不残留旧曲线", async () => {
    const sync = createClipboardPreviewSync(async () => null);
    const applied: Array<ParamClipboardData | null> = [];

    await sync.sync((next) => applied.push(next));

    expect(applied).toEqual([null]);
});

test("读取失败等价于没有参数线数据：清空而不是留着旧曲线", async () => {
    const sync = createClipboardPreviewSync(async () => {
        throw new Error("clipboard unavailable");
    });
    const applied: Array<ParamClipboardData | null> = [];

    await sync.sync((next) => applied.push(next));

    expect(applied).toEqual([null]);
});

test("槽位被连续替换时，过期（先发起、后返回）的结果不得覆盖最后一次", async () => {
    const { read, pending } = deferredReader();
    const sync = createClipboardPreviewSync(read);
    const applied: Array<ParamClipboardData | null> = [];

    const stale = clip("pitch", [1]);
    const fresh = clip("formant", [9]);

    const first = sync.sync((next) => applied.push(next));
    const second = sync.sync((next) => applied.push(next));

    // 后发起的一次先返回：这就是"槽位被替换了两次"的真实时序。
    pending[1](fresh);
    await second;
    // 先发起的一次后返回：内容已经过期，必须丢弃。
    pending[0](stale);
    await first;

    expect(applied).toEqual([fresh]);
});

test("取消后到达的结果不再落地（面板卸载）", async () => {
    const { read, pending } = deferredReader();
    const sync = createClipboardPreviewSync(read);
    const applied: Array<ParamClipboardData | null> = [];

    const inflight = sync.sync((next) => applied.push(next));
    sync.cancel();
    pending[0](clip("pitch", [1]));
    await inflight;

    expect(applied).toEqual([]);
});
