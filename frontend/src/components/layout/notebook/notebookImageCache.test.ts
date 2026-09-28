/*
 * 图片解析缓存的回归测试。
 *
 * 【为什么必须有这个文件】这里曾经有一个让整个窗口白屏的缺陷：解析失败时
 * 也在 `finally` 里通知订阅者，而订阅者（图片 NodeView）收到通知就再解析一次
 * → 再失败 → 再通知 …… 对同步返回失败的分支（远程图被禁用、未落盘工程里的
 * 相对路径）是同步递归，直接从 `useEffect` 里抛 `RangeError` 打掉整个 React
 * 树；对要走 IPC 的分支（附件缺失）则是无界的 IPC + setState 风暴。
 *
 * 下面几条用例锁定的就是"通知只在缓存被作废时发出，绝不在解析结束时发出"。
 */

import { beforeEach, expect, test, vi } from "vitest";

const readAsset = vi.fn();
const readFileBase64 = vi.fn();

vi.mock("../../../services/api/notebook", () => ({
    notebookApi: {
        readAsset: (...args: unknown[]) => readAsset(...args),
        readFileBase64: (...args: unknown[]) => readFileBase64(...args),
    },
}));

import {
    clearImageCache,
    peekImageUrl,
    resolveImage,
    subscribeAssetInvalidation,
} from "./notebookImageCache.ts";

/** jsdom/node 都没有 createObjectURL；测试里只需要一个稳定字符串。 */
let objectUrlSeq = 0;
beforeEach(() => {
    readAsset.mockReset();
    readFileBase64.mockReset();
    clearImageCache();
    objectUrlSeq = 0;
    (URL as unknown as { createObjectURL: (blob: Blob) => string }).createObjectURL = () => {
        objectUrlSeq += 1;
        return `blob:stub-${objectUrlSeq}`;
    };
    (URL as unknown as { revokeObjectURL: (url: string) => void }).revokeObjectURL = () => {};
});

test("components/layout/notebook/notebookImageCache.test.ts scripted checks", async () => {
    // ── 1. 失败的解析不通知订阅者（否则就是无界递归）───────────────
    readAsset.mockResolvedValue({ ok: false, error: "notebook_asset_not_found", missing: true });

    let invalidations = 0;
    let resolveAttempts = 0;
    const unsubscribe = subscribeAssetInvalidation(() => {
        invalidations += 1;
        // 模拟 NodeView 的真实行为：收到作废通知就重新解析一次。
        resolveAttempts += 1;
        void resolveImage("hifi-asset://missing-id.webp", {
            projectDir: null,
            allowRemoteImages: true,
        });
    });

    const result = await resolveImage("hifi-asset://missing-id.webp", {
        projectDir: null,
        allowRemoteImages: true,
    });
    expect(result.missing).toBe(true);
    expect(result.url).toBe(null);
    // 解析失败不是"缓存作废"：一次通知都不该发。
    expect(invalidations).toBe(0);
    expect(resolveAttempts).toBe(0);
    unsubscribe();

    // ── 2. 同步失败分支（未落盘工程的相对路径）不递归 ──────────────
    // 这条分支里没有任何 await，一旦在结束时通知就会同步递归到爆栈。
    let syncInvalidations = 0;
    const unsubscribeSync = subscribeAssetInvalidation(() => {
        syncInvalidations += 1;
        void resolveImage("素材/截图.png", { projectDir: null, allowRemoteImages: true });
    });
    const relative = await resolveImage("素材/截图.png", {
        projectDir: null,
        allowRemoteImages: true,
    });
    expect(relative.missing).toBe(true);
    expect(syncInvalidations).toBe(0);
    unsubscribeSync();

    // ── 3. 远程图被禁用同样是失败分支 ─────────────────────────────
    let remoteInvalidations = 0;
    const unsubscribeRemote = subscribeAssetInvalidation(() => {
        remoteInvalidations += 1;
        void resolveImage("https://example.com/a.png", {
            projectDir: null,
            allowRemoteImages: false,
        });
    });
    const blocked = await resolveImage("https://example.com/a.png", {
        projectDir: null,
        allowRemoteImages: false,
    });
    expect(blocked.url).toBe(null);
    expect(blocked.reason).toBe("remote-blocked");
    expect(remoteInvalidations).toBe(0);
    unsubscribeRemote();

    // ── 4. 成功解析也不通知（调用方直接从 promise 拿 URL）──────────
    readAsset.mockResolvedValue({
        ok: true,
        mime: "image/png",
        base64: btoa("fake-bytes"),
    });
    let successInvalidations = 0;
    const unsubscribeSuccess = subscribeAssetInvalidation(() => {
        successInvalidations += 1;
    });
    const ok = await resolveImage("hifi-asset://present-id.png", {
        projectDir: null,
        allowRemoteImages: true,
    });
    expect(ok.missing).toBe(false);
    expect(ok.url).toBe("blob:stub-1");
    expect(successInvalidations).toBe(0);
    unsubscribeSuccess();

    // 命中缓存：不再触发后端读取。
    const cachedCalls = readAsset.mock.calls.length;
    const again = await resolveImage("hifi-asset://present-id.png", {
        projectDir: null,
        allowRemoteImages: true,
    });
    expect(again.url).toBe("blob:stub-1");
    expect(readAsset.mock.calls.length).toBe(cachedCalls);

    // ── 5. 并发去重：同一 src 的并发请求只读一次 ───────────────────
    readAsset.mockClear();
    readAsset.mockResolvedValue({ ok: true, mime: "image/png", base64: btoa("x") });
    const [a, b] = await Promise.all([
        resolveImage("hifi-asset://concurrent.png", { projectDir: null, allowRemoteImages: true }),
        resolveImage("hifi-asset://concurrent.png", { projectDir: null, allowRemoteImages: true }),
    ]);
    expect(a.url).toBe(b.url);
    expect(readAsset.mock.calls.length).toBe(1);

    // ── 6. 整体清空**要**通知（订阅者据此重新解析）────────────────
    let clearNotifications = 0;
    const unsubscribeClear = subscribeAssetInvalidation(() => {
        clearNotifications += 1;
    });
    clearImageCache();
    expect(clearNotifications).toBe(1);
    // 清空后缓存确实失效。
    expect(peekImageUrl("hifi-asset://present-id.png", null)).toBe(null);
    unsubscribeClear();

    // ── 7. 退订后不再收到通知 ─────────────────────────────────────
    let afterUnsubscribe = 0;
    const unsubscribeAfter = subscribeAssetInvalidation(() => {
        afterUnsubscribe += 1;
    });
    unsubscribeAfter();
    clearImageCache();
    expect(afterUnsubscribe).toBe(0);

    // ── 8. 清空 + 失败的订阅者：不会形成循环 ──────────────────────
    // 这是崩溃场景的完整复现：一次作废 → 订阅者重新解析 → 解析失败 → 到此为止。
    readAsset.mockResolvedValue({ ok: false, missing: true });
    let loops = 0;
    const unsubscribeLoop = subscribeAssetInvalidation(() => {
        loops += 1;
        void resolveImage("hifi-asset://still-missing.png", {
            projectDir: null,
            allowRemoteImages: true,
        });
    });
    clearImageCache();
    await Promise.resolve();
    await Promise.resolve();
    expect(loops).toBe(1);
    unsubscribeLoop();
});
