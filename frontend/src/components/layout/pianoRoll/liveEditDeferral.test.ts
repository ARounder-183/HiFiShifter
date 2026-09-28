/**
 * 笔画期间"被推迟的刷新"台账的契约回归。
 *
 * 【为什么单测它】这份台账是"笔画期间不换 `paramView`"这条修复的**唯一收口**
 * （见 liveEditDeferral.ts 的文件头）。两个失败模式都很隐蔽、且都要等用户真的
 * 画一笔并撞上取数回包才会暴露：
 *
 * - `defer` 在非笔画期间也返回 true → 正常的滚动 / 缩放取数被静默吞掉，
 *   曲线**永不刷新**；
 * - `take` 没有清空、或丢了 `force` → 要么重复取数，要么补触发走"视口已覆盖"
 *   短路而**停在旧数据上**。
 *
 * 两者都不会报错，只能靠单测钉住。
 */
import { describe, expect, it } from "vitest";

import { createLiveEditDeferral } from "./liveEditDeferral";

/** 可手动切换的活跃判定。 */
function activity() {
    let active = false;
    return {
        isActive: () => active,
        set(next: boolean) {
            active = next;
        },
    };
}

describe("createLiveEditDeferral", () => {
    it("不在笔画中：defer 返回 false（调用方照常执行），不留待办", () => {
        const live = activity();
        const deferral = createLiveEditDeferral(live.isActive);

        expect(deferral.defer()).toBe(false);
        expect(deferral.defer({ force: true })).toBe(false);
        expect(deferral.take()).toBeNull();
    });

    it("笔画中：defer 返回 true 并留下对应类别的待办", () => {
        const live = activity();
        const deferral = createLiveEditDeferral(live.isActive);

        live.set(true);
        expect(deferral.defer()).toBe(true);
        expect(deferral.take()).toEqual({ force: false, plain: true });
    });

    it("★ force 与 plain 分别记账，且并集一次取走", () => {
        const live = activity();
        const deferral = createLiveEditDeferral(live.isActive);
        live.set(true);

        deferral.defer({ force: true }); // 分析完成
        deferral.defer(); // 滚动
        expect(deferral.take()).toEqual({ force: true, plain: true });
    });

    it("★ take 之后台账清零：同一批待办不会被补触发两次", () => {
        const live = activity();
        const deferral = createLiveEditDeferral(live.isActive);
        live.set(true);

        deferral.defer({ force: true });
        expect(deferral.take()).toEqual({ force: true, plain: false });
        expect(deferral.take()).toBeNull();
    });

    it("多笔连画：每一笔的待办各自独立取走", () => {
        const live = activity();
        const deferral = createLiveEditDeferral(live.isActive);

        live.set(true);
        deferral.defer({ force: true });
        expect(deferral.take()).toEqual({ force: true, plain: false });

        // 第二笔：只发生视口变化。
        deferral.defer();
        expect(deferral.take()).toEqual({ force: false, plain: true });
    });

    it("活跃判定每次 defer 时重读（不缓存笔画状态）", () => {
        const live = activity();
        const deferral = createLiveEditDeferral(live.isActive);

        live.set(true);
        expect(deferral.defer()).toBe(true);
        live.set(false);
        expect(deferral.defer()).toBe(false);
        // 笔画结束后的那次 defer 不产生新的待办类别。
        expect(deferral.take()).toEqual({ force: false, plain: true });
    });
});
