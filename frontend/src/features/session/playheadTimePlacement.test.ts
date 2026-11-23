/**
 * 播放光标时间**放置策略**的纯函数与迁移测试。
 *
 * 【为什么要单测这两件事】
 * 1. `auto` 到底落在哪儿只允许有一个答案（否则三个槽位各自理解一次，就会出现
 *    两份实例或零份实例），而且这条链必须**真的会顺延**；
 * 2. 旧配置只有 `showPlayheadTimeInTrackHeader` 一个布尔：迁移必须让老用户看到
 *    与升级前**相同**的位置，而不是被悄悄挪到别处。
 */
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, test } from "vitest";

import sessionReducer from "./sessionSlice";
import { loadUiSettings } from "./thunks/runtimeThunks";
import {
    AUTO_PLACEMENT_PRIORITY,
    PLAYHEAD_TIME_PLACEMENTS,
    isPlayheadTimePlacement,
    playheadTimePlacementLabelKey,
    resolvePlayheadTimePlacement,
    type PlayheadTimeSlotAvailability,
} from "./playheadTimePlacement";

const ALL_SLOTS: PlayheadTimeSlotAvailability = {
    toolbar: true,
    statusBar: true,
    trackHeader: true,
};

describe("resolvePlayheadTimePlacement", () => {
    /**
     * 【这条为什么是回归】`auto` 曾经写死成状态栏 —— 而状态栏在主窗口里永远存在，
     * 链上第二级永远走不到，"自动"与"状态栏"于是完全等价。
     */
    test("auto 落在链首：工具栏", () => {
        expect(resolvePlayheadTimePlacement("auto")).toBe("toolbar");
        expect(resolvePlayheadTimePlacement("auto", ALL_SLOTS)).toBe("toolbar");
    });

    test("auto 按优先级顺延到下一个真的挂载着的槽位", () => {
        expect(AUTO_PLACEMENT_PRIORITY).toEqual(["toolbar", "statusBar", "trackHeader"]);
        expect(resolvePlayheadTimePlacement("auto", { ...ALL_SLOTS, toolbar: false })).toBe(
            "statusBar",
        );
        expect(
            resolvePlayheadTimePlacement("auto", {
                ...ALL_SLOTS,
                toolbar: false,
                statusBar: false,
            }),
        ).toBe("trackHeader");
        expect(
            resolvePlayheadTimePlacement("auto", {
                toolbar: false,
                statusBar: false,
                trackHeader: false,
            }),
        ).toBe("hidden");
    });

    test("显式选择原样返回，不因槽位缺失被搬去别处", () => {
        for (const placement of PLAYHEAD_TIME_PLACEMENTS) {
            if (placement === "auto") continue;
            expect(resolvePlayheadTimePlacement(placement)).toBe(placement);
            expect(
                resolvePlayheadTimePlacement(placement, {
                    toolbar: false,
                    statusBar: false,
                    trackHeader: false,
                }),
            ).toBe(placement);
        }
    });

    test("合法取值白名单与标签键一一对应", () => {
        expect(PLAYHEAD_TIME_PLACEMENTS).toEqual([
            "auto",
            "toolbar",
            "statusBar",
            "trackHeader",
            "hidden",
        ]);
        for (const placement of PLAYHEAD_TIME_PLACEMENTS) {
            expect(isPlayheadTimePlacement(placement)).toBe(true);
            expect(playheadTimePlacementLabelKey(placement)).toMatch(/^playhead_time_placement_/);
        }
        expect(isPlayheadTimePlacement("bogus")).toBe(false);
    });
});

describe("旧配置迁移", () => {
    function hydrate(settings: Record<string, unknown>) {
        const store = configureStore({ reducer: { session: sessionReducer } });
        store.dispatch(loadUiSettings.fulfilled(settings as never, "req", undefined));
        return store.getState().session.playheadTimePlacement;
    }

    test("旧布尔 true ⇒ 轨道头（与升级前一致）", () => {
        expect(hydrate({ showPlayheadTimeInTrackHeader: true })).toBe("trackHeader");
    });

    /**
     * 【升级前的行为就是"工具栏里有一份"】`auto` 现在解析到工具栏，因此这条迁移
     * 恰好把老用户留在原来的位置 —— 而不是"自动"再被解释成别处。
     */
    test("旧布尔 false ⇒ auto（解析为工具栏，与升级前一致）", () => {
        const placement = hydrate({ showPlayheadTimeInTrackHeader: false });
        expect(placement).toBe("auto");
        expect(resolvePlayheadTimePlacement(placement)).toBe("toolbar");
    });

    test("新字段合法时以它为准", () => {
        expect(
            hydrate({ playheadTimePlacement: "toolbar", showPlayheadTimeInTrackHeader: true }),
        ).toBe("toolbar");
    });

    test("新字段非法 ⇒ 回退到旧布尔迁移，而不是把读数藏起来", () => {
        expect(
            hydrate({ playheadTimePlacement: "somewhere", showPlayheadTimeInTrackHeader: true }),
        ).toBe("trackHeader");
    });
});
