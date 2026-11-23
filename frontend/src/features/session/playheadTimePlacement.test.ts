/**
 * 播放光标时间**放置策略**的纯函数与迁移测试。
 *
 * 【为什么要单测这两件事】
 * 1. `auto` 到底落在哪儿只允许有一个答案（否则三个槽位各自理解一次，就会出现
 *    两份实例或零份实例）；
 * 2. 旧配置只有 `showPlayheadTimeInTrackHeader` 一个布尔：迁移必须让老用户看到
 *    与升级前**相同**的位置，而不是被悄悄挪到别处。
 */
import { configureStore } from "@reduxjs/toolkit";
import { describe, expect, test } from "vitest";

import sessionReducer from "./sessionSlice";
import { loadUiSettings } from "./thunks/runtimeThunks";
import {
    PLAYHEAD_TIME_PLACEMENTS,
    isPlayheadTimePlacement,
    playheadTimePlacementLabelKey,
    resolvePlayheadTimePlacement,
} from "./playheadTimePlacement";

describe("resolvePlayheadTimePlacement", () => {
    test("auto 统一解析为状态栏", () => {
        expect(resolvePlayheadTimePlacement("auto")).toBe("statusBar");
    });

    test("其余取值原样返回", () => {
        for (const placement of PLAYHEAD_TIME_PLACEMENTS) {
            if (placement === "auto") continue;
            expect(resolvePlayheadTimePlacement(placement)).toBe(placement);
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

    test("旧布尔 false ⇒ auto（升级前工具栏仍在，读数不该消失）", () => {
        expect(hydrate({ showPlayheadTimeInTrackHeader: false })).toBe("auto");
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
