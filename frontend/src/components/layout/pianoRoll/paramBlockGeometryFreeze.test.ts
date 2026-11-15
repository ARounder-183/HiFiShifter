import { describe, expect, it } from "vitest";

import { resolveParamBlockGeometryClips } from "./paramBlockGeometryFreeze";

describe("resolveParamBlockGeometryClips", () => {
    it("没有手势时用实时 clips", () => {
        const live = [{ id: "a" }, { id: "b" }];
        expect(resolveParamBlockGeometryClips(live, null)).toBe(live);
    });

    /*
     * 【引用必须稳定】块视图是 `(clips, editSet, framePeriodMs)` 的 memo；手势期间
     * 每一帧都返回同一份快照，块才不会跟着乐观更新的 clip 走。若这里返回新数组，
     * memo 每帧失效，冻结就白做了。
     */
    it("手势期间恒等于同一份快照（引用稳定）", () => {
        const live = [{ id: "a", startSec: 5 }];
        const frozen = [{ id: "a", startSec: 0 }];
        expect(resolveParamBlockGeometryClips(live, frozen)).toBe(frozen);
        expect(resolveParamBlockGeometryClips(live, frozen)).toBe(
            resolveParamBlockGeometryClips(live, frozen),
        );
    });

    it("空快照也是有效快照（手势开始时没有任何 clip）", () => {
        const live = [{ id: "a" }];
        const frozen: { id: string }[] = [];
        expect(resolveParamBlockGeometryClips(live, frozen)).toBe(frozen);
    });
});
