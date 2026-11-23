import { expect, test } from "vitest";

import { fadeRestorePatch } from "../../../features/session/fadeAxesPreview";

test("单侧与交叉点取消都用同一份完整 base 构造 shape/dir/S 还原", () => {
    const base = {
        startSec: 1,
        lengthSec: 2,
        sourceStartSec: 0,
        sourceEndSec: 2,
        fadeInSec: 0.1,
        fadeOutSec: 0.2,
        autoFadeInSec: 0,
        autoFadeOutSec: 0,
        fadeInShape: 1,
        fadeOutShape: 6,
        fadeInDir: 0.37,
        fadeOutDir: -0.62,
        fadeInS: 0.12,
        fadeOutS: 0.88,
    };
    expect(fadeRestorePatch({ base, continuous: true })).toEqual({
        fadeInShape: 1,
        fadeOutShape: 6,
        fadeInDir: 0.37,
        fadeOutDir: -0.62,
        fadeInS: 0.12,
        fadeOutS: 0.88,
    });
    // 旧轴宿主绝不能携带 S：后端 validate_fade_axes 会整条拒绝。
    expect(fadeRestorePatch({ base, continuous: false })).not.toHaveProperty("fadeInS");
    expect(fadeRestorePatch({ base, continuous: false })).not.toHaveProperty("fadeOutS");
});
