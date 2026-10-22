import { test } from "vitest";

import { captureBandGeometry } from "./captureBandGeometry";

function near(actual: number, expected: number, label: string, eps = 1e-6): void {
    if (Math.abs(actual - expected) > eps) {
        throw new Error(label + ": expected " + expected + ", received " + actual);
    }
}

test("captureBandGeometry keeps the band on the ruler's own coordinates", () => {
    // 区间 2s..4s，缩放 100px/s，滚动 1s ⇒ 左缘 100、宽 200。
    const base = captureBandGeometry({
        startSec: 2,
        endSec: 4,
        pxPerSec: 100,
        scrollLeftPx: 100,
        viewportWidthPx: 800,
        dpr: 1,
    });
    near(base.left, 100, "left");
    near(base.width, 200, "width");
    if (!base.visible) throw new Error("a band inside the viewport must be visible");

    // 同一区间、滚动改成 3s ⇒ 左缘 -100（左端越界但仍有可见部分）。
    const scrolled = captureBandGeometry({
        startSec: 2,
        endSec: 4,
        pxPerSec: 100,
        scrollLeftPx: 300,
        viewportWidthPx: 800,
        dpr: 1,
    });
    near(scrolled.left, -100, "left after scrolling past the start");
    if (!scrolled.visible) throw new Error("a band straddling the left edge is still visible");

    // 完全滚出左侧 ⇒ 不可见。
    const offLeft = captureBandGeometry({
        startSec: 0,
        endSec: 1,
        pxPerSec: 100,
        scrollLeftPx: 500,
        viewportWidthPx: 800,
        dpr: 1,
    });
    if (offLeft.visible) throw new Error("a band entirely left of the viewport must be hidden");

    // 完全滚出右侧 ⇒ 不可见。
    const offRight = captureBandGeometry({
        startSec: 100,
        endSec: 101,
        pxPerSec: 100,
        scrollLeftPx: 0,
        viewportWidthPx: 800,
        dpr: 1,
    });
    if (offRight.visible) throw new Error("a band entirely right of the viewport must be hidden");

    // 零宽区间（刚开始录）仍要看得见：宽度至少一个设备像素。
    const zero = captureBandGeometry({
        startSec: 3,
        endSec: 3,
        pxPerSec: 100,
        scrollLeftPx: 0,
        viewportWidthPx: 800,
        dpr: 2,
    });
    near(zero.width, 0.5, "zero-width band must still be one device pixel wide");
    if (!zero.visible) throw new Error("a freshly started capture must be visible");

    // 两端都吸附到设备像素栅格：dpr 1.5 下不落在半像素上。
    const snapped = captureBandGeometry({
        startSec: 0.0333,
        endSec: 0.0666,
        pxPerSec: 30,
        scrollLeftPx: 0,
        viewportWidthPx: 800,
        dpr: 1.5,
    });
    near(snapped.left * 1.5, Math.round(snapped.left * 1.5), "left must sit on a device pixel");
    near(
        (snapped.left + snapped.width) * 1.5,
        Math.round((snapped.left + snapped.width) * 1.5),
        "right edge must sit on a device pixel",
        0.51,
    );
});
