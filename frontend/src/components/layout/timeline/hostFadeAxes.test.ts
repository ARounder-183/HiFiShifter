// 宿主淡化轴的预设映射：与 Rust 侧 `fade_axes.rs` 钉住同一组实测值。
import { expect, test } from "vitest";
import { HOST_FADE_PRESET_AXES, hostFadePresetAxes, hostFadePresetForAxes } from "./hostFadeAxes";

/**
 * 黄金值来自 `probe/ara/captures/fade-axis-7.82.json` 的 `legacy_shape_only` 用例
 * （REAPER 7.82/x64：只写 `C_FADEINSHAPE = k`，读回两个新轴）。
 *
 * Rust 侧 `fade_axes.rs::preset_table_matches_the_capture` 断言同一张表。两处一起改，
 * 或一起不改 —— 单改一处会让插件写出的形状与界面认出的形状分家。
 */
test("golden values match the rust fade-axis table", () => {
    expect(HOST_FADE_PRESET_AXES).toEqual([
        [0, 0],
        [0.5, 0],
        [-0.5, 0],
        [1, 0],
        [-1, 0],
        [0, 0.5],
        [0, 1],
    ]);
});

/** 两轴正交是"新轴能完整表达七个预设"的全部理由，单独钉一条。 */
test("presets split cleanly across the two axes", () => {
    HOST_FADE_PRESET_AXES.forEach(([curvature, s], index) => {
        if (index <= 4) expect(s).toBe(0);
        else expect(curvature).toBe(0);
    });
});

/** 实测：宿主新建 item 的默认读数就是预设 1（`SHAPE=1` / `c=0.5` / `S=0`）。 */
test("the host default reading is preset one", () => {
    expect(hostFadePresetAxes(1)).toEqual([0.5, 0]);
    expect(hostFadePresetForAxes(0.5, 0)).toBe(1);
});

test("round trips every preset", () => {
    HOST_FADE_PRESET_AXES.forEach(([curvature, s], index) => {
        expect(hostFadePresetAxes(index)).toEqual([curvature, s]);
        expect(hostFadePresetForAxes(curvature, s)).toBe(index);
    });
});

/** 表外的形状号一律拒绝：宿主对越界值不报错，而是静默变成别的形状（实测 `5.1` → `SHAPE=7`）。 */
test("rejects shapes outside the table", () => {
    for (const shape of [-1, 7, 1.1, 5.1, 6.5, Number.NaN, Infinity])
        expect(hostFadePresetAxes(shape)).toBeNull();
});

/** 不在表内的坐标是常态（用户拖过滑杆），返回 `null` 而不是最近邻。 */
test("axes outside the table are not an error and never snap to a neighbour", () => {
    for (const [curvature, s] of [
        [0.25, 0],
        [0, -1],
        [-0.5, 0.5],
        [0.75, 0],
        [0.5, 0.0001],
    ])
        expect(hostFadePresetForAxes(curvature, s)).toBeNull();
    expect(hostFadePresetForAxes(Number.NaN, 0)).toBeNull();
    expect(hostFadePresetForAxes(0, Infinity)).toBeNull();
});

/**
 * 宿主自己的 `C_FADEINSHAPE` 是**多对一粗分类**（实测 `(0.5, -1)` 与 `(-1, -1)` 都读成
 * `SHAPE=5`）。本函数刻意不照抄那个分类 —— 否则用户自己拖出来的曲线会被认领成预设。
 */
test("matching stays exact rather than copying the host's coarse classification", () => {
    expect(hostFadePresetForAxes(0.5, -1)).toBeNull();
    expect(hostFadePresetForAxes(-1, -1)).toBeNull();
});
