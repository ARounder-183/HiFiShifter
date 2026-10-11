// 宿主淡化轴的预设映射：与 Rust 侧 `fade_axes.rs` 钉住同一组实测值。
import { readFileSync } from "node:fs";
import { expect, test } from "vitest";
import { HOST_FADE_PRESET_AXES, hostFadePresetAxes, hostFadePresetForAxes } from "./hostFadeAxes";

/**
 * 实测证据（提交进仓）：`backend/hifishifter-kernel/tests/fixtures/fade-axis-7.82.json`
 * 的 `legacy_shape_only` 用例（REAPER 7.82/x64：只写 `C_FADEINSHAPE = k`，读回两个新轴）。
 *
 * 【为什么不内联字面量】内联的话这条测试就退化成"常数等于自己"：改表时顺手改字面量
 * 就能变绿，证据与实现各走各的。这里读**原始数据**，让 TS 表、Rust 表
 * （`fade_axes.rs::preset_table_matches_the_committed_capture`）与同一份证据绑在一起 ——
 * 任一侧与证据不符都会红。两处一起改，或一起不改。
 */
const CAPTURE_URL = new URL(
    "../../../../../backend/hifishifter-kernel/tests/fixtures/fade-axis-7.82.json",
    import.meta.url,
);

interface FadeAxisCapture {
    baseline: Record<string, number>;
    cases: { case: string; samples: { written: string; after: Record<string, number> }[] }[];
}

function readCapture(): FadeAxisCapture {
    return JSON.parse(readFileSync(CAPTURE_URL, "utf8")) as FadeAxisCapture;
}

/** 从实测数据还原 `[curvature, S]` 表（只取整数预设 0..=6，跳过 1.1 / 5.1 小数变体）。 */
function measuredPresetAxes(): [number, number][] {
    const legacy = readCapture().cases.find((entry) => entry.case === "legacy_shape_only");
    if (!legacy) throw new Error("fade-axis capture is missing the legacy_shape_only case");
    return legacy.samples
        .map((sample) => ({ k: Number(sample.written), after: sample.after }))
        .filter(({ k }) => Number.isInteger(k) && k >= 0 && k <= 6)
        .sort((a, b) => a.k - b.k)
        .map(({ after }) => [after.D_FADEINDIR_NEW, after.D_FADEINDIR2_NEW] as [number, number]);
}

test("golden values match the committed fade-axis capture", () => {
    const measured = measuredPresetAxes();
    expect(measured).toHaveLength(HOST_FADE_PRESET_AXES.length);
    expect(HOST_FADE_PRESET_AXES.map(([curvature, s]) => [curvature, s])).toEqual(measured);
});

/** 交叉验证：实测 baseline（全新 item）的读数就是预设 1（`SHAPE=1` / `c=0.5` / `S=0`）。 */
test("the captured baseline is preset one", () => {
    const baseline = readCapture().baseline;
    expect(hostFadePresetForAxes(baseline.D_FADEINDIR_NEW, baseline.D_FADEINDIR2_NEW)).toBe(1);
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
