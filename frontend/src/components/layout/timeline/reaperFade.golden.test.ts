// 曲线数学的跨语言黄金绑定：本文件读 kernel `fade_curves` 生成的黄金向量
// （`backend/hifishifter-kernel/tests/fixtures/fade-gain-golden.json`），断言 TS
// 镜像公式（`fadeGainSigned` / `fadeGainAxes`）与 Rust 实现逐值一致。
//
// 【为什么读文件而不是内联字面量】内联的话测试退化成"常数等于自己"，改公式顺手
// 改字面量就能变绿。同一份 JSON 同时被 Rust 侧
// `fade_gain_matches_the_committed_golden_vectors` 断言 —— 任一侧与证据不符都会红，
// 两侧一起改或一起不改。重新生成见该 JSON 的 comment 字段。
import { readFileSync } from "node:fs";
import { expect, test } from "vitest";
import { fadeGainAxes, fadeGainSigned } from "./reaperFade";

const GOLDEN_URL = new URL(
    "../../../../../backend/hifishifter-kernel/tests/fixtures/fade-gain-golden.json",
    import.meta.url,
);

interface GoldenVector {
    shapeInt?: number;
    shape?: number;
    dir?: number;
    curvature?: number;
    s?: number;
    modeOut: boolean;
    t: number;
    gain: number;
}

interface GoldenDoc {
    comment: string;
    signed: GoldenVector[];
    axes: GoldenVector[];
}

const doc: GoldenDoc = JSON.parse(readFileSync(GOLDEN_URL, "utf8")) as GoldenDoc;

test("TS fadeGainSigned matches the kernel golden vectors", () => {
    expect(doc.signed.length).toBeGreaterThan(0);
    for (const vector of doc.signed) {
        const shape = vector.shapeInt ?? vector.shape ?? Number.NaN;
        const gain = fadeGainSigned(
            shape,
            vector.dir as number,
            vector.modeOut ? "out" : "in",
            vector.t,
        );
        expect(Math.abs(gain - vector.gain)).toBeLessThan(1e-9);
    }
});

test("TS fadeGainAxes matches the kernel golden vectors", () => {
    expect(doc.axes.length).toBeGreaterThan(0);
    for (const vector of doc.axes) {
        const gain = fadeGainAxes(
            vector.curvature as number,
            vector.s as number,
            vector.modeOut ? "out" : "in",
            vector.t,
        );
        expect(Math.abs(gain - vector.gain)).toBeLessThan(1e-9);
    }
});
