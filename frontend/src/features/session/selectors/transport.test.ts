import { describe, expect, test } from "vitest";

import { isTransportAdvancing } from "./transport";

describe("isTransportAdvancing", () => {
    test("停止命令 pending 后即使旧的 positionAdvancing 还在，也不得继续外推", () => {
        expect(
            isTransportAdvancing({
                isPlaying: false,
                positionAdvancing: true,
                positionAgeMs: 0,
            }),
        ).toBe(false);
    });

    test("宿主在播且位置真的前进才外推", () => {
        expect(
            isTransportAdvancing({
                isPlaying: true,
                positionAdvancing: true,
                positionAgeMs: 0,
            }),
        ).toBe(true);
    });

    test("位置没有前进时不外推（FL Studio 停播仍报 kPlaying 的场景）", () => {
        expect(
            isTransportAdvancing({
                isPlaying: true,
                positionAdvancing: false,
                positionAgeMs: 0,
            }),
        ).toBe(false);
    });

    test("读数过期时不外推", () => {
        expect(
            isTransportAdvancing({
                isPlaying: true,
                positionAdvancing: true,
                positionAgeMs: 300,
            }),
        ).toBe(false);
    });
});
