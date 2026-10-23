/**
 * 捕获区间列表的派生契约。
 *
 * 【要钉死什么】用户在捕获过程中可以把播放光标跳到任意位置再继续录。此前的实现把
 * 后端报的**并集**（首段起点 .. 末段终点）当成一条带子，于是"录了 0..3 秒、跳到
 * 30 秒、又录了 30..33 秒"在界面上是一条 0..33 秒的连续带 —— 中间 27 秒其实什么都
 * 没录到。这个文件把"必须逐段"钉成可执行的事实。
 */
import { expect, test } from "vitest";

import { deriveCaptureBands } from "./captureRange";
import type { CaptureStatus } from "./captureApi";

/**
 * 一个最小的状态载荷（只填本函数关心的字段）。
 *
 * 走 `unknown` 是有意的：派生函数只读 `capture` 与 `sampleRate`，把载荷的其余字段
 * 全部填满只会让这份夹具变成第二个 `CaptureStatus` 定义，改一处就要改两处。
 */
function status(capture: Partial<NonNullable<CaptureStatus["capture"]>> | null): CaptureStatus {
    return {
        state: capture === null ? "standby" : "recording",
        standby: capture === null,
        recording: null,
        capture:
            capture === null
                ? null
                : {
                      startFrame: 0,
                      endFrame: 0,
                      positionFrame: 0,
                      sampleRate: 48_000,
                      runs: [],
                      ...capture,
                  },
        material: null,
        hasMaterial: false,
        span: null,
        takes: [],
        missingTakes: [],
        options: {
            maxDurationSec: 60,
            segmentSec: 8,
            monitor: false,
            outsidePassthrough: false,
        },
        takeCount: 0,
        activeTake: null,
        sawContext: true,
        sawInput: true,
        sawOffline: false,
        canCapture: true,
        hostPlaying: true,
        unsupportedPrecision: false,
        limitWarning: false,
        araBound: false,
        araSupported: false,
        backendError: null,
    } as unknown as CaptureStatus;
}

test("没有在捕获时返回 null", () => {
    expect(deriveCaptureBands(status(null))).toBeNull();
    expect(deriveCaptureBands(null)).toBeNull();
});

test("跳播产生的两段必须画成两条带，而不是被并成一条", () => {
    const derived = deriveCaptureBands(
        status({
            runs: [
                { startFrame: 0, endFrame: 144_000 },
                { startFrame: 1_440_000, endFrame: 1_584_000 },
            ],
            positionFrame: 1_584_000,
        }),
    );
    expect(derived?.bands).toEqual([
        { startSec: 0, endSec: 3 },
        { startSec: 30, endSec: 33 },
    ]);
    expect(derived?.positionSec).toBe(33);
});

test("当前位置只延长**最后一段**，历史段不得被拉到当前位置", () => {
    // 若把 max(endFrame, positionFrame) 应用到每一段，第一段会变成 0..25 —— 那正是
    // 本项最容易写错的一行，也正是"又并成一整条"的成因。
    const derived = deriveCaptureBands(
        status({
            runs: [
                { startFrame: 0, endFrame: 48_000 },
                { startFrame: 1_152_000, endFrame: 1_200_000 },
            ],
            positionFrame: 1_200_000,
        }),
    );
    expect(derived?.bands[0]).toEqual({ startSec: 0, endSec: 1 });
    expect(derived?.bands[1]).toEqual({ startSec: 24, endSec: 25 });
});

test("旧载荷（没有 runs）退回并集那一段", () => {
    const derived = deriveCaptureBands(
        status({ startFrame: 0, endFrame: 96_000, positionFrame: 96_000, runs: [] }),
    );
    expect(derived?.bands).toEqual([{ startSec: 0, endSec: 2 }]);
});

test("零长度区间被丢弃；全空时返回 null（没有可画的东西）", () => {
    expect(
        deriveCaptureBands(status({ runs: [{ startFrame: 5, endFrame: 5 }], positionFrame: 5 })),
    ).toBeNull();
});

test("采样率不可用时返回 null（不猜一个率去画）", () => {
    expect(
        deriveCaptureBands(
            status({
                sampleRate: 0,
                runs: [{ startFrame: 0, endFrame: 48_000 }],
                positionFrame: 48_000,
            }),
        ),
    ).toBeNull();
});
