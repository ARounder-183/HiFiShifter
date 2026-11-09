import { timeToFrame } from "./utils";

/**
 * 参数曲线取数窗口：保证请求范围始终覆盖当前可见区。
 *
 * 【为什么需要它】旧实现把 frameCount 固定钳到 200_000（5ms 帧周期 = 1000s），
 * 而请求窗口按“可见区 + 两侧各一屏”计算。宽视口 / 最小缩放时 3×visibleDur
 * 会超过 1000s，窗口右端落在可见区右端之前，后段曲线没有数据。初始滚动时
 * 窗口从 0 开始还能盖住；滚远之后就会看到“后边的参数线消失”。
 *
 * 【预算策略】不设固定的 200k 下限：普通视口仍按 3× 可见区取数，保持旧行为
 * 与 IPC 体积；只有可见区本身接近 / 超过 1M 帧时才把预算抬到可见区大小。
 * 因而“窗口覆盖可见区”永远优先于“窗口大小上限”。
 *
 * 【不变量】返回窗口必须满足：
 *   startFrame <= visibleStartFrame 且 startFrame + frameCount > visibleEndFrame。
 */
const MAX_FRAME_BUDGET = 1_000_000;

export interface ParamFetchWindow {
    readonly startFrame: number;
    readonly frameCount: number;
}

export function resolveParamFetchWindow(args: {
    readonly visibleStartSec: number;
    readonly visibleEndSec: number;
    readonly framePeriodMs: number;
}): ParamFetchWindow {
    const fp = Math.max(1e-6, Number.isFinite(args.framePeriodMs) ? args.framePeriodMs : 5);
    const visibleStartSec = Number.isFinite(args.visibleStartSec)
        ? Math.max(0, args.visibleStartSec)
        : 0;
    const visibleEndSec = Number.isFinite(args.visibleEndSec)
        ? Math.max(visibleStartSec, args.visibleEndSec)
        : visibleStartSec;

    const visibleStartFrame = Math.max(0, timeToFrame(visibleStartSec, fp));
    const visibleEndFrame = Math.max(visibleStartFrame, timeToFrame(visibleEndSec, fp));
    const visibleFrames = visibleEndFrame - visibleStartFrame + 1;

    // 3 倍可见区：中间是可见区，两侧各留一屏余量，供滚动时复用。
    const desiredFrames = visibleFrames * 3;
    // 预算上限只在“可见区本身装得下”时生效；可见区超过 1M 帧时，
    // 必须让 frameCount 跟着可见区长大，否则覆盖不变量不成立。
    const frameCount = Math.max(
        visibleFrames,
        Math.min(desiredFrames, Math.max(MAX_FRAME_BUDGET, visibleFrames)),
    );

    // 把余量均分到两侧；start 被 0 钳住时余量全部落在右侧，仍然覆盖可见区。
    const marginBefore = Math.floor((frameCount - visibleFrames) / 2);
    const startFrame = Math.max(0, visibleStartFrame - marginBefore);

    return { startFrame, frameCount };
}
