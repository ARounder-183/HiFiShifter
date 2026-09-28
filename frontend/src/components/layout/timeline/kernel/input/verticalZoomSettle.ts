/**
 * 时间轴内核 · 竖直缩放的「在途结算」判定
 *
 * 【主要问题】画布竖直缩放由内核**当帧**原子提交（行高 + 锚点 scrollTop），而 React
 * 侧行高要等下一次提交才落地。若在行高落地之前把新 scrollTop 同步镜像进轨道头 DOM，
 * 容器还是**旧行高**的内容高：新位置一旦超过旧上限就被浏览器钳制，且该钳制 / 滚动
 * 锚定产生的 scroll 事件与"上次写入值"不符，会被 `isMirrorEcho` 判成用户输入反灌
 * 内核，把视口从锚点位置拽回。每格缩放"先跳到位再被拽回"，即用户报告的竖直抽动。
 *
 * 【处置】在途期间跳过轨道头镜像写入，等下列两个条件之一满足后补写一次：
 * 1. React 行高落地（镜像到的行高等于本次缩放请求的行高）；或
 * 2. 兜底超时（宿主脱离 React 的测试环境、或行高被外部改成别的值时防止永久跳过）。
 *
 * 【为什么单独成模块】判定是纯计算，抽出来才能单测 —— 本工程 Vitest 跑在 node 环境
 * （无 jsdom / WebGL），内核宿主本体无法在测试里构造。与 `scrollEcho`、`scrollCommit`
 * 同一模式。
 */

/** 一次竖直缩放的"在途"记录。 */
export interface VerticalZoomFlight {
    /** 本次缩放请求的行高（内核已钳制、取整后的值）。 */
    readonly rowHeight: number;
    /** 提交时刻（`performance.now()` 量纲，毫秒）。 */
    readonly startedAt: number;
}

/** 创建一条在途记录。 */
export function createVerticalZoomFlight(rowHeight: number, nowMs: number): VerticalZoomFlight {
    return { rowHeight, startedAt: nowMs };
}

/** 结算判定入参。 */
export interface ShouldSettleVerticalZoomArgs {
    /** 在途记录。 */
    readonly flight: VerticalZoomFlight;
    /** 当前 React 侧镜像到的行高（= 行高是否已落地）。 */
    readonly landedRowHeight: number;
    /** 当前时刻（与 `flight.startedAt` 同量纲）。 */
    readonly nowMs: number;
    /** 兜底超时（毫秒）。 */
    readonly timeoutMs: number;
}

/**
 * 是否应当结算在途的竖直缩放。
 *
 * 行高落地用**近似相等**（1e-6）：行高在内核与 React 两侧都经过取整，浮点值应逐位
 * 相同，容差只为防御未来引入的中间计算。
 *
 * @param args 见 {@link ShouldSettleVerticalZoomArgs}。
 * @returns 满足"行高已落地"或"已超时"时为 true。
 */
export function shouldSettleVerticalZoom(args: ShouldSettleVerticalZoomArgs): boolean {
    const landed = Math.abs(args.landedRowHeight - args.flight.rowHeight) < 1e-6;
    const timedOut = args.nowMs - args.flight.startedAt >= args.timeoutMs;
    return landed || timedOut;
}
