import { test } from "vitest";

import { resolveGridDrawViewport } from "./gridDrawViewport.ts";

test("components/layout/timeline/gridDrawViewport.test.ts scripted checks", async () => {
    let checks = 0;

    function assertEqual(actual: number, expected: number, label: string): void {
        checks += 1;
        if (Math.abs(actual - expected) > 1e-9) {
            throw new Error(`${label}: expected ${expected}, received ${actual}`);
        }
    }

    // ── 回归场景：量化 scrollLeft 死区（REACT_SCROLL_STEP_PX = 256px）────
    // 用户小幅滚动（净位移 < 256px）后，React 提交的 scrollLeft 永久滞后于
    // 原生 scroller；随后拖拽 Clip 改变 contentWidth 触发 React 重绘路径。
    // 网格必须画在总线快照（原生滚动真值）上，而不是滞后的 prop 上——否则
    // 帧提交器按 axisEquals 去重、对账循环发现总线与 DOM 一致，错误偏移会
    // 一直保持到下一次滚动/缩放（"拖拽 Clip 后网格偏移"）。
    const laggedPropScrollLeft = 0;
    const authoritativeScrollLeft = 200.4;
    const resolved = resolveGridDrawViewport({
        busAxis: { scrollLeftPx: authoritativeScrollLeft, scrollTopPx: 0 },
        propScrollLeftPx: laggedPropScrollLeft,
        propViewportTopPx: 0,
    });
    assertEqual(resolved.scrollLeftPx, authoritativeScrollLeft, "bus scrollLeft wins over stale prop");
    assertEqual(resolved.scrollTopPx, 0, "bus scrollTop wins over prop");

    // 滞后量可以是最 Dead 区宽度以内的任意值（0–255px），方向也可为反向
    // （回滚滚动后 prop 反而领先）：只要总线可用，一律以总线为准。
    const resolvedReverse = resolveGridDrawViewport({
        busAxis: { scrollLeftPx: 180, scrollTopPx: 64 },
        propScrollLeftPx: 300,
        propViewportTopPx: 0,
    });
    assertEqual(resolvedReverse.scrollLeftPx, 180, "bus scrollLeft wins when prop is ahead");
    assertEqual(resolvedReverse.scrollTopPx, 64, "bus scrollTopPx is authoritative (vertical)");

    // ── 无总线（参数编辑器过渡路径）：保持 props 行为不变 ────────────────
    const fallback = resolveGridDrawViewport({
        busAxis: null,
        propScrollLeftPx: 123.5,
        propViewportTopPx: 45,
    });
    assertEqual(fallback.scrollLeftPx, 123.5, "no bus falls back to prop scrollLeft");
    assertEqual(fallback.scrollTopPx, 45, "no bus falls back to prop viewportTop");

    // props 缺省/非法时归零，保证 draw() 拿到有限值。
    const fallbackDefaults = resolveGridDrawViewport({
        busAxis: undefined,
        propScrollLeftPx: undefined,
        propViewportTopPx: undefined,
    });
    assertEqual(fallbackDefaults.scrollLeftPx, 0, "missing props default scrollLeft to 0");
    assertEqual(fallbackDefaults.scrollTopPx, 0, "missing props default viewportTop to 0");

    const fallbackNonFinite = resolveGridDrawViewport({
        busAxis: null,
        propScrollLeftPx: Number.NaN,
        propViewportTopPx: Number.NaN,
    });
    assertEqual(fallbackNonFinite.scrollLeftPx, 0, "non-finite prop scrollLeft falls back to 0");
    assertEqual(fallbackNonFinite.scrollTopPx, 0, "non-finite prop viewportTop falls back to 0");

    // ── 总线快照非法（防御）：退回 props，绝不让 NaN 进入绘制偏移 ────────
    const invalidBus = resolveGridDrawViewport({
        busAxis: { scrollLeftPx: Number.NaN, scrollTopPx: Number.NaN },
        propScrollLeftPx: 42,
        propViewportTopPx: 7,
    });
    assertEqual(invalidBus.scrollLeftPx, 42, "invalid bus snapshot falls back to prop scrollLeft");
    assertEqual(invalidBus.scrollTopPx, 7, "invalid bus snapshot falls back to prop viewportTop");

    // 部分非法（仅竖直非法）也应整体退回 props：两条路径必须输出同源快照，
    // 半用总线半用 props 会重现"React 重绘与总线 paint 输出不一致"。
    const halfInvalidBus = resolveGridDrawViewport({
        busAxis: { scrollLeftPx: 100, scrollTopPx: Number.NaN },
        propScrollLeftPx: 42,
        propViewportTopPx: 7,
    });
    assertEqual(halfInvalidBus.scrollLeftPx, 42, "half-invalid bus falls back entirely to props");
    assertEqual(halfInvalidBus.scrollTopPx, 7, "half-invalid bus falls back entirely to props");

    if (checks !== 14) {
        throw new Error(`expected 14 checks, ran ${checks}`);
    }
});
