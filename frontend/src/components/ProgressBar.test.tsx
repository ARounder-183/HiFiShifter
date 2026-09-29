// @vitest-environment jsdom
/*
 * `ProgressBar` 的契约测试。
 *
 * 【为什么需要】它是全仓唯一的进度条组件（当前唯一消费者是导出对话框），而
 * "进度条看起来不动"曾被两度误诊。真正的根因在**颜色令牌的求值作用域**（见
 * `src/index.css` 的 `.radix-themes` 规则，以及 `src/ui/designSystemGates.test.ts`
 * 里针对它的门禁）；组件这一层能锁定的是"出问题时怎么快速判断"以及无障碍语义：
 *
 *   - `role="progressbar"` + aria 值：读屏软件与自动化唯一的取值入口；
 *   - 填充元素的 `data-hs-progress-fill` / `data-percentage`：现场诊断
 *     "宽度在动但颜色透明"这类 portal 令牌问题的取点（不是靠
 *     `[style*="width"]` 这种脆弱选择器）；
 *   - 0% 不给最小宽度：避免出现一个"已经开始了"的假点。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test } from "vitest";

import { I18nProvider } from "../i18n/I18nProvider";
import { ProgressBar } from "./ProgressBar";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    document.body.innerHTML = "";
});

function renderProgressBar(percentage: number, label?: string): void {
    act(() => {
        root.render(
            <I18nProvider>
                <ProgressBar percentage={percentage} label={label} />
            </I18nProvider>,
        );
    });
}

test("呈现读数、填充宽度与诊断钩子", () => {
    renderProgressBar(37, "导出中");

    expect(container.textContent).toContain("导出中");
    expect(container.textContent).toContain("37%");

    const track = container.querySelector('[role="progressbar"]');
    expect(track).not.toBeNull();
    expect(track?.getAttribute("aria-valuenow")).toBe("37");
    expect(track?.getAttribute("aria-valuemin")).toBe("0");
    expect(track?.getAttribute("aria-valuemax")).toBe("100");

    // 诊断取点：宽度在动而颜色透明的 portal 令牌问题时，用它能一眼定位。
    const fill = container.querySelector<HTMLElement>('[data-hs-progress-fill="1"]');
    expect(fill).not.toBeNull();
    expect(fill?.dataset.percentage).toBe("37");
    expect(fill?.style.width).toBe("37%");
    // 小百分比仍可见。
    expect(fill?.style.minWidth).toBe("3px");
});

test("0% 不给最小宽度（不出现“已经开始了”的假点）", () => {
    renderProgressBar(0);

    const fill = container.querySelector<HTMLElement>('[data-hs-progress-fill="1"]');
    expect(fill?.style.width).toBe("0%");
    expect(fill?.style.minWidth).toBe("0px");
    expect(container.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe(
        "0",
    );
});

test("越界值被夹取到 [0, 100]", () => {
    renderProgressBar(140);
    let fill = container.querySelector<HTMLElement>('[data-hs-progress-fill="1"]');
    expect(fill?.style.width).toBe("100%");
    expect(container.querySelector('[role="progressbar"]')?.getAttribute("aria-valuenow")).toBe(
        "100",
    );

    renderProgressBar(-5);
    fill = container.querySelector<HTMLElement>('[data-hs-progress-fill="1"]');
    expect(fill?.style.width).toBe("0%");
    expect(fill?.style.minWidth).toBe("0px");
});
