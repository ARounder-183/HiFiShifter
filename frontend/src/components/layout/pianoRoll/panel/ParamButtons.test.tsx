// @vitest-environment jsdom
/*
 * 参数药丸的**内容不变量**：层级只能改 `display`，不能改 DOM。
 *
 * 【为什么这是最重要的一条回归测试】层级是"按当前内容有多宽"量出来的。只要某一级的
 * 内容比另一级少（例如被门禁的药丸 `return null`），判据就变成自反馈：
 *
 *   2 级放不下 → 升到 3 级 → 3 级少一个药丸 → 再量，2 级又放得下了 → 降回 2 级
 *   → 药丸回来 → 又放不下 → 升到 3 级 → …
 *
 * 每帧一轮、永不收敛，用户看到的就是工具栏上的控件反复横跳、一直闪。因此这里对
 * **每一个层级**渲染一遍，断言剥掉 `style` 之后的 HTML 与第 0 级**逐字节相同** ——
 * 任何"按层级增删控件"的改动都会在这里断掉。
 */

import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import type { ProcessorParamDescriptor } from "../../../../types/api";
import { NEVER_HIDE_TIER } from "../toolbarOverflow";
import { ParamButtons } from "./ParamButtons";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/** 组件拿到的模型切片（由组件自身的 props 推导，避免手抄一份类型）。 */
type Model = Parameters<typeof ParamButtons>[0]["m"];

/** 三个参数够用：一个普通参数 + 两个被分离门禁的药丸（让位阶梯的第 3 / 4 级）。 */
const PARAMS: ProcessorParamDescriptor[] = ["pitch", "hifigan_tension", "breath_gain"].map(
    (id) => ({
        id,
        display_name: id,
        group: "pitch",
        kind: { type: "automation_curve", unit: "", default_value: 0, min_value: 0, max_value: 1 },
    }),
);

function model(overrides: Partial<Model> = {}): Model {
    return {
        childFormantOffsetParam: null,
        composeEnabled: true,
        dispatch: (() => undefined) as unknown as Model["dispatch"],
        editParam: "pitch",
        // 顺序与 gatedParamHideOrder 一致：先张力、后气声。
        gatedPillHideOrder: ["hifigan_tension", "breath_gain"],
        getProcessorParamLabel: (param: ProcessorParamDescriptor) => param.display_name,
        getProcessorParamShortLabel: (param: ProcessorParamDescriptor) => param.id,
        handleStaticParamChange: async () => undefined,
        orderedProcessorParams: PARAMS,
        processorParamsRef: { current: PARAMS },
        secondaryParamVisible: {},
        selectedIsChildTrack: false,
        separationEnabled: false,
        separationSwitchAnchorParamId: "breath_gain",
        t: (key: string) => key,
        toggleSecondaryParam: () => undefined,
        toolbarTier: 0,
        ...overrides,
    } as unknown as Model;
}

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

function renderAt(tier: number, overrides: Partial<Model> = {}): HTMLElement {
    act(() => {
        root.render(<ParamButtons m={model({ toolbarTier: tier, ...overrides })} />);
    });
    return container;
}

/** 剥掉内联 `style`（唯一随层级变化的东西）之后的 DOM。 */
function structureAt(tier: number): string {
    return renderAt(tier).innerHTML.replace(/ style="[^"]*"/g, "");
}

describe("ParamButtons 的内容不随层级变化", () => {
    it("第 0–12 级渲染出**完全一样**的 DOM（只有 display 变）", () => {
        const baseline = structureAt(0);
        expect(baseline).toContain("hifigan_tension");
        expect(baseline).toContain("breath_gain");

        for (let tier = 1; tier <= 12; tier += 1) {
            expect(structureAt(tier), "第 " + tier + " 级的 DOM 与第 0 级不同").toBe(baseline);
        }
    });

    it("确实有槽位在某级被隐藏（否则上一条会退化成「什么都没测」）", () => {
        const hidden = Array.from(
            renderAt(4).querySelectorAll<HTMLElement>("[data-hide-at]"),
        ).filter((element) => element.style.display === "none");
        expect(hidden.map((element) => element.dataset.hideAt)).toEqual(["3", "4"]);
    });
});

describe("被门禁的药丸：让位而不是消失", () => {
    it("第 3 级隐藏张力药丸，第 4 级再隐藏气声药丸", () => {
        expect(renderAt(2).querySelector<HTMLElement>('[data-hide-at="3"]')!.style.display).toBe(
            "contents",
        );

        const tension = renderAt(3).querySelector<HTMLElement>('[data-hide-at="3"]')!;
        expect(tension.textContent).toContain("hifigan_tension");
        expect(tension.style.display).toBe("none");

        expect(renderAt(4).querySelector<HTMLElement>('[data-hide-at="4"]')!.style.display).toBe(
            "none",
        );
    });

    it("气声药丸让位后，它左侧的分离开关仍在（重新开启的唯一入口）", () => {
        const switchButton = renderAt(4).querySelector<HTMLElement>(".param-pill__seg--breath");
        expect(switchButton).toBeTruthy();
        for (let node = switchButton; node && node !== container; node = node.parentElement) {
            expect(node.style.display).not.toBe("none");
        }
    });

    it("分离开启时两个药丸都不参与让位（它们是活入口，第 4–10 级没有别的入口）", () => {
        const at = renderAt(12, { separationEnabled: true });
        expect(at.querySelector('[data-hide-at="3"]')).toBeNull();
        expect(at.querySelector('[data-hide-at="4"]')).toBeNull();
        expect(at.textContent).toContain("hifigan_tension");
    });

    it("NEVER_HIDE_TIER（菜单副本）下所有槽位都显示", () => {
        const at = renderAt(NEVER_HIDE_TIER);
        const slots = Array.from(at.querySelectorAll<HTMLElement>("[data-hide-at]"));
        expect(slots.length).toBeGreaterThan(0);
        expect(slots.every((element) => element.style.display === "contents")).toBe(true);
    });
});
