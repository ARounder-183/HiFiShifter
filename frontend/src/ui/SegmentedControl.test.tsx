// @vitest-environment jsdom
/*
 * 段控原语的交互契约。
 *
 * 【为什么必须有】它是"第 7 个角色 + 段控原语"补完轮的新原语，三个消费者
 * （外观面板、导出音频、QuickClip 导出）会同时换上来 —— 键盘模型
 * （roving tabindex + 方向键循环选中）与 radiogroup 语义必须在源头钉死，
 * 而不是靠三个调用点各自记住。
 */
import { act } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, expect, test } from "vitest";

import { AppSegmentedControl } from "./SegmentedControl";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const OPTIONS = [
    { value: "a", label: "A" },
    { value: "b", label: "B" },
    { value: "c", label: "C" },
] as const;

let host: HTMLDivElement | null = null;
let root: ReturnType<typeof createRoot> | null = null;

async function mountControl(initial: string, onChange: (next: string) => void): Promise<void> {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    await act(async () => {
        root!.render(
            <AppSegmentedControl
                value={initial as "a"}
                options={OPTIONS}
                onChange={onChange}
                ariaLabel="格式"
            />,
        );
    });
}

async function unmountControl(): Promise<void> {
    if (root && host) {
        await act(async () => root!.unmount());
        host.remove();
    }
    root = null;
    host = null;
}

afterEach(unmountControl);

test("渲染 radiogroup 与 radio，激活项有 aria-checked 且独占 Tab 停留点", async () => {
    await mountControl("b", () => {});
    const group = host!.querySelector('[role="radiogroup"]');
    expect(group?.getAttribute("aria-label")).toBe("格式");
    const radios = [...host!.querySelectorAll('[role="radio"]')];
    expect(radios.length).toBe(3);
    expect(radios[1].getAttribute("aria-checked")).toBe("true");
    expect(radios[1].getAttribute("tabindex")).toBe("0");
    expect(radios[0].getAttribute("tabindex")).toBe("-1");
});

test("方向键循环移动并选中，Home/End 到两端", async () => {
    const seen: string[] = [];
    const render = (value: string) => {
        root!.render(
            <AppSegmentedControl
                value={value as "a"}
                options={OPTIONS}
                onChange={(next) => {
                    seen.push(next);
                    render(next);
                }}
                ariaLabel="格式"
            />,
        );
    };
    await mountControl("a", (next) => {
        seen.push(next);
        render(next);
    });
    const group = host!.querySelector('[role="radiogroup"]')!;
    const fire = (key: string) => {
        act(() => {
            group.dispatchEvent(
                new KeyboardEvent("keydown", { key, bubbles: true, cancelable: true }),
            );
        });
    };
    fire("ArrowRight");
    fire("ArrowRight");
    fire("ArrowRight"); // 从 c 循环回 a
    fire("ArrowLeft"); // 回 c
    fire("Home");
    fire("End");
    expect(seen).toEqual(["b", "c", "a", "c", "a", "c"]);
});

test("点击选项触发 onChange", async () => {
    let picked = "";
    await mountControl("a", (next) => {
        picked = next;
    });
    const second = host!.querySelectorAll('[role="radio"]')[1] as HTMLButtonElement;
    await act(async () => second.click());
    expect(picked).toBe("b");
});
