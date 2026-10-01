// @vitest-environment jsdom
/*
 * 「添加颤音」应用弹窗的行为契约。
 *
 * 【为什么值得测】这个弹窗有两处"点下去毫无动静"的失败形态，都不抛错：
 * 1. **应用载荷**：op 侧从"按 presetId 解析"改成"接受完整预设对象"，若弹窗仍
 *    只传 id，本地微调的深度 / 速率会被静默丢弃；
 * 2. **保存到预设**：默认关闭时**绝不能**写库（"预设被悄悄改掉"是这类工具最
 *    恼人的错法），勾选后才 upsert。
 *
 * 这里用真实 store + 真实 i18n 挂载，断言选中、应用载荷与保存开关三条链路。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { Provider } from "react-redux";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import keybindingsReducer from "../../features/keybindings/keybindingsSlice";
import sessionReducer from "../../features/session/sessionSlice";
import { sanitizeVibratoPreset } from "../../features/vibrato/vibratoPresets";
import type { VibratoPreset } from "../../features/vibrato/vibratoTypes";
import { I18nProvider } from "../../i18n/I18nProvider";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { VibratoApplyDialog } from "./VibratoApplyDialog";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom 没有 ResizeObserver；预览画布与滚动区都会构造它。
class ResizeObserverStub {
    observe(): void {}
    unobserve(): void {}
    disconnect(): void {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;
// 预览画布用 2D context 绘制；jsdom 默认返回 null，会让组件提前返回（无害）。
(HTMLCanvasElement.prototype.getContext as unknown) ??= () => null;

const PRESETS: VibratoPreset[] = [
    sanitizeVibratoPreset({ id: "builtin.natural", depthCents: 30, rateHz: 5.5 }),
    sanitizeVibratoPreset({ id: "custom_a", name: "Mine", depthCents: 42, rateHz: 6.5 }),
];

/*
 * 一段真实的音高曲线：C4 附近 ±0.5 个半音。
 *
 * 【为什么从 60 起而不是从 0 起】音高参数里 **0 = 未检测到音高**（见
 * `vibratoPitch.ts`）。用 `sin(i/8) * 0.5` 当音高夹具既不是真实音高，i=0 处又恰好
 * 落在 0 上 —— 那是在测一个不存在的情形，新契约还会把那一帧画成断口。
 */
const ORIGINAL = Array.from({ length: 64 }, (_, i) => 60 + Math.sin(i / 8) * 0.5);

let host: HTMLDivElement;
let root: Root;

beforeEach(() => {
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
});

afterEach(async () => {
    await act(async () => root.unmount());
    document.body.innerHTML = "";
});

function findButton(text: string): HTMLButtonElement | undefined {
    return [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === text,
    ) as HTMLButtonElement | undefined;
}

async function mountDialog(
    overrides: {
        onApply?: (preset: VibratoPreset) => void;
        onExtract?: () => void;
        loadOriginal?: () => Promise<{ values: number[]; framePeriodMs: number } | null>;
    } = {},
) {
    const store = configureStore({
        reducer: { session: sessionReducer, keybindings: keybindingsReducer },
    });
    const onApply = overrides.onApply ?? vi.fn();
    const loadOriginal =
        overrides.loadOriginal ?? (() => Promise.resolve({ values: ORIGINAL, framePeriodMs: 5 }));
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <VibratoApplyDialog
                            open
                            onOpenChange={() => undefined}
                            presets={PRESETS}
                            activePresetId="custom_a"
                            editParam="pitch"
                            paramRange={{ min: -24, max: 24 }}
                            loadOriginal={loadOriginal}
                            onApply={onApply}
                            onExtract={overrides.onExtract}
                        />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
    // 让 `loadOriginal` 的 promise 落地。
    await act(async () => {
        await Promise.resolve();
    });
    return { store, onApply, loadOriginal };
}

test("左列列出全部预设，打开时预选活动预设", async () => {
    await mountDialog();
    const rows = document.querySelectorAll('[role="option"]');
    expect(rows.length).toBe(PRESETS.length);
    // 活动预设（custom_a）那一行被选中。
    const selected = [...rows].filter((row) => row.getAttribute("aria-selected") === "true");
    expect(selected.length).toBe(1);
    expect(selected[0]?.textContent ?? "").toContain("Mine");
});

test("拿到选区数据后画出套用预览（画布存在）", async () => {
    const { loadOriginal } = await mountDialog();
    expect(loadOriginal).toBeDefined();
    expect(document.querySelector("canvas[role=img]")).toBeTruthy();
});

test("应用：把完整预设对象交给编辑管线（本地微调随之传入）", async () => {
    const onApply = vi.fn();
    await mountDialog({ onApply });

    const apply = findButton("Apply");
    expect(apply, "应用按钮应已渲染").toBeTruthy();
    await act(async () => {
        apply!.click();
    });

    expect(onApply).toHaveBeenCalledTimes(1);
    const payload = onApply.mock.calls[0][0] as VibratoPreset;
    expect(payload.id).toBe("custom_a");
    expect(payload.depthCents).toBe(42);
    expect(payload.rateHz).toBe(6.5);
});

test("默认不写库：应用只改选区，不动预设", async () => {
    const { store } = await mountDialog();
    await act(async () => {
        findButton("Apply")!.click();
    });
    expect(store.getState().session.vibratoPresets.length).toBe(0);
});

test("勾选「保存到预设」后才 upsert（并且仍然应用）", async () => {
    const onApply = vi.fn();
    const { store } = await mountDialog({ onApply });

    const checkbox = document.querySelector<HTMLButtonElement>('[role="checkbox"]');
    expect(checkbox, "保存开关应已渲染").toBeTruthy();
    await act(async () => {
        checkbox!.click();
    });
    await act(async () => {
        findButton("Apply")!.click();
    });

    const saved = store.getState().session.vibratoPresets;
    expect(saved.length).toBe(1);
    expect(saved[0]?.id).toBe("custom_a");
    expect(saved[0]?.depthCents).toBe(42);
    expect(onApply).toHaveBeenCalledTimes(1);
});

test("选中系统预设时，保存开关被禁用（系统预设只读）", async () => {
    await mountDialog();
    const rows = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    const builtinRow = rows.find((row) => row.textContent?.includes("Natural"));
    expect(builtinRow, "系统预设行应已渲染").toBeTruthy();
    await act(async () => {
        builtinRow!.click();
    });
    const checkbox = document.querySelector<HTMLButtonElement>('[role="checkbox"]');
    expect(checkbox?.disabled).toBe(true);
});

test("页脚「从选区提取…」可用（有 onExtract 时）", async () => {
    const onExtract = vi.fn();
    await mountDialog({ onExtract });
    const extract = [...document.querySelectorAll("button")].find((button) =>
        (button.textContent ?? "").includes("Create vibrato preset from selection"),
    );
    expect(extract, "提取按钮应已渲染").toBeTruthy();
    await act(async () => {
        extract!.click();
    });
    expect(onExtract).toHaveBeenCalledTimes(1);
});

test("无选区数据时显示占位提示而不是空画布", async () => {
    await mountDialog({ loadOriginal: () => Promise.resolve(null) });
    expect(document.querySelector("canvas[role=img]")).toBeNull();
    expect(document.body.textContent ?? "").toContain("Select a range");
});

/*
 * 音高不可调制：说清原因，而不是复用"选一段"那句提示。
 *
 * 【为什么值得测】音高参数里 0 = 未检测；此外浊清边界上还有"低而非零"的过渡帧，
 * 以及短得不成其为音符的碎片（见 `vibratoPitch.ts`）。若沿用同一句"选中一段后可在此
 * 预览效果"，用户会以为是自己没选对区域，反复重选 —— 而真正的原因是这段里没有可
 * 加颤音的音高。同时必须没有画布：一条"平在 0"的曲线会被读成"结果把音高拉平了"。
 */
test("音高不可调制时（未检测 / 没有够长的音符）：说明原因，且不画曲线", async () => {
    // 整段未检测。
    await mountDialog({
        loadOriginal: () => Promise.resolve({ values: [0, 0, 0, 0], framePeriodMs: 5 }),
    });
    expect(document.querySelector("canvas[role=img]")).toBeNull();
    expect(document.body.textContent ?? "").toContain("No pitch to apply vibrato to");
    expect(document.body.textContent ?? "").not.toContain("Select a range");
});

test("音高有值但短得不成音符时，同样给出原因而不是画一条直线", async () => {
    // 3 帧（15ms）的真实音高夹在气口之间 —— 够不上 `MIN_NOTE_MS` 的门槛。
    await mountDialog({
        loadOriginal: () =>
            Promise.resolve({
                values: [
                    ...new Array<number>(10).fill(0),
                    60,
                    61,
                    62,
                    ...new Array<number>(10).fill(0),
                ],
                framePeriodMs: 5,
            }),
    });
    expect(document.querySelector("canvas[role=img]")).toBeNull();
    expect(document.body.textContent ?? "").toContain("No pitch to apply vibrato to");
    expect(document.body.textContent ?? "").not.toContain("Select a range");
});
