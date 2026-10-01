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
        onEditPresets?: (presetId: string) => void;
        initialPresetId?: string;
        onOpenChange?: (open: boolean) => void;
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
                            onOpenChange={overrides.onOpenChange ?? (() => undefined)}
                            presets={PRESETS}
                            activePresetId="custom_a"
                            initialPresetId={overrides.initialPresetId}
                            editParam="pitch"
                            paramRange={{ min: -24, max: 24 }}
                            loadOriginal={loadOriginal}
                            onApply={onApply}
                            onExtract={overrides.onExtract}
                            onEditPresets={overrides.onEditPresets}
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

/*
 * 选中系统预设时，保存开关**仍然可用**（勾上会自动另存为自定义副本）。
 *
 * 【为什么改了】系统预设只读指的是"改不了库里那一条"，不是"用户的调整不能留"。
 * 禁用开关等于把"调完留住"这条路堵死，用户只能手动去管理器复制一遍。
 */
test("选中系统预设时，保存开关仍可用（保存即另存为自定义）", async () => {
    await mountDialog();
    const rows = [...document.querySelectorAll<HTMLElement>('[role="option"]')];
    const builtinRow = rows.find((row) => row.textContent?.includes("Natural"));
    expect(builtinRow, "系统预设行应已渲染").toBeTruthy();
    await act(async () => {
        builtinRow!.click();
    });
    const checkbox = document.querySelector<HTMLButtonElement>('[role="checkbox"]');
    expect(checkbox?.disabled).toBe(false);
});

test("页脚「从选区提取颤音预设」可用（有 onExtract 时）", async () => {
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
 * 预览是**一张图**：轮廓叠在颤音偏移之上，右侧是 A/B 试听。
 *
 * 【为什么值得测】轮廓与颤音差两个数量级，早先拆成上下两条带（各有各的标尺），
 * 但那样要对比"颤音走在音高的哪一段上"就得上下看 —— 用户明确要求叠起来。
 * 现在同图叠放：轮廓按自身范围铺满画布做背景虚线，颤音偏移按 cents 标尺画在前景；
 * 试听按同一条时间轴、同一个中心给出原参数线 / 新参数线两条，供 A/B。
 */
test("预览是一张图：轮廓叠在颤音偏移之上，并带 A/B 试听按钮", async () => {
    await mountDialog();
    const canvases = [...document.querySelectorAll("canvas[role=img]")];
    expect(canvases.length).toBe(1);
    // 图例文案已按要求移除（那条虚线由试听按钮的文字承担说明）。
    expect(document.body.textContent ?? "").not.toContain("Dashed");
    const labels = [...document.querySelectorAll("button")].map((button) =>
        button.textContent?.trim(),
    );
    expect(labels).toContain("Audition original");
    expect(labels).toContain("Audition result");
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

/*
 * 系统预设也能"把这些调整保存到预设"：自动另存为一份自定义副本。
 *
 * 【为什么值得测】系统预设本身只读，但用户调完旋钮想留住结果。旧行为是静默跳过保存
 * （勾了等于没勾），用户会以为存下了 —— 这是这类工具最恼人的错法之一。而且**存下的
 * 那份必须就是应用的那份**：否则"保存"与"应用"分叉，用户回头找预设会发现对不上。
 */
test("系统预设勾选保存到预设：自动另存为自定义副本，且应用的就是它", async () => {
    const onApply = vi.fn();
    const { store } = await mountDialog({ onApply });

    // 切到系统预设「Natural」。
    const naturalRow = [...document.querySelectorAll<HTMLElement>('[role="option"]')].find((row) =>
        row.textContent?.includes("Natural"),
    );
    expect(naturalRow, "系统预设行应已渲染").toBeTruthy();
    await act(async () => {
        naturalRow!.click();
    });
    // 只读提示已按要求移除；保存开关对系统预设同样可用。
    expect(document.body.textContent ?? "").not.toContain("read-only");

    const saveBox = document.querySelector<HTMLElement>('[role="checkbox"]');
    expect(saveBox, "保存开关应已渲染").toBeTruthy();
    await act(async () => {
        saveBox!.click();
    });

    const apply = findButton("Apply");
    expect(apply, "应用按钮应已渲染").toBeTruthy();
    await act(async () => {
        apply!.click();
    });

    const saved = store.getState().session.vibratoPresets.filter((preset) => !preset.builtin);
    // 名字按显示名预填编号（与管理器里「复制为自定义」同一套规则）。
    expect(saved.map((preset) => preset.name)).toContain("Natural 2");
    // 应用的就是存下的那一份（同一个 id）。
    expect(onApply).toHaveBeenCalledTimes(1);
    expect((onApply.mock.calls[0][0] as VibratoPreset).id).toBe(saved[0]?.id);
});

/*
 * 「编辑预设」：跳到预设管理器去改库，改完由管理器那边「返回添加颤音」送回来。
 *
 * 【为什么先关窗再跳】跳转前要走本弹窗的关闭路径 —— 否则试听会一直响着，而用户已经
 * 在看另一个窗口了。
 */
test("点「编辑预设」：先关掉本弹窗，再带着选中项交给宿主", async () => {
    const onEditPresets = vi.fn();
    const onOpenChange = vi.fn();
    await mountDialog({ onEditPresets, onOpenChange });
    const edit = [...document.querySelectorAll("button")].find(
        (button) => button.textContent?.trim() === "Edit presets",
    );
    expect(edit, "编辑预设按钮应已渲染").toBeTruthy();
    await act(async () => {
        edit!.click();
    });
    // 跳转前先关窗（否则试听会一直响着，而用户已经去看另一个窗口了）。
    expect(onOpenChange).toHaveBeenCalledWith(false);
    // 带过去的是**选中的**那条（默认是活动预设 custom_a）。
    expect(onEditPresets).toHaveBeenCalledWith("custom_a");
});

/*
 * initialPresetId 决定应用弹窗打开时**选中**哪一条。
 *
 * 从管理器返回时带上用户在那边编辑的那条 —— 接着应用的就是它，而不是"当前使用"的
 * 那条（用户刚在管理器里挑了半天，回来又要重新挑一次是不能接受的）。
 */
test("initialPresetId 决定打开时选中哪一条（优先于当前活动预设）", async () => {
    const onApply = vi.fn();
    await mountDialog({ onApply, initialPresetId: "builtin.natural" });
    const apply = findButton("Apply");
    await act(async () => {
        apply!.click();
    });
    expect((onApply.mock.calls[0][0] as VibratoPreset).id).toBe("builtin.natural");
});

/*
 * 预览纵轴由「适应」控制，编辑期间保持不动（与预设管理器同一套逻辑）。
 *
 * 【为什么值得测】标尺若跟着深度自适应，波形永远填满画布 —— 调深度时看到的只是整幅
 * 在竖直方向"抖一下"，读不出幅度大小。标尺固定住，波形高度才等于深度。
 *
 * 【测到哪一步】纵轴画在画布上，jsdom 里读不到；这里钉的是"入口在"（与预设管理器
 * 的同类测试一致）—— 稳定性由依赖数组保证：换选区 / 换预设 / 点适应才重算。
 */
test("预览有「适应」按钮（重新拟合纵轴）", async () => {
    await mountDialog();
    const fit = findButton("Fit");
    expect(fit, "适应按钮应已渲染").toBeTruthy();
    await act(async () => {
        fit!.click();
    });
    // 点完画布还在（重算标尺不该把预览弄没）。
    expect(document.querySelector("canvas[role=img]")).toBeTruthy();
});
