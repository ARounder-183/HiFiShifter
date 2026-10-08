/**
 * MIDI 导入对话框在插件模式下的**导入目标**。
 *
 * 【要钉死什么】插件的时间线是宿主清单的投影（`workspace_timeline_locked` 只保留
 * 已分配 region 的 clip），因此"建成片段"（`import_midi_as_clip` /
 * `replace_midi_clip_data`）在插件里做不到 —— 造出来的片段会在下一次宿主同步时
 * 消失。而"导入到音高曲线"写的是插件自己的权威，完全可用。
 *
 * 于是对话框必须做两件事：把目标**锁**到音高曲线（不沿用独立 App 里持久化的
 * `pitchRef`），并把那个不可用的选项**禁用并说明**。此前整条链路在开启点就被挡掉，
 * 用户在一个本来能用的功能上什么也得不到。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { MidiTrackSelectDialog } from "./MidiTrackSelectDialog";

// `vi.mock` 会被提升到文件顶部，所以这个 spy 必须用 `vi.hoisted` 一起提升 ——
// 否则工厂函数在初始化前就引用了它。
const { importMidiToPitch } = vi.hoisted(() => ({
    importMidiToPitch: vi.fn(async () => ({ ok: true, notes_imported: 4, frames_touched: 40 })),
}));

vi.mock("../../services/api/params", async () => {
    const actual = await vi.importActual<typeof import("../../services/api/params")>(
        "../../services/api/params",
    );
    return {
        ...actual,
        paramsApi: {
            ...actual.paramsApi,
            getMidiTracks: vi.fn(async () => ({
                ok: true,
                tracks: [{ index: 0, name: "Piano", note_count: 4, min_note: 60, max_note: 64 }],
                initial_bpm: 120,
                has_bpm: true,
            })),
            readMidiClipboardToMemory: vi.fn(async () => ({ ok: false })),
            importMidiToPitch,
        },
    };
});

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// Radix 的 RadioGroup 量尺寸；jsdom 没有 ResizeObserver。
class ResizeObserverStub {
    observe() {}
    unobserve() {}
    disconnect() {}
}
(globalThis as { ResizeObserver?: unknown }).ResizeObserver ??= ResizeObserverStub;

let container: HTMLDivElement;
let root: Root;

beforeEach(() => {
    localStorage.setItem("hifishifter.locale", "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    importMidiToPitch.mockClear();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem("hifishifter.locale");
    delete window.__HFS_PLUGIN_BOOTSTRAP__;
    vi.restoreAllMocks();
});

/** 挂载对话框；`onImportAsClip` 用 spy 以便区分两条导入路径。 */
async function render(onImportAsClip = vi.fn()) {
    await act(async () =>
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <MidiTrackSelectDialog
                        open
                        onOpenChange={() => undefined}
                        midiPath="C:\\midi\\note.mid"
                        importTarget="pitchRef"
                        projectBpm={120}
                        onImportAsClip={onImportAsClip}
                    />
                </I18nProvider>
            </AppThemeProvider>,
        ),
    );
    // 轨道列表是异步拉回来的，页脚按钮要等它落地。
    await act(async () => undefined);
    return onImportAsClip;
}

/**
 * 按标签文字定位"导入目标"的两个单选。
 *
 * 【为什么不直接查 `[role="radio"]`】对话框里还有别的单选组（导入位置、音符 BPM
 * 模式…），全量查询会把它们一起捞进来。这里按标签文字收窄到目标那一组。
 */
function targetRadio(labelText: string) {
    const label = Array.from(document.querySelectorAll("label")).find((candidate) =>
        candidate.textContent?.includes(labelText),
    );
    const radio = label?.querySelector<HTMLButtonElement>('[role="radio"]');
    if (!label || !radio) throw new Error(`target radio not found: ${labelText}`);
    return { label, radio };
}

function buttonNamed(label: string): HTMLButtonElement | undefined {
    return Array.from(document.querySelectorAll("button")).find((b) =>
        b.textContent?.includes(label),
    );
}

test("standalone keeps both targets, including the persisted clip target", async () => {
    await render();
    const param = targetRadio(enUS.midi_import_target_pitch_param);
    const block = targetRadio(enUS.midi_import_target_pitch_block);
    expect(param.radio.disabled).toBe(false);
    expect(block.radio.disabled).toBe(false);
    // 独立 App 里持久化的 `pitchRef` 照常生效。
    expect(block.radio.getAttribute("aria-checked")).toBe("true");
});

test("plugin locks the target to the pitch curve and explains the missing one", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "midi" };
    const onImportAsClip = await render();

    const param = targetRadio(enUS.midi_import_target_pitch_param);
    const block = targetRadio(enUS.midi_import_target_pitch_block);
    // 目标被锁到音高曲线：持久化的 `pitchRef` 不得在插件里生效。
    expect(param.radio.getAttribute("aria-checked")).toBe("true");
    expect(block.radio.getAttribute("aria-checked")).toBe("false");
    // 不可用的选项必须禁用**并给出原因**，而不是留着点了没反应。
    expect(block.radio.disabled).toBe(true);
    expect(block.label.getAttribute("title")).toBe(enUS.midi_import_clip_plugin_unavailable);

    const importButton = buttonNamed(enUS.midi_import);
    expect(importButton).toBeTruthy();
    await act(async () => importButton!.click());
    // 走的是"导入到音高曲线"，而不是建本地片段。
    expect(importMidiToPitch).toHaveBeenCalledTimes(1);
    expect(onImportAsClip).not.toHaveBeenCalled();
});
