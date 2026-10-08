// @vitest-environment jsdom
/**
 * 宿主音频提示条的契约。
 *
 * 【这个文件钉住两件事】
 * 1. **只在 folder 父轨这一种情形出现**：`awaiting_regions` 是打开工程时的正常中间态，
 *    为它弹横条等于每次打开工程都打扰用户一次；
 * 2. 文案**走词表**且措辞正确 —— 不能说"ARA 规范不支持跨轨"（ARA 2.0 规范允许一个实例
 *    服务多个 region sequence；真正的原因是 REAPER 按轨道管理 ARA 插件）。措辞错了会把
 *    用户引向错误的排查方向。
 */
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { HostAudioNotice } from "./HostAudioNotice";
import type { HostAudioPayload } from "../../types/api";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let container: HTMLDivElement;
let root: Root;

const LOCALE_KEY = "hifishifter.locale";

beforeEach(() => {
    localStorage.setItem(LOCALE_KEY, "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
});

afterEach(() => {
    act(() => root.unmount());
    container.remove();
    document.body.innerHTML = "";
    localStorage.removeItem(LOCALE_KEY);
    vi.restoreAllMocks();
});

async function render(status: HostAudioPayload | null) {
    await act(async () =>
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <HostAudioNotice status={status} />
                </I18nProvider>
            </AppThemeProvider>,
        ),
    );
}

test("folder parent track shows the reason and the next step", async () => {
    await render({ state: "folder_parent_without_regions", waiting_clips: 2 });
    const notice = container.querySelector('[role="status"]');
    expect(notice).toBeTruthy();
    expect(notice?.textContent).toContain(enUS.ara_host_audio_folder_title);
    expect(notice?.textContent).toContain(enUS.ara_host_audio_folder_body);
    // 正确的因果表述：REAPER 按轨道管理 ARA 插件。
    expect(enUS.ara_host_audio_folder_body).toContain("REAPER manages ARA plug-ins per track");
    // 措辞红线：不得出现"ARA 规范不支持跨轨"这类错误解释。
    expect(enUS.ara_host_audio_folder_body).not.toMatch(/spec|specification/i);
});

test("a plain waiting state stays quiet", async () => {
    await render({ state: "awaiting_regions", waiting_clips: 2 });
    expect(container.querySelector('[role="status"]')).toBeNull();
    expect(container.textContent).toBe("");
});

test("a healthy instance stays quiet", async () => {
    await render({ state: "ready", waiting_clips: 0 });
    expect(container.querySelector('[role="status"]')).toBeNull();
});

test("an unknown reading stays quiet", async () => {
    await render(null);
    expect(container.querySelector('[role="status"]')).toBeNull();
});
