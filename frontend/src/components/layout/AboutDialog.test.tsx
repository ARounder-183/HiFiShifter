/**
 * 关于对话框在两种形态下的行为差异。
 *
 * 【要钉死什么】插件跑在宿主进程里，WebView2 会取消一切指向外部域的导航，也没有
 * 任何打开 URL 的原生通道 —— 于是 `openExternal` 在插件里**静默什么都不做**。
 * 一个点了没反应的按钮比一个被明确替换掉的按钮更糟：用户会以为对话框坏了。
 *
 * 数据本来就是齐的（`get_about_info` 与独立 App 逐字段同形），所以插件里改成
 * "复制链接"；同时 commit 必须变成**可选中**的文本 —— 全站默认 `user-select: none`。
 */
// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

import { I18nProvider } from "../../i18n/I18nProvider";
import { enUS } from "../../i18n/en-US";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { AboutDialog } from "./AboutDialog";

vi.mock("../../services/api/core", async () => {
    const actual =
        await vi.importActual<typeof import("../../services/api/core")>("../../services/api/core");
    return {
        ...actual,
        coreApi: {
            ...actual.coreApi,
            getAboutInfo: vi.fn(async () => ({
                version: "1.2.3",
                commit: "0123456789abcdef0123456789abcdef01234567",
                commitShort: "0123456789a",
                dirty: false,
                repoUrl: "https://github.com/example/repo",
            })),
        },
    };
});

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const ABOUT = {
    version: "1.2.3",
    commit: "0123456789abcdef0123456789abcdef01234567",
    commitShort: "0123456789a",
    dirty: false,
    repoUrl: "https://github.com/example/repo",
};

let container: HTMLDivElement;
let root: Root;
let writeText: ReturnType<typeof vi.fn>;

beforeEach(() => {
    localStorage.setItem("hifishifter.locale", "en-US");
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    writeText = vi.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });
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

async function render() {
    await act(async () =>
        root.render(
            <AppThemeProvider>
                <I18nProvider>
                    <AboutDialog open onOpenChange={() => undefined} />
                </I18nProvider>
            </AppThemeProvider>,
        ),
    );
}

function buttonNamed(label: string): HTMLButtonElement | undefined {
    return Array.from(document.querySelectorAll("button")).find((b) =>
        b.textContent?.includes(label),
    );
}

test("standalone offers the repository link and the commit permalink", async () => {
    await render();
    expect(buttonNamed(enUS.about_open_repo)).toBeTruthy();
    // 独立 App 里 commit 是可点击的链接，不是"复制"按钮。
    expect(buttonNamed(enUS.about_copy_commit_link)).toBeUndefined();
});

test("plugin replaces the dead link with a copy action", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "about" };
    await render();

    // 打开仓库的按钮在插件里必须是"复制"，不能是点了没反应的链接。
    expect(buttonNamed(enUS.about_open_repo)).toBeUndefined();
    const copyRepo = buttonNamed(enUS.about_copy_repo_link);
    expect(copyRepo).toBeTruthy();

    await act(async () => copyRepo!.click());
    expect(writeText).toHaveBeenCalledWith(ABOUT.repoUrl);

    // commit 的永久链接同样可以复制。
    const copyCommit = buttonNamed(enUS.about_copy_commit_link);
    expect(copyCommit).toBeTruthy();
    await act(async () => copyCommit!.click());
    expect(writeText).toHaveBeenCalledWith(`${ABOUT.repoUrl}/tree/${ABOUT.commit}`);
});

test("plugin makes the commit selectable text", async () => {
    window.__HFS_PLUGIN_BOOTSTRAP__ = { version: 1, viewId: "about" };
    await render();
    const selectable = Array.from(
        document.querySelectorAll<HTMLElement>('[data-hs-selectable="true"]'),
    ).map((element) => element.textContent);
    // 短哈希必须能被选中 —— 否则用户连抄给开发者都做不到。
    expect(selectable).toContain(ABOUT.commitShort);
});
