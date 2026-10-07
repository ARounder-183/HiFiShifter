// @vitest-environment jsdom
/*
 * 外观设置面板的渲染冒烟测试。
 *
 * 【为什么必须有】2026-09-28 的重排把六张卡片换成了 `AppFormSection` 留白分组、
 * 圆角磁贴补上了可见文字。这些是**用户可见**的承诺，纯逻辑测试盖不住：
 * "节存在"与"节可见"、"磁贴有图形"与"磁贴有文字"是两件事 —— 后者一路绿灯
 * 的教训见 `hifiClipNodeView.test.tsx`（同一个文件开头的说明）。
 *
 * 【挂载方式】`createRoot` + `act` 的真实挂载：`Provider`（面板要 dispatch
 * `closeForm`）、`AppThemeProvider`（面板读写主题草稿）、`I18nProvider`（文案；
 * 非 Tauri 环境下不会调用后端）。
 */
import { configureStore } from "@reduxjs/toolkit";
import { act } from "react";
import { createRoot } from "react-dom/client";
import { Provider } from "react-redux";
import { expect, test } from "vitest";

import { I18nProvider } from "../../i18n/I18nProvider";
import sessionReducer from "../../features/session/sessionSlice";
import { AppThemeProvider } from "../../theme/AppThemeProvider";
import { AppearanceSettingsPanel } from "./AppearanceSettingsPanel";

// React 19 要求显式声明这是 act() 环境，否则每次 act 都会打印一条警告。
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

/**
 * 挂载一个外观设置面板。
 *
 * 【为什么掐掉 canvas】jsdom 没有画布实现；面板挂载时的系统字体探测会走
 * canvas fallback，让它安静地拿到 `null` 并返回空列表即可。
 */
async function mountPanel() {
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = () => null;
    /*
     * 面板只在"关闭自己"时 dispatch（冒烟测试点不到），但「搜索」页签要读
     * `session.searchSettings` —— 那是它渲染匹配方式控件的真值来源，
     * 因此挂上真实的 session 切片而不是哑 reducer。
     */
    const store = configureStore({
        reducer: { session: sessionReducer },
    });
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    await act(async () => {
        root.render(
            <Provider store={store}>
                <AppThemeProvider>
                    <I18nProvider>
                        <AppearanceSettingsPanel formId="appearance#test" />
                    </I18nProvider>
                </AppThemeProvider>
            </Provider>,
        );
    });
    return {
        host,
        unmount: async () => {
            await act(async () => root.unmount());
            host.remove();
            HTMLCanvasElement.prototype.getContext = originalGetContext;
        },
    };
}

test("主题页按 AppFormSection 分节，节标题走角色层，没有卡片堆叠", async () => {
    const mounted = await mountPanel();
    try {
        const sections = mounted.host.querySelectorAll("section");
        // 已保存主题 / 主题模式 / 强调色 / 圆角 / 颜色
        expect(sections.length).toBe(5);
        for (const section of sections) {
            expect(
                section.querySelector("h3.hs-type-section"),
                "节标题必须用角色类，而不是自己写字号",
            ).not.toBeNull();
        }
        // 旧的卡片语言必须消失：不再有 "rounded-md + 边框 + 底色" 的卡片容器。
        expect(
            mounted.host.querySelector(".rounded-md.border.border-qt-border.bg-qt-panel"),
        ).toBeNull();
    } finally {
        await mounted.unmount();
    }
});

test("圆角磁贴有可见文字，不再是无字图形", async () => {
    const mounted = await mountPanel();
    try {
        const labels = Array.from(mounted.host.querySelectorAll("button span")).map(
            (span) => span.textContent ?? "",
        );
        // en-US 目录下的五个档位名（jsdom 的默认语言是 en-US）
        for (const expected of ["None", "Small", "Medium", "Large", "Full"]) {
            expect(labels, `圆角磁贴缺少可见文字：${expected}`).toContain(expected);
        }
    } finally {
        await mounted.unmount();
    }
});

/*
 * 只是打开面板、什么都没改，不得改动已持久化的外观。
 *
 * 预览 effect 的每一次执行都会 `saveAppearance`，而它此前在挂载的第一次 pass 就
 * 无条件跑一遍、把 `activeCustomThemeId` 写死为 null —— 于是"打开看一眼"就会把
 * 正在启用的自定义主题停用并落盘；此时不点「应用」直接关，卸载清理又因草稿不脏
 * 而跳过回滚，停用就永久留下。这条契约无法靠人眼在回归时稳定复验，必须钉住。
 */
test("打开面板（未编辑）不会停用正在使用的自定义主题", async () => {
    localStorage.setItem(
        "hifishifter.appearance",
        JSON.stringify({ mode: "dark", activeCustomThemeId: "ct_test" }),
    );
    localStorage.setItem(
        "hifishifter.customThemes",
        JSON.stringify([
            { id: "ct_test", name: "Test", base: "dark", colors: { "qt-highlight": "#ff0000" } },
        ]),
    );
    const mounted = await mountPanel();
    try {
        const stored = JSON.parse(localStorage.getItem("hifishifter.appearance") ?? "{}") as {
            activeCustomThemeId?: string | null;
        };
        expect(stored.activeCustomThemeId).toBe("ct_test");
    } finally {
        await mounted.unmount();
        localStorage.removeItem("hifishifter.appearance");
        localStorage.removeItem("hifishifter.customThemes");
    }
});
