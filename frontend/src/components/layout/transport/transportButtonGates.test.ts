/**
 * 传输栏与工具栏图标按钮的**源门禁**。
 *
 * 【为什么用源扫描而不是渲染测试】这里要钉的不是"某个按钮长什么样"，而是"同一行里
 * 不许再出现第二套盒模型"。渲染测试只能看见被渲染的那一棵树，而"有人又写了一个裸
 * `IconButton`"是**代码形态**的问题 —— 它会在下一次改动时悄悄回来。
 */
import { describe, expect, test } from "vitest";

const read = (file: string) => import("node:fs").then((fs) => fs.readFileSync(file, "utf8"));

/** 去掉注释，避免"注释里提到 <IconButton>"被误判成真实用法。 */
function stripComments(source: string): string {
    return source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/^\s*\/\/.*$/gm, "");
}

describe("ActionBar 传输组", () => {
    test("stop / play / record / undo / redo 都走 TransportIconButton", async () => {
        const source = stripComments(await read("src/components/layout/toolbar/ToolbarStrip.tsx"));
        expect(source).toContain("<TransportIconButton");
        // 传输栏里不再出现裸的 Radix 按钮：那是"三套盒模型"的直接来源。
        expect(source).not.toMatch(/<IconButton[\s>]/);
        expect(source).not.toMatch(/<Button[\s>]/);
    });

    /**
     * 录音键的填充是**语义**：待机 soft 红、录制中 solid 红。
     *
     * 【为什么钉的是"不要把它固定成 solid"】这条曾经反过来钉过 —— 那时的问题是
     * "红色填充与播放键不齐"，而 fix 选的是把变体固定成 solid。真正的原因是三个
     * 按钮各用一套盒模型：fix 落错了层，还顺手改掉了用户已习惯的配色（待机看着
     * 像"一直在录"）。盒模型已由下面那条 CSS 归一到全部变体，因此这里只钉配色：
     * 常态 soft、`active` 交给原语强制实心。
     */
    test("录音键：常态 soft 红，录制中由 active 转实心", async () => {
        const source = stripComments(await read("src/components/layout/toolbar/ToolbarStrip.tsx"));
        const recordAt = source.indexOf('tone="danger"');
        expect(recordAt).toBeGreaterThan(-1);
        const record = source.slice(recordAt, recordAt + 400);
        expect(record).toContain('fill="soft"');
        expect(record).toContain("active={recording.active}");
        // 不再绕回手写三元变体：那会重新把盒模型交回给调用点。
        expect(source).not.toMatch(/variant=\{recording\.active \? "solid" : "soft"\}/);
    });
});

describe("传输按钮盒模型 CSS", () => {
    test("对所有变体归一化，而不只是 ghost", async () => {
        const css = await read("src/index.css");
        const rule =
            css.match(/\[data-hs-transport-button\]\.rt-IconButton \{([^}]*)\}/)?.[1] ?? "";
        expect(rule).not.toBe("");
        expect(rule).toContain("box-sizing: border-box");
        expect(rule).toContain("width: var(--base-button-height)");
        expect(rule).toContain("height: var(--base-button-height)");
        expect(rule).toContain("padding: 0");
    });
});

describe("Capture 控件", () => {
    test("与传输栏共用同一原语", async () => {
        const source = stripComments(await read("src/features/capture/CaptureControl.tsx"));
        expect(source).toContain("TransportIconButton");
        expect(source).not.toMatch(/<AppIconButton[\s>]/);
    });
});
