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
        const source = stripComments(await read("src/components/layout/ActionBar.tsx"));
        expect(source).toContain("<TransportIconButton");
        // 传输栏里不再出现裸的 Radix 按钮：那是"三套盒模型"的直接来源。
        expect(source).not.toMatch(/<IconButton[\s>]/);
        expect(source).not.toMatch(/<Button[\s>]/);
    });

    test("录音键不再被 Box 包裹，且固定实心危险色", async () => {
        const source = stripComments(await read("src/components/layout/ActionBar.tsx"));
        const recordAt = source.indexOf('tone="danger"');
        expect(recordAt).toBeGreaterThan(-1);
        // 录音键必须是**固定** fill：待机 soft / 录制中 solid 会让盒子变宽 1px。
        expect(source.slice(recordAt, recordAt + 400)).toContain('fill="solid"');
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
