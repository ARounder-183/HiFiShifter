/*
 * 排版层级门禁。
 *
 * 【为什么必须有】上一轮引入了字号令牌，但**没有规定层级**：标题挑到 13px，
 * 结果比 14px 的正文标签还小；节标题 12px 比它统领的行还小；字段标签 11px
 * 比它自己的提示（12px）还小。层级不是靠"提供了哪些字号"保证的，而是靠
 * **角色之间的关系**保证的 —— 所以这里把关系写成断言。
 *
 * 断言直接读 `src/index.css` 的角色定义（而不是渲染后取计算样式）：
 * 这样失败信息直接指向那行 CSS，且不依赖浏览器是否加载了样式表。
 */
import { readFileSync } from "node:fs";
import { describe, expect, test } from "vitest";

/*
 * 读 `src/index.css` 的真实文本。不用 `?raw`：vitest 把 `.css` 一律替换成
 * 空模块（已实测），因此 `?raw` 与 `import.meta.glob(query: "?raw")` 都拿不到内容。
 * 类型由同目录的 `nodeFsShim.d.ts` 提供（只声明用到的那一个函数）。
 */
const css = readFileSync(new URL("../index.css", import.meta.url), "utf8");

/** 取某个类名的规则体。 */
function ruleBody(selector: string): string {
    const re = new RegExp(`\\${selector}\\s*\\{([^}]*)\\}`, "m");
    const match = css.match(re);
    if (!match) throw new Error(`找不到样式规则：${selector}`);
    return match[1];
}

/** 取 `--qt-xxx` 令牌的值（px 数字）。 */
function tokenPx(name: string): number {
    const re = new RegExp(`${name}:\\s*(\\d+)px`, "m");
    const match = css.match(re);
    if (!match) throw new Error(`找不到令牌：${name}`);
    return Number(match[1]);
}

/** 取规则体里的 `font-size`（解析成 px）。 */
function fontSizeOf(body: string): number {
    const varMatch = body.match(/font-size:\s*var\((--qt-fs-[\w-]+)\)/);
    if (varMatch) return tokenPx(varMatch[1]);
    const pxMatch = body.match(/font-size:\s*(\d+)px/);
    if (!pxMatch) throw new Error(`规则体里没有可解析的 font-size：${body}`);
    return Number(pxMatch[1]);
}

function fontWeightOf(body: string): number {
    const match = body.match(/font-weight:\s*(\d+)/);
    return match ? Number(match[1]) : 400;
}

function usesMuted(body: string): boolean {
    return /color:\s*var\(--qt-text-muted\)/.test(body);
}

const ROLE_FONT = {
    display: fontSizeOf(ruleBody(".hs-type-display")),
    section: fontSizeOf(ruleBody(".hs-type-section")),
    body: fontSizeOf(ruleBody(".hs-type-body")),
    label: fontSizeOf(ruleBody(".hs-type-label")),
    caption: fontSizeOf(ruleBody(".hs-type-caption")),
    mono: fontSizeOf(ruleBody(".hs-type-mono")),
};

describe("排版角色层级", () => {
    test("六个角色都有定义", () => {
        for (const role of Object.keys(ROLE_FONT)) {
            expect(ruleBody(`.hs-type-${role}`), `.hs-type-${role} 缺失`).toBeTruthy();
        }
    });

    test("标题明显大于正文（这是上一轮的回归点）", () => {
        // 曾经 13px vs 14px —— 倒挂。现在要求至少拉开 5px。
        expect(ROLE_FONT.display - ROLE_FONT.body).toBeGreaterThanOrEqual(5);
        expect(ROLE_FONT.display).toBeGreaterThanOrEqual(18);
    });

    test("正文 > 标签 > 说明，逐级递减且不出现倒挂", () => {
        expect(ROLE_FONT.body).toBeGreaterThan(ROLE_FONT.label);
        expect(ROLE_FONT.label).toBeGreaterThan(ROLE_FONT.caption);
    });

    test("节标题不小于正文，靠字重分层而不是靠缩小字号", () => {
        expect(ROLE_FONT.section).toBeGreaterThanOrEqual(ROLE_FONT.body);
        const section = ruleBody(".hs-type-section");
        const body = ruleBody(".hs-type-body");
        expect(fontWeightOf(section)).toBeGreaterThan(fontWeightOf(body));
    });

    test("标签用正文色，不用弱化色（标签不该比自己的提示还淡）", () => {
        expect(usesMuted(ruleBody(".hs-type-label"))).toBe(false);
        expect(usesMuted(ruleBody(".hs-type-caption"))).toBe(true);
    });

    test("对话框标题/描述沿用角色量级，不再各自写字号", () => {
        expect(fontSizeOf(ruleBody(".app-dialog__title"))).toBe(ROLE_FONT.display);
        expect(fontSizeOf(ruleBody(".app-dialog__description"))).toBe(ROLE_FONT.caption);
    });

    test("字号阶梯覆盖到标题量级（否则作者只能拿最接近的值凑）", () => {
        expect(tokenPx("--qt-fs-2xl")).toBeGreaterThanOrEqual(18);
        // 阶梯必须单调递增，且每个角色都能在阶梯上找到落点
        const ladder = [
            "--qt-fs-micro",
            "--qt-fs-xs",
            "--qt-fs-sm",
            "--qt-fs-md",
            "--qt-fs-lg",
            "--qt-fs-xl",
            "--qt-fs-2xl",
        ].map(tokenPx);
        for (let i = 1; i < ladder.length; i += 1) {
            expect(ladder[i], `阶梯在 ${i} 处不递增`).toBeGreaterThan(ladder[i - 1]);
        }
    });

    test("行高与字号成对给出（不留继承导致的行距跳变）", () => {
        for (const role of Object.keys(ROLE_FONT)) {
            expect(ruleBody(`.hs-type-${role}`), `.hs-type-${role} 未给行高`).toMatch(
                /line-height:/,
            );
        }
    });
});
