/*
 * 上下文菜单样式模型的内部契约。
 *
 * 【与 `designSystemGates.test.ts` 的分工】那一组守的是**采纳**（每个菜单表面都
 * 挂共享壳、壳的外观不许被调用方重写、旧写法不许回来）；本文件守的是**模型自身**
 * 的形状：两套主题都要给出取值、悬停与键盘高亮必须同色、禁用态不许用不透明度、
 * 以及"建了类没人用"（抽象层空转）。
 *
 * 【为什么值得单独一组】上一轮的教训是"建了抽象层，没人用"—— 一个令牌/类如果
 * 没有消费者，改它对界面没有任何效果，而读注释的人会以为自己改一处就能改全局。
 * 这些断言让"模型是活的"成为可判定的事实。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const TOKEN_CSS = join("src", "index.css");

interface CssBlock {
    readonly selector: string;
    readonly body: string;
}

/** 把样式表切成规则块（已剔除注释）。与 `designSystemGates.test.ts` 同一做法。 */
function cssBlocks(source: string): CssBlock[] {
    const css = source.replace(/\/\*[\s\S]*?\*\//g, "");
    const blocks: CssBlock[] = [];
    const open: { selector: string; bodyStart: number }[] = [];
    let selectorStart = 0;
    for (let i = 0; i < css.length; i += 1) {
        const ch = css[i];
        if (ch === "{") {
            const raw = css.slice(selectorStart, i);
            const tail = raw.slice(Math.max(raw.lastIndexOf("}"), raw.lastIndexOf(";")) + 1);
            open.push({ selector: tail.trim(), bodyStart: i + 1 });
            selectorStart = i + 1;
        } else if (ch === "}") {
            const frame = open.pop();
            if (frame)
                blocks.push({ selector: frame.selector, body: css.slice(frame.bodyStart, i) });
            selectorStart = i + 1;
        }
    }
    return blocks;
}

const CSS = readFileSync(TOKEN_CSS, "utf8");
const BLOCKS = cssBlocks(CSS);

/** 菜单样式模型块（`.hs-menu` 及其元素/变体）。 */
function menuBlocks(): CssBlock[] {
    return BLOCKS.filter((block) => /(^|[\s,])\.hs-menu/.test(block.selector));
}

/** 某个规则块里某条声明的取值。 */
function declaredValue(block: CssBlock, property: string): string | null {
    const match = block.body.match(new RegExp(`(?:^|;)\\s*${property}\\s*:\\s*([^;]+)`));
    return match ? match[1].trim() : null;
}

function sourceFiles(re: RegExp): string[] {
    const out: string[] = [];
    const visit = (dir: string): void => {
        for (const entry of readdirSync(dir)) {
            const full = join(dir, entry);
            if (statSync(full).isDirectory()) {
                if (entry !== "node_modules") visit(full);
                continue;
            }
            if (/\.(test|bench)\.tsx?$/.test(entry)) continue;
            if (re.test(entry)) out.push(full);
        }
    };
    visit(join("src"));
    return out;
}

describe("菜单样式模型的令牌", () => {
    /*
     * 菜单的**颜色**令牌必须两套主题各给一份取值：只定义一套时，另一套主题下
     * 该令牌求值为空，`background: var(--qt-menu-item-hover)` 整条失效（回落到
     * `transparent`）—— 表现为"浅色主题下悬停没反应"，且不报任何错。
     */
    const COLOR_TOKENS = [
        "--qt-menu-item-hover",
        "--qt-menu-item-active",
        "--qt-menu-item-disabled",
    ];

    for (const theme of ["dark", "light"]) {
        test(`${theme} 主题定义了全部菜单颜色令牌`, () => {
            const blocks = BLOCKS.filter((block) =>
                block.selector.includes(`data-theme="${theme}"`),
            );
            expect(blocks.length, `找不到 ${theme} 主题的令牌块`).toBeGreaterThan(0);
            const body = blocks.map((block) => block.body).join("\n");
            const missing = COLOR_TOKENS.filter(
                (token) => !new RegExp(`${token}\\s*:\\s*\\S`).test(body),
            );
            expect(missing, `${theme} 主题缺少菜单颜色令牌`).toEqual([]);
        });
    }

    test("每个菜单令牌都有定义（没有悬空引用）", () => {
        const defined = new Set(CSS.match(/--qt-menu-[a-z0-9-]+(?=\s*:)/g) ?? []);
        const referenced = new Set(CSS.match(/var\(--qt-menu-[a-z0-9-]+/g) ?? []);
        const dangling = [...referenced]
            .map((ref) => ref.replace("var(", ""))
            .filter((token) => !defined.has(token));
        expect(dangling, "样式表引用了未定义的菜单令牌").toEqual([]);
    });
});

describe("菜单样式模型的形状", () => {
    test("壳的层级与最小宽度都取令牌", () => {
        const shell = menuBlocks().find((block) => block.selector === ".hs-menu");
        expect(shell, "找不到 `.hs-menu` 规则").toBeTruthy();
        expect(declaredValue(shell!, "z-index")).toBe("var(--qt-z-menu)");
        expect(declaredValue(shell!, "min-width")).toBe("var(--qt-menu-min-w)");
        expect(declaredValue(shell!, "max-height")).toBe("var(--qt-menu-max-h)");
    });

    test("悬停与键盘高亮是同一条规则", () => {
        /*
         * 拆成两条规则时，鼠标划过去与方向键走过去会长成两种视觉反馈 —— 键盘用户
         * 看到的是另一套界面。判据：同一个选择器列表里同时出现 `:hover` 与
         * `[data-active="1"]`，且声明块只有一处给 `background`。
         */
        const rule = menuBlocks().find(
            (block) => block.selector.includes(":hover") && block.selector.includes("[data-active"),
        );
        expect(
            rule,
            "`.hs-menu__item` 的悬停与键盘高亮必须在同一个选择器列表里（同色）",
        ).toBeTruthy();
        expect(declaredValue(rule!, "background")).toBe("var(--qt-menu-item-hover)");
    });

    test("禁用态用弱化色令牌，不用不透明度", () => {
        /*
         * 不透明度会同时削弱文字与底色的对比度，浅色主题下不可控。判据：存在一条
         * **禁用态**规则（选择器含 `:disabled` 且不含 `:hover` —— 悬停规则里的
         * `:not(:disabled)` 也含这个词），它给弱化色令牌；且没有任何菜单规则用
         * `opacity` 表达状态。
         */
        const disabled = menuBlocks().find(
            (block) =>
                block.selector.includes(":disabled") &&
                !block.selector.includes(":hover") &&
                declaredValue(block, "color") !== null,
        );
        expect(disabled, "找不到禁用态规则").toBeTruthy();
        expect(declaredValue(disabled!, "color")).toBe("var(--qt-menu-item-disabled)");
        for (const block of menuBlocks()) {
            expect(
                declaredValue(block, "opacity"),
                `${block.selector} 用不透明度表达状态 —— 请改用弱化色令牌`,
            ).toBeNull();
        }
    });

    test("分隔线两侧外边距对称（不把高度转嫁给相邻项）", () => {
        const separator = menuBlocks().find((block) => block.selector === ".hs-menu__separator");
        expect(separator, "找不到 `.hs-menu__separator` 规则").toBeTruthy();
        // `margin: <上> <右> <下> <左>` 的上与下必须相等。
        const margin = declaredValue(separator!, "margin");
        expect(margin, "分隔线必须有外边距").toBeTruthy();
        const parts = margin!.split(/\s+/);
        const [top, , bottom = top] = parts;
        expect(bottom, "分隔线上下的外边距必须相等，否则相邻项会比别的项高一截").toBe(top);
    });
});

describe("菜单样式模型不能空转", () => {
    test("每个 .hs-menu* 类都有真实消费者", () => {
        /*
         * 与"每个 barrel 导出都被 import"同一判据的类名版本：只认类名出现在
         * 源文件里（含 `src/ui` 自身 —— `AppContextMenu` 就是壳与项的第一个
         * 消费者）。豁免必须写明理由，且清单**当前为空**：一个菜单类如果暂时
         * 没人用，正确做法是删掉它，而不是先建着。
         */
        const NO_CONSUMER_ALLOWED: Record<string, string> = {};

        const classes = [
            ...new Set(
                (CSS.match(/\.hs-menu(?:__[a-z-]+|--[a-z-]+)?/g) ?? []).map((name) =>
                    name.slice(1),
                ),
            ),
        ];
        expect(classes.length, "没有找到任何 .hs-menu* 类，正则已失效").toBeGreaterThan(8);

        const files = sourceFiles(/\.tsx?$/);
        const sources = files.map((file) => readFileSync(file, "utf8"));
        const unadopted = classes.filter(
            (name) => !(name in NO_CONSUMER_ALLOWED) && !sources.some((s) => s.includes(name)),
        );
        expect(unadopted, "以下菜单类没有任何消费者 —— 抽象层在空转，删掉或接上：").toEqual([]);
    });
});

/*
 * 菜单项高亮的"键盘焦点"那一档必须是 `:focus-visible`，不能是 `:focus`。
 *
 * 【为什么要钉这一条】`:focus` 对**鼠标点击**也成立，而它不会自己消失：点过的
 * 那一项从此一直亮着，指针再划到别的项上就是**两条同时高亮**（实测：点「格式」
 * 打开子菜单后再把指针划进子面板，「格式」与子项一起亮；划过「全选」再划过
 * 一个二级触发项，也是两条）。`:focus-visible` 只在键盘交互导致的焦点上生效，
 * 鼠标点击不匹配 —— 而子面板的键盘导航正是靠它才看得见。
 *
 * 注意 `:focus-visible` 本身包含 `:focus` 子串，所以要按逗号切分后**整段**比较。
 */
test("菜单项高亮的键盘焦点用 :focus-visible 而不是 :focus", () => {
    const blocks = cssBlocks(readFileSync(TOKEN_CSS, "utf8"));
    const block = blocks.find(
        (entry) =>
            entry.body.includes("--qt-menu-item-hover") &&
            entry.selector.includes(".hs-menu__item"),
    );
    expect(block, "找不到菜单项的高亮规则").toBeDefined();
    const parts = block!.selector.split(",").map((part) => part.trim());
    expect(parts).toContain(".hs-menu__item:focus-visible");
    expect(parts).not.toContain(".hs-menu__item:focus");
});
