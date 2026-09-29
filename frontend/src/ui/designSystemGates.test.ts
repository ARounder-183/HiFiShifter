/*
 * 设计系统完整性门禁。
 *
 * 【为什么需要这一组】前几轮反复出现同一个失败模式：**建了原语，没接消费者**。
 * 曾经 20 个值导出零真实消费者，而它们要消除的重复一处未动 —— 于是每件事都有
 * 两套看起来都"官方"的写法，一致性净下降。
 *
 * 【为什么第一版门禁没拦住】它断言的是**文本事实**（名字在文件里出现过、旧写法
 * 没出现过），而不是**结构事实**（真的 import 并渲染了、真的没有别的写法）。
 * 文本事实能被注释、字符串、变体拼写绕过 —— 实测第一版把注释里的提及也算作
 * 消费者，并且只检查 barrel 导出，约 12 个导出对它是隐形的。
 *
 * 因此本文件：
 *   1. 消费者检查只看 **import 语句**（结构化），不再看整文件文本；
 *   2. 匹配前先**剥离注释与字符串**，再做跨行匹配；
 *   3. 断言"抽象层不能空转"（采用率下限 + 棘轮）。
 */
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const UI_DIR = join("src", "ui");
const SDK_BARREL = join("src", "sdk", "index.ts");

/**
 * 允许"暂时没有内部消费者"的导出。每条都必须写明理由。
 * 这份清单是**已审计的记录**，不是静音开关。
 */
const NO_CONSUMER_ALLOWED: Record<string, string> = {
    AppButton:
        "由 AppDialog 的页脚使用 —— 42 个对话框都经由它渲染按钮，属于壳内部件。" +
        "门禁只在 src/ui 之外找消费者，因此看不到这一层。",
};

/**
 * 允许使用 Tailwind 固定调色板的文件。每条都必须写明理由。
 * 与 `NO_CONSUMER_ALLOWED` 同一约定：这是**已审计的记录**，不是静音开关。
 *
 * 允许的都是"色相即语义"的域内表面 —— 颜色在这里表达类别（文件类型、电平档位），
 * 而不是主题角色，因此**必须**固定，不能跟随用户的强调色。
 */
const PALETTE_ALLOWED: Record<string, string> = {
    [join("src", "components", "layout", "FileBrowserPanel.tsx")]:
        "文件类型图标：色相即类型标识（文件夹 / 视频 / 音频 / 工程），不随主题变化。",
    [join("src", "components", "layout", "timeline", "TrackList.tsx")]:
        "电平表：色相即电平档位（削顶 / 过载 / 偏高 / 正常），必须固定，否则读数失去意义。",
};

function walk(dir: string, out: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            if (entry !== "node_modules") walk(full, out);
        } else if (/\.tsx?$/.test(entry)) {
            out.push(full);
        }
    }
    return out;
}

/** 非测试、非设计系统自身、非纯转发 barrel 的源文件。 */
function consumerFiles(): string[] {
    return walk("src").filter(
        (file) =>
            !file.startsWith(UI_DIR) &&
            file !== SDK_BARREL &&
            !/\.test\.tsx?$/.test(file) &&
            !/\.bench\.ts$/.test(file),
    );
}

/**
 * 采用率门禁的扫描范围：`src/ui` 之外、非测试的全部源文件，可按扩展名筛选。
 *
 * 与 `consumerFiles()` 的区别只在扩展名可控 —— 采用率要连 `.css` 一起看，
 * 因为"横条多高"这类取值在样式表里同样能被绕开。
 */
function sourceFiles(re: RegExp): string[] {
    const files: string[] = [];
    const visit = (dir: string): void => {
        for (const entry of readdirSync(dir)) {
            const full = join(dir, entry);
            if (statSync(full).isDirectory()) {
                if (entry !== "node_modules") visit(full);
                continue;
            }
            if (full.startsWith(UI_DIR)) continue;
            if (/\.(test|bench)\.tsx?$/.test(entry)) continue;
            if (re.test(entry)) files.push(full);
        }
    };
    visit("src");
    return files;
}

/**
 * 剥离注释与字符串字面量。
 *
 * 这一步是门禁可信的前提：`// 旧写法用 window.alert(...)` 这类说明性文字
 * 不应被当成真实调用，而 `"<input type=\"checkbox\">"` 这类字符串同理。
 */
function stripCommentsAndStrings(source: string, keepStrings = false): string {
    let out = "";
    let i = 0;
    let state: "code" | "line" | "block" | "single" | "double" | "template" = "code";
    while (i < source.length) {
        const ch = source[i];
        const next = source[i + 1];
        if (state === "code") {
            if (ch === "/" && next === "/") {
                state = "line";
                i += 2;
                continue;
            }
            if (ch === "/" && next === "*") {
                state = "block";
                i += 2;
                continue;
            }
            if (ch === '"') {
                state = "double";
                if (keepStrings) out += ch;
                i += 1;
                continue;
            }
            if (ch === "'") {
                state = "single";
                if (keepStrings) out += ch;
                i += 1;
                continue;
            }
            if (ch === "`") {
                state = "template";
                if (keepStrings) out += ch;
                i += 1;
                continue;
            }
            out += ch;
            i += 1;
            continue;
        }
        if (state === "line") {
            if (ch === "\n") {
                state = "code";
                out += ch;
            }
            i += 1;
            continue;
        }
        if (state === "block") {
            if (ch === "*" && next === "/") {
                state = "code";
                i += 2;
                continue;
            }
            if (ch === "\n") out += ch; // 保留行号
            i += 1;
            continue;
        }
        // 字符串内：保留换行以维持行号；`keepStrings` 时原样保留（类名等字面量在字符串里）
        if (keepStrings) {
            out += ch;
            i += 1;
            if (ch === "\\") {
                if (i < source.length) out += source[i];
                i += 1;
                continue;
            }
            if (
                (state === "double" && ch === '"') ||
                (state === "single" && ch === "'") ||
                (state === "template" && ch === "`")
            ) {
                state = "code";
            }
            continue;
        }
        if (ch === "\\") {
            i += 2;
            continue;
        }
        if ((state === "double" && ch === '"') || (state === "single" && ch === "'")) {
            state = "code";
            i += 1;
            continue;
        }
        if (state === "template" && ch === "`") {
            state = "code";
            i += 1;
            continue;
        }
        if (ch === "\n") out += ch;
        i += 1;
    }
    return out;
}

/** 样式表路径（颜色令牌与主题作用域的唯一真值源）。 */
const TOKEN_CSS = join("src", "index.css");

/** 样式表里的一个规则块（`选择器 { 体 }`；`context` 为祖先选择器链，便于报错定位）。 */
interface CssBlock {
    readonly selector: string;
    readonly context: string;
    readonly body: string;
}

/**
 * 把样式表切成规则块（已剔除注释）。
 *
 * 【为什么需要】颜色令牌的**求值作用域**是结构事实，只有解析出"某条声明写在哪个
 * 选择器里"才能断言它 —— 正则扫全文做不到这件事。
 *
 * 特殊说明：`index.css` 里没有 `@media` / `@supports`（本函数仍按栈处理嵌套，
 * 因此将来加了也能正确给出祖先链）。
 */
function cssBlocks(source: string): CssBlock[] {
    const css = source.replace(/\/\*[\s\S]*?\*\//g, "");
    const blocks: CssBlock[] = [];
    const open: { selector: string; bodyStart: number; chain: string }[] = [];
    let selectorStart = 0;
    for (let i = 0; i < css.length; i += 1) {
        const ch = css[i];
        if (ch === "{") {
            const raw = css.slice(selectorStart, i);
            // 只取"上一个块/语句之后"的文本：`@tailwind …;` 这类语句与空行不属于选择器。
            const tail = raw.slice(Math.max(raw.lastIndexOf("}"), raw.lastIndexOf(";")) + 1);
            const selector = tail.trim();
            const parent = open.length > 0 ? open[open.length - 1] : null;
            const chain = parent ? `${parent.chain} > ${selector}` : selector;
            open.push({ selector, bodyStart: i + 1, chain });
            selectorStart = i + 1;
        } else if (ch === "}") {
            const frame = open.pop();
            if (frame) {
                blocks.push({
                    selector: frame.selector,
                    context: frame.chain,
                    body: css.slice(frame.bodyStart, i),
                });
            }
            selectorStart = i + 1;
        }
    }
    return blocks;
}

/** 取从 `open`（指向 `var(` 的左括号）起与之配对的右括号之间的文本。 */
function varArgsAt(value: string, open: number): string {
    let depth = 0;
    for (let i = open; i < value.length; i += 1) {
        const ch = value[i];
        if (ch === "(") depth += 1;
        else if (ch === ")") {
            depth -= 1;
            if (depth === 0) return value.slice(open + 1, i);
        }
    }
    return "";
}

/** 值里**任意**一个 `var()` 是否提供了 fallback（形参顶层出现逗号）。 */
function hasVarFallback(value: string): boolean {
    let index = value.indexOf("var(");
    while (index >= 0) {
        const args = varArgsAt(value, index + 3);
        let depth = 0;
        for (const ch of args) {
            if (ch === "(") depth += 1;
            else if (ch === ")") depth -= 1;
            else if (ch === "," && depth === 0) return true;
        }
        index = value.indexOf("var(", index + 1);
    }
    return false;
}

/** 从 barrel 解析值导出名（跳过 `type`）。 */
function barrelValueExports(): string[] {
    const source = readFileSync(join(UI_DIR, "index.ts"), "utf8");
    const names = new Set<string>();
    for (const match of source.matchAll(/export\s*\{([\s\S]*?)\}\s*from/g)) {
        for (const raw of match[1].split(",")) {
            const name = raw.trim();
            if (!name || name.startsWith("type ") || name.startsWith("//")) continue;
            names.add(name);
        }
    }
    for (const match of source.matchAll(/export\s+(?:const|function|class)\s+(\w+)/g)) {
        names.add(match[1]);
    }
    return [...names].sort();
}

/** 某个文件从 ui 路径 import 了哪些名字（含 `import * as ns` 的命名空间）。 */
function uiImportsOf(source: string): { names: Set<string>; namespaces: Set<string> } {
    const names = new Set<string>();
    const namespaces = new Set<string>();
    /*
     * 子句用 `[^;"']*?` 而不是 `[\s\S]*?`：后者会**跨过语句边界**，把
     * `import { a } from "react";\nimport { b }` 整段当成一个子句，于是路径
     * 匹配到后面的 ui 导入、名字却取成了前一条 import 的 —— 门禁会因此
     * 把已被正常 import 的原语误报为"无人使用"。
     */
    /*
     * 路径要同时接受 barrel（`../../ui`）与深层模块（`../../ui/Dialog`）——
     * 后者是仓库里的常见写法，只匹配 barrel 会把它们全部误报为"无人使用"。
     * `/ui` 作为**路径段**足够特异，不会误伤 `utils` 之类。
     */
    const re = /import\s+([^;"']*?)\s+from\s+["']([^"']*\/ui(?:\/[^"']*)?)["']/g;
    let match;
    while ((match = re.exec(source)) !== null) {
        const clause = match[1];
        const ns = clause.match(/\*\s+as\s+(\w+)/);
        if (ns) namespaces.add(ns[1]);
        const braced = clause.match(/\{([\s\S]*)\}/);
        if (braced) {
            for (const raw of braced[1].split(",")) {
                const name = raw
                    .replace(/^\s*type\s+/, "")
                    .trim()
                    .split(/\s+as\s+/)[0];
                if (name) names.add(name);
            }
        }
        const def = clause.match(/^\s*(\w+)\s*(?:,|$)/);
        if (def && !clause.includes("{")) names.add(def[1]);
    }
    return { names, namespaces };
}

describe("原语必须有消费者（结构性检查）", () => {
    test("每个 barrel 值导出都被真实 import 使用", () => {
        const consumed = new Set<string>();
        for (const file of consumerFiles()) {
            /*
             * 注意：import 解析必须用**原始源码**，不能用剥离后的 ——
             * 剥离会连模块路径字符串一起删掉，`from "../../ui"` 会变成 `from ;`，
             * 于是所有原语都被误报为"无人使用"。
             * 注释掉的 import 属于极少数情况，不值得为它牺牲正确性。
             */
            const source = readFileSync(file, "utf8");
            const { names, namespaces } = uiImportsOf(source);
            const stripped = stripCommentsAndStrings(source);
            for (const name of names) consumed.add(name);
            // `import * as ui` → 认 `ui.Name` 这种成员访问
            for (const ns of namespaces) {
                for (const name of barrelValueExports()) {
                    if (new RegExp(`\\b${ns}\\.${name}\\b`).test(stripped)) consumed.add(name);
                }
            }
        }

        const unadopted = barrelValueExports().filter(
            (name) => !(name in NO_CONSUMER_ALLOWED) && !consumed.has(name),
        );
        expect(
            unadopted.length === 0
                ? []
                : [
                      "以下原语没有任何 import 消费者 —— 要么接上，要么删除：",
                      ...unadopted.map((name) => `  ${name}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    test("注释与字符串里的提及**不算**消费者", () => {
        // 防止门禁退化成"文本匹配"：一个只在注释里出现的名字不应通过
        const fake = `// AppText is great\nconst s = "AppText";\n`;
        const stripped = stripCommentsAndStrings(fake);
        expect(stripped).not.toContain("AppText");
    });

    test("豁免清单里每条都写了理由", () => {
        for (const [name, reason] of Object.entries(NO_CONSUMER_ALLOWED)) {
            expect(reason.length, `${name} 的豁免理由过短`).toBeGreaterThan(20);
        }
    });
});

/*
 * 采用率门禁（G2）。
 *
 * 【为什么必须有】前几轮的失败不是"没建抽象层"，而是"建了抽象层，没人用"：
 * `--qt-bar-main` 只在一个组件的映射表里躺着，改它在 5 条横条上没有任何效果；
 * 排版角色全应用只有个位数消费者。抽象层空转比没有抽象层更糟 —— 它让读者
 * 以为自己改一处就能改全局，实际改不动。
 *
 * 【为什么用棘轮而不是一步到位的下限】`<Text size="N">` 那 130 处迁移要逐处
 * 判断语义（标签 / 说明 / 正文）并做视觉验证，不是机械替换能完成的。因此本轮
 * 先把**能确定的事实**钉死（字号不许再有裸值、横条令牌必须有消费者、类型旁路
 * 归零），再给排版角色留一个只增不减的棘轮：数字写死在下面，涨上去以后就不许
 * 跌回来。
 */
describe("抽象层不能空转（采用率）", () => {
    test("每个 --qt-bar-* 档位都至少有一个真实消费者", () => {
        const tiers = ["main", "title", "compact", "status"];
        const files = sourceFiles(/\.(tsx?|css)$/);
        const missing: string[] = [];
        for (const tier of tiers) {
            // 只认"用掉"的写法：Tailwind 高度类，或 CSS 里读该变量。
            // `--qt-bar-x: 32px` 这种定义本身不算消费者，因此不匹配。
            const re = new RegExp(`(?:h-qt-bar-${tier}\\b|var\\(--qt-bar-${tier}\\))`);
            if (!files.some((file) => re.test(readFileSync(file, "utf8")))) {
                missing.push(`--qt-bar-${tier}`);
            }
        }
        expect(
            missing.length === 0
                ? []
                : [
                      "以下横条令牌没有任何消费者 —— 令牌层在空转，删掉或接上：",
                      ...missing.map((name) => `  ${name}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    test("字号只从阶梯取，没有裸 px 字面量", () => {
        const offenders: string[] = [];
        for (const file of sourceFiles(/\.(tsx?|css)$/)) {
            const source = readFileSync(file, "utf8");
            if (file.endsWith(".css")) {
                /*
                 * 只查绝对 px。`font-size: 1.5em` 是记事本文档自己的比例尺
                 * （相对用户的笔记字号设置），不是"各写各的绝对量"那类问题。
                 */
                const hits = source.match(/font-size:\s*[\d.]+px/g) ?? [];
                if (hits.length > 0) offenders.push(`${file}: ${hits.length} 处`);
                continue;
            }
            const stripped = stripCommentsAndStrings(source);
            const hits = [
                ...(stripped.match(/text-\[\d+px\]/g) ?? []),
                ...(stripped.match(/fontSize:\s*\d+(?![\d.])/g) ?? []),
            ];
            if (hits.length > 0) offenders.push(`${file}: ${hits.length} 处`);
        }
        expect(
            offenders,
            "字号请取 `text-qt-*` 工具类或 `var(--qt-fs-*)`（见 src/index.css 的字号阶梯）——" +
                "各写各的 px 会让阶梯失效，也会让 9px 与 10px 这种差异无人记录",
        ).toEqual([]);
    });

    test("圆角只从语义令牌取，没有裸 px 字面量", () => {
        /*
         * 【为什么禁止】圆角是**外观设置的一部分**（视图 → 外观设置 → 圆角风格，
         * 五档）。写死 px 的圆角不跟随它 —— 用户报告的"参数编辑器工具栏的参数胶囊
         * 没有适配圆角风格"就是这么来的：那条规则写的是 `border-radius: 6px`。
         * 采集时全仓 30 余处写死值（对话框 10px、菜单项 7px、提示 6px、参数胶囊 6px、
         * 停靠浮窗 4px…），全都替换成了 `var(--qt-radius-*)`。
         *
         * 【豁免】圆形与个别"形状"必须是字面量，逐条列在下面：
         *   - `50%` / `9999px`：圆点、滑块拇指、开关轨道是**形状**而不是风格；
         *   - 停靠浮窗最大化时的 `0`：贴边展开，任何圆角都会露出背景；
         *   - 不对称角（如 `2px 0 2px 0`）：那是装饰性缺口，不是圆角档位；
         *   - 记事本导出文档里的内联样式：它是一份**独立 HTML**，与运行中的应用
         *     不共享令牌（导出后拿到别处打开也要正常）。
         */
        const ALLOWED_FILES = new Set([
            // 导出的独立 HTML 模板（自带一套固定样式）。
            join("src", "components", "layout", "notebook", "NotebookDialogs.tsx"),
            // 圆角档位的**示意**磁贴：它画的正是"每个选项长什么样"，因此不能取当前值。
            join("src", "components", "layout", "AppearanceSettingsPanel.tsx"),
        ]);
        const offenders: string[] = [];
        for (const file of sourceFiles(/\.(tsx?|css)$/)) {
            if (ALLOWED_FILES.has(file)) continue;
            const source = readFileSync(file, "utf8");
            const literals: string[] = [];
            if (file.endsWith(".css")) {
                for (const match of source.matchAll(/border-radius:\s*([^;]+);/g)) {
                    literals.push(match[1].trim());
                }
            } else {
                const stripped = stripCommentsAndStrings(source);
                for (const match of stripped.matchAll(/borderRadius:\s*(?:"([^"]*)"|'([^']*)')/g)) {
                    literals.push((match[1] ?? match[2] ?? "").trim());
                }
            }
            // 只保留"纯 px 数值"：`var(--qt-radius-*)`、`50%`、`0`、`9999px` 都不算。
            const numeric = literals.filter((value) => /^[\d.]+px(\s+[\d.]+px)*$/i.test(value));
            if (numeric.length > 0) offenders.push(`${file}: ${numeric.join(" / ")}`);
        }
        expect(
            offenders,
            "圆角请取 `var(--qt-radius-sm|md|lg|pill)`（或 `rounded-qt-*`）——" +
                "写死 px 的圆角不会跟随「圆角风格」设置；确属形状/独立文档的加进本测试的豁免清单",
        ).toEqual([]);
    });

    test("tf 有真实消费者（无类型翻译器不能是摆设）", () => {
        const count = sourceFiles(/\.tsx?$/).reduce((total, file) => {
            const stripped = stripCommentsAndStrings(readFileSync(file, "utf8"));
            return total + (stripped.match(/\btf\(/g)?.length ?? 0);
        }, 0);
        expect(count).toBeGreaterThan(0);
    });

    test("不得再出现 `as (key: string) => string` 类型旁路（棘轮：基线 0）", () => {
        /*
         * 这个 cast 是在告诉编译器"别管键名"，于是一个拼错的键名会一路走到
         * 运行时才变成界面上的原始英文键。`tf` 用键名字面量联合类型消除了它，
         * 这里禁止它回来。
         */
        const offenders: string[] = [];
        for (const file of sourceFiles(/\.tsx?$/)) {
            const stripped = stripCommentsAndStrings(readFileSync(file, "utf8"));
            const hits = stripped.match(/as\s*\(\s*key:\s*string\s*\)\s*=>\s*string/g) ?? [];
            if (hits.length > 0) offenders.push(`${file}: ${hits.length} 处`);
        }
        expect(offenders, "请改用 `tf`（键名有类型，拼错会在编译期报错）").toEqual([]);
    });

    test("颜色只从语义令牌取（域内固定色相有豁免清单）", () => {
        /*
         * 【为什么禁止 Tailwind 固定调色板】`bg-gray-700` / `text-blue-600` 这类
         * 取值不跟随主题：浅色主题下它们要么对比度不足、要么和周围 chrome 脱节，
         * 用户在外观设置里换主题也影响不到它们。本仓库为此有整套
         * `--qt-*` 语义色（含 danger / warning / success / info 四组）。
         *
         * 【豁免的是什么】有两处色相**必须**固定，因为它们承载的是"类别"而不是
         * "主题语义"：文件类型图标（按类型分色）与电平表（按电平分色）。电平表
         * 若跟随用户的强调色，就再也读不出"这一段是不是要削顶了"。
         */
        const PALETTE =
            /\b(?:text|bg|border|from|to|via|ring|outline|fill|stroke|divide|placeholder|decoration|caret|shadow)-(?:gray|slate|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose)-\d{2,3}\b/g;

        const offenders: string[] = [];
        for (const file of sourceFiles(/\.tsx?$/)) {
            if (file in PALETTE_ALLOWED) continue;
            const source = stripCommentsAndStrings(readFileSync(file, "utf8"));
            const hits = source.match(PALETTE) ?? [];
            if (hits.length > 0) offenders.push(`${file}: ${[...new Set(hits)].join(", ")}`);
        }
        expect(
            offenders,
            "请改用 `qt-*` 语义色（danger/warning/success/info/text/text-muted/border…）。" +
                "确实需要固定色相时，把文件加进本测试的 ALLOWED 并写明理由",
        ).toEqual([]);
    });

    test("调色板豁免清单里每条都写了理由，且指向真实文件", () => {
        for (const [name, reason] of Object.entries(PALETTE_ALLOWED)) {
            expect(reason.length, `${name} 的豁免理由过短`).toBeGreaterThan(20);
            expect(existsSync(name), `豁免清单指向了不存在的文件：${name}`).toBe(true);
        }
    });

    test("派生令牌在 portal 安全的作用域声明，或自带 fallback", () => {
        /*
         * 【为什么需要这条】`--qt-accent: var(--accent-9)` 这类"值引用其它自定义属性"
         * 的令牌，其解析发生在**声明它的元素**上：声明在 `:root` 时，`<html>` 上并没有
         * `--accent-9`（Radix 只把它定义在 `.radix-themes` 上），于是该令牌计算为
         * **无效值**并被后代按无效值继承。应用外壳恰好还有一个更近的声明（外壳元素同时
         * 带 `.qt-theme` 与 `.radix-themes`）所以看不出问题 —— 而 Dialog / Select /
         * Tooltip 等 Radix portal 的内容挂在 `<body>` 下、只有 `radix-themes`，拿到的
         * 就是无效值：`bg-qt-accent` 解析成 `transparent`。
         *
         * 【实测症状】导出进度条的填充条一直不可见（它是全仓唯一使用 `bg-qt-accent`
         * 的进度组件，且唯一使用点就在对话框里）；`AboutDialog` 的链接色、
         * `SegmentedControl` 的激活底色在对话框内同样失效。
         *
         * 【判据】要么声明在与来源变量相同的作用域（选择器含 `.radix-themes` —— 每个
         * Radix portal 包裹层都有这个类），要么给 `var()` 提供 fallback。
         */
        const offenders: string[] = [];
        for (const block of cssBlocks(readFileSync(TOKEN_CSS, "utf8"))) {
            const declaration = /--qt-[a-z0-9-]+:\s*([^;]+);/g;
            let match: RegExpExecArray | null;
            while ((match = declaration.exec(block.body)) !== null) {
                const value = match[1];
                // 字面色（不引用其它自定义属性）在任何作用域都有效。
                if (!value.includes("var(")) continue;
                if (hasVarFallback(value)) continue;
                if (/(^|[\s,])\.radix-themes([\s,:.]|$)/.test(block.selector)) continue;
                offenders.push(`${block.context} → ${match[0].trim()}`);
            }
        }
        expect(
            offenders,
            "派生令牌必须在 `.radix-themes` 作用域声明（该类的元素上才有 --accent-9），" +
                "或给 var() 写 fallback；否则 portal 内容里会解析成 transparent",
        ).toEqual([]);
    });

    test("没有「有界滚动盒」造成的嵌套滚动条", () => {
        /*
         * 【为什么禁止】`max-h-[240px] overflow-y-auto` 这种写法等于在页面/对话框里
         * 再嵌一个滚动区。用户已经报过两次同一个现象：**两层竖直滚动条，而里面那条
         * 滚下去什么也看不到**（内层内容其实没有溢出，只是它的 `max-h` 比可用高度小）。
         *
         * 正确做法是让这一层参与外层的 flex 布局（`min-h-0 flex-1`），使外层永不溢出，
         * 只留一条滚动条 —— 对话框的滚动契约见 `src/index.css` 的 `.app-dialog` 注释。
         *
         * 允许清单里的是**例外且有意**的一处：设置页里的字体列表。外层是页面滚动、
         * 内层是有界的列表滚动，两条都各自有用（见该处注释）。
         */
        const ALLOWED = new Set([
            join("src", "components", "layout", "AppearanceSettingsPanel.tsx"),
        ]);
        const BOUNDED_SCROLL =
            /["'`](?=[^"'`]*max-h-\[\d+px\])(?=[^"'`]*overflow-(?:x|y)?-auto)[^"'`]*["'`]/g;

        const offenders: string[] = [];
        for (const file of sourceFiles(/\.tsx?$/)) {
            if (ALLOWED.has(file)) continue;
            // 保留字符串（类名在字符串里）、剥掉注释（否则解释性文字会被误判）。
            const source = stripCommentsAndStrings(readFileSync(file, "utf8"), true);
            const hits = source.match(BOUNDED_SCROLL) ?? [];
            if (hits.length > 0) offenders.push(`${file}: ${hits.length} 处`);
        }
        expect(
            offenders,
            "有界滚动盒会和外层滚动条叠成两层。请改为 `min-h-0 flex-1` 参与外层 flex 布局，" +
                "让外层不溢出（确需保留两层时加进本测试的 ALLOWED 并写明理由）",
        ).toEqual([]);
    });

    test("排版角色在 src/ui 之外的采用率只增不减（棘轮）", () => {
        /*
         * 【目标与现状】目标是 ≥ 50（把 130 处 `<Text size="N">` 收敛到角色层）。
         * 实测基线从 **4** 涨到 **12**：对话框标题复用 `.hs-type-display`、
         * 主消息复用 `.hs-type-body`、外观设置的区块标签改用 `.hs-type-label`，
         * 并且这三处原本都在"自己写一份声明"。
         *
         * 【2026-09-28 回落到 9】重排外观设置面板时，7 处节标签**迁入了
         * `AppFormSection`**（`src/ui`，本统计按设计不计原语内部）。消费者没有
         * 消失 —— 而是变成了原语内部的一份，被 `AppFormSection` 的每一位使用者
         * 共享。这是把"调用点各写一份"换成"原语集中一份"的正常落点，因此下限
         * 如实钉在回落后的 **9**。原语的内部采用由 `Field.tsx` 自身的声明承担，
         * 不需要这条统计来守。
         *
         * 【为什么本轮没做那个迁移】`<Text size>` 不能一对一替换成角色类：
         * Radix 的 `size="2"` 是 14px/24px，而设计系统的正文字号是 13px/20px；
         * `size="1"` 是 12px/16px，而 `hs-type-label` 是 12px/18px 且**强制**
         * `--qt-text` 颜色（Radix 默认继承 currentColor）。也就是说这 130 处
         * 每一处都要判断"这是标签、说明还是正文"，并接受行高与颜色的变化 ——
         * 属于一次全应用排版调整，必须逐屏截图核对。
         *
         * 【为什么不做一半】部分迁移会留下两套看起来都"官方"的写法，一致性净下降
         * —— 那正是本文件开头记下的失败模式。所以这里只钉住水位：涨了不管，
         * 跌回来就红（例如有人把角色层唯一的消费者删掉）。
         */
        const floor = 9;
        const count = sourceFiles(/\.tsx?$/).reduce((total, file) => {
            /*
             * 这里必须**保留字符串**：`hs-type-*` 是写在 `className="..."` 里的
             * 类名，剥掉字符串就等于把全部消费者删掉（实测会得到 0）。
             * 注释仍然剥掉，否则文档里提一句就算采用。
             */
            const source = stripCommentsAndStrings(readFileSync(file, "utf8"), true);
            return total + (source.match(/hs-type-[a-z]+/g)?.length ?? 0);
        }, 0);
        expect(count, "排版角色的采用率跌回了本轮基线以下").toBeGreaterThanOrEqual(floor);
    });
});

describe("禁止浏览器原生 affordance", () => {
    test("没有 window.alert / confirm（跨行、忽略注释与字符串）", () => {
        const offenders: string[] = [];
        for (const file of consumerFiles()) {
            if (!/\.tsx?$/.test(file)) continue;
            const source = stripCommentsAndStrings(readFileSync(file, "utf8"));
            if (/(?<![\w.])(?:window\s*\.\s*)?\b(?:alert|confirm)\s*\(/.test(source)) {
                offenders.push(file);
            }
        }
        expect(
            offenders,
            "桌面应用不该弹浏览器原生弹窗，请用 AppNoticeDialog / AppConfirmDialog",
        ).toEqual([]);
    });

    test('没有裸 <input type="checkbox">（跨行、覆盖 .ts 与 .tsx）', () => {
        const offenders: string[] = [];
        for (const file of consumerFiles()) {
            const source = stripCommentsAndStrings(readFileSync(file, "utf8"));
            const matches =
                source.match(/<input\b[^>]*?type\s*=\s*\{?\s*["'`]checkbox["'`]/g) ?? [];
            if (matches.length > 0) offenders.push(`${file}: ${matches.length} 处`);
        }
        expect(
            offenders,
            '请用 AppSwitchRow（control="checkbox"）—— 系统默认复选框与 Radix 控件高度不一致',
        ).toEqual([]);
    });
});
