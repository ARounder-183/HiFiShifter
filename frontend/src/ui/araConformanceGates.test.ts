/*
 * ARA 插件合并带来的回归类别门禁。
 *
 * 【为什么单独一个文件，而不是并进 `designSystemGates.test.ts`】
 * 那一组门禁守的是"设计系统的**结构**事实"（原语有没有消费者、令牌有没有被绕开）。
 * 这里守的是合并 `e834ef54` 暴露出的**另一类**失败：新写的界面绕开既有体系 ——
 * 文案不走词表、表单控件绕开原语、尺寸写死在调色板上。两者失效原因不同，
 * 放在一起会让"哪条规则拦住了什么"变得难读。
 *
 * 【为什么既有门禁全都看不见这一类】实测：
 *   - `designSystemGates.test.ts` 只禁了 `<input type="checkbox">`，不认
 *     原生 `<select>` / `<input type="range">`；
 *   - 它的调色板规则只匹配 Tailwind 的固定色阶**类名**，看不见
 *     `var(--gray-6)` 这类内联的 Radix 内部变量；
 *   - 它的圆角规则要求值是**带引号的字符串**，`borderRadius: 4` 从缝里漏过；
 *   - `keyReferenceIntegrity.test.ts` 只校验"被引用的键存在"，**不校验文案
 *     是否走了键** —— 硬编码中文对它完全隐形。
 * 于是同一类写法可以整批合进 `develop` 而不触发任何一条红线。
 *
 * 【设计原则（沿用 `designSystemGates.test.ts`）】
 *   1. 匹配前先剥离注释 —— 本仓库的注释里大量引用"旧写法"，
 *      不剥离就会把说明文字当成真实调用（`ActionBar` 的注释里就写着
 *      `<input type="range">`）；
 *   2. 白名单是**已审计的记录**，每条必须写明理由，不是静音开关；
 *   3. 每条规则都断言扫描量下界 —— 扫描路径失效时门禁必须变红，而不是静默通过。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const UI_DIR = join("src", "ui");
const I18N_DIR = join("src", "i18n");

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

/** 待审的源文件：排除设计系统自身、词表目录与测试。 */
function consumerFiles(): string[] {
    return walk("src").filter(
        (file) =>
            !file.startsWith(UI_DIR) &&
            !file.startsWith(I18N_DIR) &&
            !/\.(test|bench)\.tsx?$/.test(file),
    );
}

/**
 * 剥离注释，**保留字符串字面量**。
 *
 * 【为什么这次保留字符串】文案门禁要找的正是字符串里的硬编码中文，剥掉就没得查了。
 * 但注释必须剥掉：本仓库的注释习惯引用"旧写法"，不剥就会把说明文字判成违规。
 *
 * 因此这里是一个完整的状态机，而不是逐行的正则 —— 字符串里的 `//`
 * （`"https://…"`）不能被当成行注释。
 */
function stripComments(source: string): string {
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
            if (ch === '"') state = "double";
            else if (ch === "'") state = "single";
            else if (ch === "`") state = "template";
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
            // 保留换行以维持行号（报告里要能定位）。
            if (ch === "\n") out += ch;
            i += 1;
            continue;
        }
        // 字符串内部：原样保留，处理转义。
        out += ch;
        i += 1;
        if (ch === "\\" && i < source.length) {
            out += source[i];
            i += 1;
            continue;
        }
        if (state === "double" && ch === '"') state = "code";
        else if (state === "single" && ch === "'") state = "code";
        else if (state === "template" && ch === "`") state = "code";
    }
    return out;
}

/** 扫描一份源文件，返回 `文件:行 命中片段` 形式的报告。 */
function scan(
    files: readonly string[],
    matcher: (source: string, file: string) => string[],
): { offenders: string[]; scanned: number } {
    const offenders: string[] = [];
    let scanned = 0;
    for (const file of files) {
        const source = stripComments(readFileSync(file, "utf8"));
        scanned += 1;
        offenders.push(...matcher(source, file).map((hit) => `${file}: ${hit}`));
    }
    return { offenders, scanned };
}

const CJK = /[\u4e00-\u9fff]/;

/**
 * 行内豁免标记。
 *
 * 【为什么需要行级豁免】少数文案**必须**是中文，最典型的是"字体预览" ——
 * 预览一款中文字体就得渲染中文字形。这类豁免挂在具体那一行上（而不是整个文件），
 * 这样同一个文件里新写的硬编码文案仍然会被拦下。
 */
const TEXT_EXEMPT_MARKER = "hs-text-exempt";

describe("文案：用户可见文本必须走词表", () => {
    /** 用户可见的 JSX 属性。`label` 也在内：它渲染成可见标签。 */
    const ATTR_RE =
        /\b(title|aria-label|placeholder|alt|label)\s*=\s*(?:"([^"]*)"|\{`([^`]*)`\}|\{"([^"]*)"\})/g;
    /** 单行 JSX 文本节点：`>纯文本<`，且不含引号（含引号的是字符串，不是文本节点）。 */
    const JSX_TEXT_RE = />\s*([^<>{}"'`]*[\u4e00-\u9fff][^<>{}"'`]*?)\s*</g;

    test("用户可见属性与 JSX 文本里没有硬编码中文", () => {
        const { offenders, scanned } = scan(consumerFiles(), (source, file) => {
            /*
             * 只看 `.tsx`：JSX 文本节点只可能出现在那里。
             * （`.ts` 里的 GLSL 源码字符串含有 `>` 与 `<`，逐行匹配会把它们
             * 当成文本节点 —— 实测误报过两处着色器。）
             */
            if (!file.endsWith(".tsx")) return [];
            const hits: string[] = [];
            for (const line of source.split("\n")) {
                if (line.includes(TEXT_EXEMPT_MARKER)) continue;
                for (const match of line.matchAll(ATTR_RE)) {
                    const value = match[2] ?? match[3] ?? match[4] ?? "";
                    if (CJK.test(value)) hits.push(`${match[1]}="${value}"`);
                }
                for (const match of line.matchAll(JSX_TEXT_RE)) {
                    if (CJK.test(match[1])) hits.push(`JSX 文本 "${match[1].trim()}"`);
                }
            }
            return hits;
        });
        expect(scanned, "没有扫到任何源文件，扫描路径已失效").toBeGreaterThan(100);
        expect(
            offenders,
            [
                "用户可见文案必须走 useI18n() 的 t()/tf()/tVars()，并补齐五个语系（见 docs/i18n/style-guide.md）。",
                "硬编码中文只在中文界面里正确 —— 其余四种语言会看到中文。",
                `确实必须保留中文时，在那一行加 ${TEXT_EXEMPT_MARKER} 标记并写明理由。`,
            ].join("\n"),
        ).toEqual([]);
    });
});

describe("表单控件：走设计系统原语", () => {
    /**
     * 允许直用 Radix `Button`/`IconButton`/`Select`/`TextField` 的文件。
     *
     * 这是一个**棘轮**：数量只能减不能增。把它降下来是受欢迎的改动；
     * 新增一个则必须在 code review 里被看见（并顺手改成原语）。
     *
     * 之所以不逐文件列白名单：现存 15 个是合并前就有的历史欠账，逐个写理由
     * 会把本文件的信噪比压垮，而它们与本轮要防的回归无关。
     */
    const RADIX_FORM_PRIMITIVE_BASELINE = 15;

    const RADIX_IMPORT_RE = /import\s*\{([^}]*)\}\s*from\s*["']@radix-ui\/themes["']/g;

    function importsFormPrimitive(source: string): boolean {
        for (const match of source.matchAll(RADIX_IMPORT_RE)) {
            const names = match[1].split(",").map((name) => name.trim().split(/\s+as\s+/)[0]);
            if (
                names.some((name) => ["Button", "IconButton", "Select", "TextField"].includes(name))
            ) {
                return true;
            }
        }
        return false;
    }

    test("直用 Radix 表单原语的文件数不超过基线", () => {
        const files = consumerFiles();
        const offenders = files.filter((file) =>
            importsFormPrimitive(stripComments(readFileSync(file, "utf8"))),
        );
        expect(files.length, "没有扫到任何源文件，扫描路径已失效").toBeGreaterThan(100);
        expect(
            offenders.length,
            [
                `已有 ${offenders.length} 个文件直用 Radix 的 Button/IconButton/Select/TextField，基线是 ${RADIX_FORM_PRIMITIVE_BASELINE}。`,
                "请改用 src/ui 的 AppButton / AppIconButton / AppSelect —— 它们带主题适配、尺寸体系与滚轮支持。",
                ...offenders,
            ].join("\n"),
        ).toBeLessThanOrEqual(RADIX_FORM_PRIMITIVE_BASELINE);
    });

    /**
     * 本轮改造过的表面：**零容忍**。
     *
     * 上面那条棘轮允许存量存在，这一条不允许这几个文件再退回去 ——
     * 它们正是"绕开原语"这类回归的原始出处。
     */
    const CONVERTED_SURFACES = [
        join("src", "features", "ara", "AraHostPanel.tsx"),
        join("src", "features", "ara", "PluginApplyStatus.tsx"),
        join("src", "components", "layout", "keybindings", "KeybindingsActionRow.tsx"),
    ];

    test("已改造的 ARA / 快捷键行表面不再直用 Radix 表单原语", () => {
        const offenders = CONVERTED_SURFACES.filter((file) =>
            importsFormPrimitive(stripComments(readFileSync(file, "utf8"))),
        );
        expect(offenders, "这些文件已改用 src/ui 原语，不得回退").toEqual([]);
    });

    test('没有裸 <select> 或裸 <input type="range">', () => {
        /*
         * 【为什么这两条单列】它们是"控件绕开原语"最直白的形式，而且既有门禁
         * 恰好只禁了 `<input type="checkbox">`（见 designSystemGates 的说明），
         * 于是 `<select>` 与 `<input type="range">` 成了唯一的缝。
         *
         * 白名单里的是**合并前既有**的写法：`App.tsx` 的滚轮下拉是一个给原生
         * select 补滚轮支持的局部包装，迁移它属于另一件事，不在本轮的回归类别里。
         */
        const ALLOWED: Record<string, string> = {
            [join("src", "App.tsx")]:
                "局部组件 WheelSelect：给原生 <select> 补滚轮换项，早于 AppSelect 存在。",
            [join("src", "components", "layout", "PianoRollPanel.tsx")]:
                "边缘平滑度的裸 range（带 .qt-range），早于 AppSlider；迁移要单独核对步长语义。",
            [join("src", "components", "layout", "timeline", "FadeContextMenu.tsx")]:
                "淡变曲率的裸 range，早于 AppSlider；迁移要单独核对步长语义。",
        };
        const files = consumerFiles().filter((file) => !(file in ALLOWED));
        const { offenders, scanned } = scan(files, (source) => {
            const hits: string[] = [];
            for (const _ of source.matchAll(/<select\b/g)) hits.push("<select>");
            for (const match of source.matchAll(/<input\b[^>]*?type\s*=\s*["'`]range["'`]/g))
                hits.push(`<input type="range"> (${match[0]})`);
            return hits;
        });
        expect(scanned, "没有扫到任何源文件，扫描路径已失效").toBeGreaterThan(100);
        expect(
            offenders,
            "请用 AppSelect / AppSlider —— 它们内建滚轮步进、精细调整修饰键与主题适配。",
        ).toEqual([]);
    });
});

describe("令牌：不绕过 --qt-* 取值", () => {
    test("没有数值型 borderRadius（应走 rounded-* / --qt-radius-*）", () => {
        /*
         * 既有门禁的圆角规则要求值是**带引号的字符串**（`borderRadius: "8px"`），
         * 于是 `borderRadius: 4` 这类数值写法从缝里漏过 —— 它同样绕开了
         * "圆角跟随用户的圆角风格设置"这件事。
         */
        const { offenders, scanned } = scan(consumerFiles(), (source) => {
            const hits: string[] = [];
            for (const match of source.matchAll(/borderRadius:\s*[0-9]+/g)) hits.push(match[0]);
            return hits;
        });
        expect(scanned, "没有扫到任何源文件，扫描路径已失效").toBeGreaterThan(100);
        expect(offenders, "数值圆角不随用户的圆角风格设置变化，请用 rounded-* 类。").toEqual([]);
    });

    test("没有内联的 Radix 内部调色板变量", () => {
        /*
         * `var(--gray-6)` / `var(--color-panel-solid)` 是 Radix Themes 的**内部**
         * 实现变量，不是本项目的令牌层。既有门禁的调色板规则只匹配 Tailwind 的
         * 固定色阶**类名**，这类内联写法对它隐形。
         *
         * 白名单里是合并前既有的一处：它是一条按物理像素对齐的竖分隔线，
         * 取 `--gray-8` 是刻意的可见度（`--qt-divider` 是给 1px 发丝线用的，
         * 换过去会改变观感）。改它属于视觉调整，不在本轮范围。
         */
        const ALLOWED: Record<string, string> = {
            [join("src", "components", "layout", "PianoRollPanel.tsx")]:
                "设备像素对齐的竖分隔线，刻意取 gray-8 的可见度。",
        };
        const files = consumerFiles().filter((file) => !(file in ALLOWED));
        const { offenders, scanned } = scan(files, (source) => {
            const hits: string[] = [];
            for (const match of source.matchAll(/var\(--(gray|color-panel)-[a-z0-9]+\)/g))
                hits.push(match[0]);
            return hits;
        });
        expect(scanned, "没有扫到任何源文件，扫描路径已失效").toBeGreaterThan(100);
        expect(
            offenders,
            "请用 --qt-* 语义令牌（见 src/index.css 的令牌块）—— 内部变量不跟随用户主题。",
        ).toEqual([]);
    });
});
