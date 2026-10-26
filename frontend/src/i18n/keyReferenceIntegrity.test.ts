/*
 * i18n **键引用**完整性门禁。
 *
 * 【为什么必须有】上一轮把 29 个裸键改名为 `common_*` 时，改写的是**调用点**
 * （`t("notebook")` → `t("common_notebook")`），没有覆盖**字符串字面量引用**。
 * 于是 `registerBuiltinPanels.ts` 里的 `titleKey: "notebook"` 指向了一个不存在
 * 的键 —— 而 `tf()` 的契约是"查不到时返回键名本身"（那是给扩展键设计的可诊断
 * 行为），标签条与浮动标题就把 `notebook` 直接渲染给了用户。
 *
 * 【这不是第一次】更早一轮已经发生过一次同类事故：独立窗口标题栏显示
 * `undo_history_title`。当时的修法是在**调用点**加 `translateOutsideReact()`
 * 并给那个**函数**补了单测（见 `I18nProvider.test.ts`）—— 但键本身仍然无人守，
 * 所以同一个 bug 换一条路径又回来了。
 *
 * 本文件补的正是缺掉的那一层：**验证被翻译的键真实存在**，而不只是
 * "翻译函数可用"。三道检查：
 *   1. 所有 `*Key` 字面量字段都指向真实键（类型已收紧到 `MessageKey`，
 *      但字符串拼接与 `as` 断言仍可能绕过，这里做一次数据级兜底）；
 *   2. 每个**真实注册的面板**的标题都能翻译出文案（不是键名）；
 *   3. 动态拼接的键族，其**每一个枚举值**展开后都存在。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

import { translateOutsideReact } from "./I18nProvider";
import { enUS } from "./en-US";
import { listPanels } from "../features/dock/panelRegistry";
import { registerBuiltinPanels } from "../components/dock/registerBuiltinPanels";
import { THEME_MODE_SETTINGS } from "../theme/themeTypes";
import type { RippleMode } from "../features/session/ripplePreview";

const CATALOG_KEYS = new Set(Object.keys(enUS));

/** 表示"i18n 键"的字段名（不是键盘键、不是菜单 id）。 */
const KEY_FIELD_NAMES = [
    "titleKey",
    "labelKey",
    "messageKey",
    "descKey",
    "descriptionKey",
    "textKey",
    "nameKey",
];

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

describe("键引用完整性", () => {
    test("所有 *Key 字面量字段都指向真实存在的词典键", () => {
        const re = new RegExp(`\\b(?:${KEY_FIELD_NAMES.join("|")})\\s*:\\s*"([^"]+)"`, "g");
        const offenders: string[] = [];
        let checked = 0;

        for (const file of walk("src")) {
            if (/\.test\.tsx?$/.test(file)) continue;
            const source = readFileSync(file, "utf8");
            for (const match of source.matchAll(re)) {
                checked += 1;
                if (!CATALOG_KEYS.has(match[1])) {
                    offenders.push(`${file}: ${match[1]}`);
                }
            }
        }

        // 自检：扫描本身必须真的扫到东西，否则"0 处违规"毫无意义。
        expect(checked, "没有扫到任何 *Key 字段，正则或字段名清单已失效").toBeGreaterThan(50);
        expect(
            offenders.length === 0
                ? []
                : [
                      "以下键引用指向不存在的词典键（界面上会显示成键名）：",
                      ...offenders.map((line) => `  ${line}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    test("每个注册面板的标题都能翻译成文案，而不是键名", () => {
        /*
         * 这一条比上一条更强：它走的是**运行期真实注册表**，因此拼接出来的
         * 键、被 `as` 断言绕过的键、以及注册时才决定的键都逃不掉。
         */
        registerBuiltinPanels();

        const panels = listPanels();
        expect(panels.length, "内置面板一个都没注册上").toBeGreaterThan(0);

        const offenders = panels
            .filter((panel) => {
                const translated = translateOutsideReact(panel.titleKey);
                return translated === panel.titleKey || translated.trim() === "";
            })
            .map((panel) => `  ${panel.id}: titleKey="${panel.titleKey}"`);

        expect(
            offenders.length === 0
                ? []
                : ["以下面板的标题翻译不出文案（用户会看到键名）：", ...offenders].join("\n"),
        ).toEqual([]);
    });

    test("动态拼接的键族：发现到的每一族都必须登记，且每个枚举值都存在", () => {
        /*
         * 模板拼接的键没有字面量可以扫描，因此这里**从源码里发现**拼接点，
         * 再要求每一族都在下面的表里登记取值域。这样新增一个拼接点（而不是
         * 新增一个枚举值）也会被抓住 —— 那正是最容易漏掉词条的写法。
         *
         * 取值域取自**代码里的常量**，而不是手抄，新增模式时忘了加词条会红。
         */
        const FAMILIES: Record<string, { values: readonly string[]; note: string }> = {
            theme_: {
                values: THEME_MODE_SETTINGS,
                note: "MenuBar / 外观设置的主题模式标签",
            },
            ripple_tooltip_: {
                // 取值域与 `RippleMode` 绑定：多一个模式就多一个键。
                values: ["off", "track", "all"] satisfies readonly RippleMode[],
                note: "ActionBar 的波纹模式提示",
            },
        };

        // 发现：`t(\`prefix_${var}\`)` 形态（排除本文件自身 —— 门禁不该扫自己）。
        const siteRe = new RegExp("\\b(?:tf?|tAny)\\(`([a-zA-Z_]+)\\$\\{", "g");
        const discovered = new Set<string>();
        for (const file of walk("src")) {
            if (/\.test\.tsx?$/.test(file)) continue;
            const source = readFileSync(file, "utf8");
            for (const match of source.matchAll(siteRe)) discovered.add(match[1]);
        }

        // 自检：确实发现了拼接点，否则下面的断言全是空转。
        expect(discovered.size, "没有发现任何动态键拼接点，正则已失效").toBeGreaterThan(0);

        const unregistered = [...discovered].filter((prefix) => !(prefix in FAMILIES));
        expect(
            unregistered.length === 0
                ? []
                : [
                      "以下动态键族没有登记取值域，无法验证其词条是否齐全：",
                      ...unregistered.map((p) => `  ${p}`),
                  ].join("\n"),
        ).toEqual([]);

        const missing: string[] = [];
        for (const [prefix, family] of Object.entries(FAMILIES)) {
            for (const value of family.values) {
                const key = prefix + value;
                if (!CATALOG_KEYS.has(key)) missing.push(`  ${key}  （${family.note}）`);
            }
        }
        expect(
            missing.length === 0 ? [] : ["以下动态键展开后不存在：", ...missing].join("\n"),
        ).toEqual([]);
        // 【为什么显式给超时】这条要读上千个源文件（含 `scripts/`），全量跑测试时
        // 与其它 worker 抢 I/O，默认 5 s 会偶发超时 —— 而它一旦超时，报出来的是
        // "测试超时"而不是"有没有死键"，等于把门禁变成噪声。
    }, 30_000);

    /*
     * 调用的**字面量**键必须存在于参考目录。
     *
     * 【为什么必须补这一条】`t()` 的 `MessageKey` 类型保护只覆盖字面量调用，
     * 而快速搜索弹窗当年正是用 `(t as (key: string) => string)("qs_placeholder")`
     * 显式绕开它，后来那批强转又被换成同样无类型的 `tf()` —— 于是 8 个**根本不存在**
     * 的键一直在界面上以键名原文显示（`tf` 查不到时原样返回键名）。目录一致性门禁
     * 只管目录、类型只管 `t` 的字面量，**没有一层在守调用点** —— 本用例就是那一层。
     *
     * 【为什么必须剥离注释】`src/ui/*` 的 JSDoc 示例里写着 `t("bitrate")` 这类
     * 示范键，不剥离会误报 8 处、逼人加无意义的豁免。
     */
    /*
     * 反向检查：目录里的每个键都至少被**引用**一次。
     *
     * 【为什么必须有】正向检查（"调用的键存在"）只保证不写错名字，不保证不写没用的
     * 文案。捕获面板从"一整块面板"改成"工具栏一个图标 + 设置窗口"时，旧的一整族
     * 文案（`capture_guide_*`、`capture_output_*` …）没有跟着删 —— 实测 **49 个键、
     * 245 条译文**再也不会被渲染。它们不会报错、不会被发现，只会让下一个维护者以为
     * "词表里有，说明界面上有"。
     *
     * 【为什么豁免那几个前缀】它们是**动态拼接**出来的键族（`history_op_<op>`、
     * `theme_<mode>` …），源码里没有字面量。这份豁免清单与
     * `keyReferenceIntegrity` 里的 `FAMILIES` 是同一份知识 —— 新增一个拼接点必须
     * 同时在这里登记，否则那些键会被判成死键。
     */
    test("词典里没有死键：每个键都至少被引用一次", () => {
        const DYNAMIC_PREFIXES = [
            "history_op_",
            "theme_",
            "ripple_tooltip_",
            "recording_error_",
            "installer_",
        ];
        /** 不按 `.tsx?` 过滤：`scripts/` 里的安装器词表是 `.mjs` / `.nsi`。 */
        function walkAny(dir: string, out: string[] = []): string[] {
            for (const entry of readdirSync(dir)) {
                const full = join(dir, entry);
                if (statSync(full).isDirectory()) {
                    if (entry !== "node_modules") walkAny(full, out);
                } else if (/\.(tsx?|mjs|js|ps1|nsi)$/.test(entry)) {
                    out.push(full);
                }
            }
            return out;
        }

        const sources: string[] = [];
        for (const file of walkAny("src")) {
            if (file.startsWith(join("src", "i18n"))) continue;
            sources.push(readFileSync(file, "utf8"));
        }
        // 安装器的五语词表由 `scripts/release/build-installer-strings.mjs` 读取。
        for (const file of walkAny(join("..", "scripts"))) {
            sources.push(readFileSync(file, "utf8"));
        }
        const corpus = sources.join("\n");

        // 【为什么先切成标识符再查，而不是 `corpus.includes(key)`】源文件加起来
        // 约 30 MB，2000 个键逐个 `includes` 是 60 GB 的扫描量（实测 4 s，全量跑
        // 测试时会撞上 5 s 的默认超时）。切一次标识符得到集合，查询变成 O(1)。
        // 而且它比 `includes` **更精确**：`foo_capture_guide_body` 这种更长的
        // 标识符不会被误判成引用了 `capture_guide_body`。
        const tokens = new Set(corpus.match(/[A-Za-z0-9_]+/g) ?? []);

        const dead = [...CATALOG_KEYS]
            .filter((key) => !DYNAMIC_PREFIXES.some((prefix) => key.startsWith(prefix)))
            .filter((key) => !tokens.has(key));

        // 自检：扫描本身必须真的读到东西，否则"0 个死键"毫无意义。
        expect(sources.length, "没有读到任何源码，扫描路径已失效").toBeGreaterThan(300);
        expect(tokens.size, "没有切出任何标识符，扫描路径已失效").toBeGreaterThan(10_000);
        expect(
            dead.length === 0
                ? []
                : [
                      "以下词条没有任何调用点（界面上永远不会出现）：",
                      ...dead.map((key) => `  ${key}`),
                  ].join("\n"),
        ).toEqual([]);
    }, 30_000);

    test("调用的字面量键必须在参考目录里", () => {
        /*
         * 两族写法都要守：
         * - `t("key")` / `tf("key")` / `tAny("key")` —— 单参调用，闭合括号必须紧跟；
         * - `tVars("key", { … })` / `plural("key", n)` / `shortcut("key")`，以及
         *   **取值器形态** `lookup.t("key", { … })` —— 带插值变量的调用。`tVars`
         *   正是为了取代手写 `.replace("{name}", …)` 而存在的，用它的地方同样会
         *   "查不到就显示键名"。它们的第二个参数是对象/数字，因此这里只要求
         *   **引号闭合**，不要求紧跟 `)`。
         */
        const callRe =
            /\b(?:t|tf|tAny)\("([a-zA-Z0-9_]+)"\)|\b(?:tVars|plural|shortcut|t)\("([a-zA-Z0-9_]+)"/g;
        const stripCommentLines = (source: string): string =>
            source
                .split("\n")
                .filter((line) => !/^\s*(\/\/|\*|\/\*)/.test(line))
                .join("\n");
        const offenders: string[] = [];
        let scanned = 0;
        for (const file of walk("src")) {
            if (/\.test\.tsx?$/.test(file)) continue;
            // 目录自身的键定义处不必检查（那是"键存在"的地方）
            if (file.startsWith(join("src", "i18n"))) continue;
            const source = stripCommentLines(readFileSync(file, "utf8"));
            for (const match of source.matchAll(callRe)) {
                const key = match[1] ?? match[2];
                scanned += 1;
                if (!CATALOG_KEYS.has(key)) offenders.push(`  ${file}: ${key}`);
            }
        }

        // 自检：正则必须真的扫到东西，否则"0 处违规"毫无意义（实测约 1500 处）。
        expect(scanned, "没有扫到字面量键调用，正则或扫描路径已失效").toBeGreaterThan(1000);
        expect(
            offenders.length === 0
                ? []
                : ["以下调用点使用了目录里不存在的键（界面会原样显示键名）：", ...offenders].join(
                      "\n",
                  ),
        ).toEqual([]);
    });
});
