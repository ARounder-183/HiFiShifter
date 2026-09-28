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
    });
});
