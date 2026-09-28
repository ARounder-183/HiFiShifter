/*
 * 交互能力覆盖率门禁。
 *
 * 【为什么用"禁止旧写法"而不是"统计覆盖率"】统计覆盖率要解析 JSX 属性、判断
 * 某个 `onWheel` 是否真的调了 `applySelectWheelChange` —— 脆弱、易漏、随写法
 * 变化而失效。而"禁止 `<Select.Root`"是**可判定**的文本事实：迁移一处就少一处，
 * 门禁随迁移自动收紧，不会误报。
 *
 * 【它防的是什么】上一轮的教训：能力（滚轮 / 精细调整）如果是"调用方要记得接"
 * 的东西，就一定会有 45% 的控件漏掉 —— 实测 65/135 无滚轮、39/66 无精细调整。
 * 因此本门禁要求：**取值控件必须用能力层原语**（`AppSelect` / `AppNumberField` /
 * `AppSlider`），而不是裸 Radix `Select` 或裸 `<input>`。
 *
 * 【豁免】确实需要特例时，在文件里写一行
 *   `// hs-interaction-exempt: <理由>`
 * 门禁会放行**整个文件**。理由必须写清楚 —— 豁免是"已审计"的记录，不是静音开关。
 *
 * 更关键的是：下面的用例会断言**豁免清单恰好等于已知的那几个文件**。
 * 因此新增豁免必须同时改测试，豁免无法悄悄扩大 —— 否则这条门禁会在几轮之后
 * 被逐个 marker 蛀空，重演"能力靠自觉"的老问题。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

/** 能力层自身的实现目录：它就是这些写法的**定义处**。 */
const PRIMITIVE_DIR = join("src", "ui");

const EXEMPT_MARKER = "hs-interaction-exempt";

/**
 * 已审计并接受的豁免清单。
 *
 * 这四个文件的共同点：都是**紧凑 chrome**（主工具栏 / 标尺行 / 淡变面板 /
 * 参数编辑器），控件尺寸与行高耦合、需要内联底色，而能力层原语是表单尺寸。
 * 它们的滚轮与精细调整接线**已经完备** —— 豁免的理由是"原语不适用"，
 * 不是"能力缺失"。
 */
const ACCEPTED_EXEMPTIONS = [
    join("src", "components", "layout", "ActionBar.tsx"),
    join("src", "components", "layout", "timeline", "FadeContextMenu.tsx"),
    join("src", "components", "layout", "timeline", "TempoMapRulerRow.tsx"),
    join("src", "components", "layout", "PianoRollPanel.tsx"),
];

interface Violation {
    file: string;
    line: number;
    rule: string;
    text: string;
}

const RULES: { rule: string; pattern: RegExp }[] = [
    {
        rule: "裸 Radix Select：请用 AppSelect（内建滚轮换项）",
        pattern: /<Select\.Root\b/,
    },
    {
        rule: "裸 number 输入：请用 AppNumberField（内建滚轮 + 精细调整）",
        pattern: /<input\b[^>]*type="number"|type="number"[^>]*\/?>/,
    },
    {
        rule: "裸 range 滑块：请用 AppSlider（内建滚轮 + 精细调整 + 主题样式）",
        pattern: /<input\b[^>]*type="range"|type="range"[^>]*\/?>/,
    },
    {
        rule: "已废弃的滚轮守卫：请用 useRangeWheelGuard / AppSlider",
        pattern: /useWheelScrollGuard/,
    },
    {
        /*
         * 文件选择：三处（导入主题 / 导入布局 / 插入图片）曾各自手写一遍，其中一处
         * 漏了"选完清空 `value`"，于是再次选同一个文件不再触发 `change` ——
         * 表现为"点了没反应"。统一到 `AppFileInput` 后这条规则防止它再被手写出来。
         */
        rule: "裸 file 输入：请用 AppFileInput（统一处理 value 清空）",
        pattern: /type\s*=\s*\{?\s*["'`]file["'`]/,
    },
];

/** 行级注释判定：行注释、块注释续行、JSX 注释续行。 */
function isCommentLine(line: string): boolean {
    const trimmed = line.trim();
    return (
        trimmed.startsWith("//") ||
        trimmed.startsWith("/*") ||
        trimmed.startsWith("*") ||
        trimmed.startsWith("{/*")
    );
}

/**
 * 去掉块注释（含 JSX 的 `{/* … *\/}`）。
 *
 * 【为什么需要】`isCommentLine` 只能判断**单行**，而 JSX 注释常跨多行：续行的
 * 文字不以 `*` 开头，于是"说明为什么不用 `fixed inset-0`"这类注释会被当成
 * 真实代码报违规。规则要检查的是代码，不是解释。
 */
function stripBlockComments(source: string): string {
    return source
        .replace(/\{\/\*[\s\S]*?\*\/\}/g, "")
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/[^\n]*/g, "");
}

function walk(dir: string, out: string[] = []): string[] {
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) {
            if (entry === "node_modules") continue;
            walk(full, out);
        } else if (/\.tsx?$/.test(entry) && !/\.test\.tsx?$/.test(entry)) {
            out.push(full);
        }
    }
    return out;
}

/** 收集违规项。 */
function collect(): Violation[] {
    const violations: Violation[] = [];
    for (const file of walk("src")) {
        if (file.startsWith(PRIMITIVE_DIR)) continue;
        const source = readFileSync(file, "utf8");
        if (source.includes(EXEMPT_MARKER)) continue;

        const lines = source.split("\n");
        for (const { rule, pattern } of RULES) {
            for (let i = 0; i < lines.length; i += 1) {
                // 注释里提到旧写法不算违规（例如说明"为什么替换掉它"）。
                // 这条是必要的：新 hook 的文档注释必须能写出被替换者的名字。
                if (isCommentLine(lines[i])) continue;
                if (pattern.test(lines[i])) {
                    violations.push({
                        file,
                        line: i + 1,
                        rule,
                        text: lines[i].trim().slice(0, 70),
                    });
                }
            }
        }
    }
    return violations;
}

describe("取值控件必须走能力层", () => {
    test("没有裸 Select / number / range / 废弃守卫", () => {
        const violations = collect();
        const report = violations.map((v) => `${v.file}:${v.line}  [${v.rule}]\n    ${v.text}`);
        expect(
            violations.length === 0 ? [] : [`共 ${violations.length} 处：`, ...report].join("\n"),
        ).toEqual([]);
    });
});

describe("豁免清单是受控的", () => {
    test("带豁免标记的文件恰好等于已审计的那几个（新增豁免必须改测试）", () => {
        const marked = walk("src")
            .filter((file) => readFileSync(file, "utf8").includes(EXEMPT_MARKER))
            .sort();
        expect(marked).toEqual([...ACCEPTED_EXEMPTIONS].sort());
    });

    test("每个豁免都写了具体理由（不是空标记）", () => {
        for (const file of ACCEPTED_EXEMPTIONS) {
            const source = readFileSync(file, "utf8");
            const line = source.split("\n").find((l) => l.includes(EXEMPT_MARKER));
            expect(line, `${file} 缺少豁免理由`).toBeTruthy();
            const reason = line!.slice(line!.indexOf(EXEMPT_MARKER) + EXEMPT_MARKER.length).trim();
            expect(reason.length, `${file} 的豁免理由过短`).toBeGreaterThan(20);
        }
    });
});

/*
 * 模态表面必须走组合壳。
 *
 * 【为什么禁 `fixed inset-0`】这是"手写模态"的指纹：自己铺一层遮罩 + 自己定位
 * 卡片。实测过 5 处，其中 4 处缺 Esc、全部不抑制全局快捷键（框内按空格会触发
 * 播放）、没有焦点管理、没有 Enter 默认动作，标题字号与宽度也各自为政
 * （13px vs 20px；380/420px vs 四档 400/520/640/800）。
 *
 * `AppDialog` 与 `AppContextMenu` 已经把这 9 件事各做一次，没有理由再手写。
 * 允许清单里只剩快速搜索面板：它不是对话框也不是菜单，而是一个带输入框的
 * 命令面板（自己的输入、自己的列表、自己的快捷键模型），并且已经完整实现了
 * 键盘契约（Esc / Enter / 方向键 / 快捷键抑制）—— 强行套 `AppDialog` 会把它
 * 变成一个"对话框里放输入框"的别扭结构。
 */
const MODAL_SHELL_ALLOWED = [join("src", "components", "layout", "QuickSearchPopup.tsx")];

describe("模态表面必须走组合壳", () => {
    test("没有手写的 fixed inset-0 遮罩", () => {
        const offenders: string[] = [];
        for (const file of walk("src")) {
            if (MODAL_SHELL_ALLOWED.includes(file)) continue;
            // 先剥注释：说明性文字里提到这个写法不算违规。
            const lines = stripBlockComments(readFileSync(file, "utf8")).split("\n");
            for (let i = 0; i < lines.length; i += 1) {
                if (/fixed inset-0/.test(lines[i])) offenders.push(`  ${file}:${i + 1}`);
            }
        }
        expect(
            offenders.length === 0
                ? []
                : [
                      "以下位置手写了模态遮罩 —— 请用 AppDialog / AppContextMenu：",
                      ...offenders,
                  ].join("\n"),
        ).toEqual([]);
    });

    test("允许清单里的文件仍然完整实现了键盘契约", () => {
        /*
         * 豁免的前提是"它自己做得对"，所以这里把前提钉住：一旦快速搜索面板
         * 丢了关闭键或快捷键抑制，豁免就不再成立。
         *
         * 【关闭键为什么接受两种形态】它走的是 `quickSearch.close` **键位绑定**
         * （默认 Escape，但用户可改），而不是写死的 `"Escape"` 字面量 ——
         * 这是更好的做法，门禁不该因此判它不合格。
         */
        for (const file of MODAL_SHELL_ALLOWED) {
            const source = readFileSync(file, "utf8");
            const hasDismiss = /Escape|escape/.test(source) || /close"\]/.test(source);
            expect(hasDismiss, `${file} 没有任何关闭键（Escape 或键位绑定）`).toBe(true);
            expect(source, `${file} 未抑制全局快捷键`).toContain("useShortcutSuppression");
        }
    });
});

describe("菜单表面必须走组合壳", () => {
    /*
     * `role="menu"` 且自带 `fixed` 定位的**菜单框**只允许出现在两个地方：组合壳
     * `ui/Menu.tsx`，以及登记在案的「组合菜单面」。
     *
     * 【组合菜单面为什么不是"没迁完"】扁平条目模型（`AppMenuItemSpec`）表达不了
     * 它们的内容：
     *   - `ClipContextMenu` / `FadeContextMenu`：行内嵌控件（take 行的反向/声道
     *     按钮）、飞出子菜单、淡变形状图标条 —— 迁成扁平条目会**删功能**；
     *   - `ClipRateEditorDialog`：速率编辑浮层，内嵌多个输入框与按钮。
     * 它们已共享壳的全部机制（`useMenuKeyboard` 键盘模型、`clampAxisPosition`
     * 定位钳制、`data-hs-floating-menu` 契约、同一套条目样式）。
     *
     * ActionBar 的弹层与快速搜索不是菜单（内嵌滑杆 / 自有输入框），它们不写
     * `role="menu"`，因此不在本门禁范围内 —— 这正是"角色如实声明"的例子。
     *
     * 新增一处菜单框必须改这个清单：让"又手写了一个"在 review 里被看见。
     */
    const MENU_SURFACE_ALLOWED = [
        join("src", "ui", "Menu.tsx"),
        join("src", "components", "layout", "timeline", "ClipContextMenu.tsx"),
        join("src", "components", "layout", "timeline", "FadeContextMenu.tsx"),
        join("src", "components", "layout", "timeline", "ClipRateEditorDialog.tsx"),
    ];

    test("没有白名单之外的菜单框", () => {
        const offenders: string[] = [];
        for (const file of walk("src")) {
            if (!/\.tsx$/.test(file) || /\.test\.tsx$/.test(file)) continue;
            if (MENU_SURFACE_ALLOWED.includes(file)) continue;
            const source = readFileSync(file, "utf8");
            if (/role="menu"/.test(source) && /fixed z-|"fixed /.test(source)) {
                offenders.push(`  ${file}`);
            }
        }
        expect(
            offenders.length === 0
                ? []
                : ["以下位置手写了菜单框 —— 请用 AppContextMenu：", ...offenders].join("\n"),
        ).toEqual([]);
    });

    test("豁免的组合菜单面仍在共享键盘模型与时间轴契约", () => {
        // 豁免的前提是"它们做对了"，这里把前提钉住。
        for (const file of MENU_SURFACE_ALLOWED.slice(1)) {
            const source = readFileSync(file, "utf8");
            expect(source, `${file} 丢了共享键盘模型`).toContain("useMenuKeyboard");
            expect(source, `${file} 丢了时间轴浮动菜单契约`).toContain("data-hs-floating-menu");
        }
    });
});

describe("能力层原语本身是齐备的", () => {
    test("三个原语都有实现且已从 barrel 导出", () => {
        const barrel = readFileSync(join("src", "ui", "index.ts"), "utf8");
        for (const name of ["AppSelect", "AppNumberField", "AppSlider"]) {
            expect(barrel, `${name} 未从 src/ui 导出`).toContain(name);
        }
    });
});
