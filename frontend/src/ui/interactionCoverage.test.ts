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

describe("能力层原语本身是齐备的", () => {
    test("三个原语都有实现且已从 barrel 导出", () => {
        const barrel = readFileSync(join("src", "ui", "index.ts"), "utf8");
        for (const name of ["AppSelect", "AppNumberField", "AppSlider"]) {
            expect(barrel, `${name} 未从 src/ui 导出`).toContain(name);
        }
    });
});
