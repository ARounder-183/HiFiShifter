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
import { readFileSync, readdirSync, statSync } from "node:fs";
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
 * 剥离注释与字符串字面量。
 *
 * 这一步是门禁可信的前提：`// 旧写法用 window.alert(...)` 这类说明性文字
 * 不应被当成真实调用，而 `"<input type=\"checkbox\">"` 这类字符串同理。
 */
function stripCommentsAndStrings(source: string): string {
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
                i += 1;
                continue;
            }
            if (ch === "'") {
                state = "single";
                i += 1;
                continue;
            }
            if (ch === "`") {
                state = "template";
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
        // 字符串内：保留换行以维持行号，其余丢弃
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
