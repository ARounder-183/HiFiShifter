/*
 * 设计系统完整性门禁。
 *
 * 【为什么需要这一组】前两轮反复出现同一个失败模式：**建了原语，没接消费者**。
 * 实测曾有 20 个值导出零真实消费者（11 个全仓无人引用），而它们要消除的重复
 * **一处未动** —— 于是每件事都有两套看起来都"官方"的写法，一致性净下降。
 *
 * 这类问题类型检查抓不到、人眼也 review 不出来（没有"错误"，只有"多余"），
 * 因此必须由门禁守住。三条规则：
 *
 *   1. **原语必须有消费者** —— 没有消费者的原语是负债，不是资产；
 *   2. **禁止浏览器原生 affordance** —— 桌面应用不该弹 `window.alert`；
 *   3. **禁止裸 `<input type="checkbox">`** —— 应走 `AppSwitchRow`（否则 13px 的
 *      系统复选框与 12px 标签、Radix 控件混排，这是"同一窗口两种控件高度"的来源）。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, test } from "vitest";

const UI_DIR = join("src", "ui");
const SDK_BARREL = join("src", "sdk", "index.ts");

/**
 * 允许"暂时没有内部消费者"的导出。
 *
 * 每一条都必须写明理由 —— 这份清单是**已审计的记录**，不是静音开关。
 * 新增条目必须同时改这里，因此豁免无法悄悄扩大。
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

/** 非测试、非设计系统自身的源文件（SDK barrel 是纯转发，不算消费者）。 */
function consumerFiles(): string[] {
    return walk("src").filter(
        (file) =>
            !file.startsWith(UI_DIR) &&
            file !== SDK_BARREL &&
            !/\.test\.tsx?$/.test(file) &&
            !/\.bench\.ts$/.test(file),
    );
}

/** 从 barrel 里解析出所有值导出（跳过 `type` 导出）。 */
function valueExports(): string[] {
    const source = readFileSync(join(UI_DIR, "index.ts"), "utf8");
    const names = new Set<string>();
    for (const match of source.matchAll(/export\s*\{([\s\S]*?)\}\s*from/g)) {
        for (const raw of match[1].split(",")) {
            const name = raw.trim();
            if (!name || name.startsWith("type ") || name.startsWith("//")) continue;
            names.add(name);
        }
    }
    return [...names].sort();
}

describe("原语必须有消费者", () => {
    test("src/ui 的每个值导出都被 src/ 之外的真实代码用到", () => {
        const files = consumerFiles();
        const sources = files.map((file) => ({ file, text: readFileSync(file, "utf8") }));
        const unadopted: string[] = [];

        for (const name of valueExports()) {
            if (name in NO_CONSUMER_ALLOWED) continue;
            const used = sources.some(({ text }) =>
                new RegExp(`\\b${name}\\b`).test(text),
            );
            if (!used) unadopted.push(name);
        }

        expect(
            unadopted.length === 0
                ? []
                : [
                      `以下原语没有任何消费者 —— 要么接上，要么删除：`,
                      ...unadopted.map((name) => `  ${name}`),
                  ].join("\n"),
        ).toEqual([]);
    });

    test("豁免清单里每条都写了理由", () => {
        for (const [name, reason] of Object.entries(NO_CONSUMER_ALLOWED)) {
            expect(reason.length, `${name} 的豁免理由过短`).toBeGreaterThan(20);
        }
    });
});

describe("禁止浏览器原生 affordance", () => {
    test("没有 window.alert / window.confirm", () => {
        const offenders: string[] = [];
        for (const file of consumerFiles()) {
            if (!/\.tsx$/.test(file)) continue;
            const lines = readFileSync(file, "utf8").split("\n");
            lines.forEach((line, index) => {
                const trimmed = line.trim();
                if (trimmed.startsWith("//") || trimmed.startsWith("*")) return;
                if (/window\.(alert|confirm)\s*\(/.test(line)) {
                    offenders.push(`${file}:${index + 1}  ${trimmed.slice(0, 60)}`);
                }
            });
        }
        expect(offenders, "桌面应用不该弹浏览器原生弹窗，请用 AppNoticeDialog / AppConfirmDialog").toEqual(
            [],
        );
    });

    test("没有裸 <input type=\"checkbox\">", () => {
        const offenders: string[] = [];
        for (const file of consumerFiles()) {
            if (!/\.tsx$/.test(file)) continue;
            const source = readFileSync(file, "utf8");
            const matches = source.match(/<input[\s\S]{0,300}?type="checkbox"/g) ?? [];
            if (matches.length > 0) offenders.push(`${file}: ${matches.length} 处`);
        }
        expect(
            offenders,
            "请用 AppSwitchRow（control=\"checkbox\"）—— 系统默认复选框与 Radix 控件高度不一致",
        ).toEqual([]);
    });
});
