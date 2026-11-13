/**
 * 生产源码里的诊断输出必须走 services/frontendLog.ts 的**唯一**通道。
 *
 * 【为什么从"只禁 console.log"升级】过去只禁 console.log，于是
 * console.error/warn/debug 在 30+ 个文件里合法存在，全都不进文件日志；
 * 用户选了「无」档它们照样输出。四档设置落地之后，任何直接 console 调用都会
 * 绕过档位、去重、截断与 CWE-117 单行化，必须一律禁止。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { expect, test } from "vitest";

const SRC = join(process.cwd(), "src");
const ALLOWED = new Set(["src/services/frontendLog.ts"]);
const CONSOLE = /console\.(log|debug|info|warn|error|trace|assert|dir|table|group|time)\s*\(/;

function sourceFiles(dir = SRC): string[] {
    const out: string[] = [];
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) out.push(...sourceFiles(full));
        else if (/.(ts|tsx)$/.test(entry) && !/.(test|bench)./.test(entry)) out.push(full);
    }
    return out;
}

test("生产源码不直接调用任何 console.*", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles()) {
        const rel = relative(process.cwd(), file).replaceAll("\\", "/");
        if (ALLOWED.has(rel)) continue;
        const lines = readFileSync(file, "utf8").split("\n");
        lines.forEach((line, index) => {
            if (CONSOLE.test(line)) offenders.push(rel + ":" + (index + 1));
        });
    }
    expect(offenders).toEqual([]);
});

test("生产源码不再依赖独立的 localStorage 调试开关", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles()) {
        const rel = relative(process.cwd(), file).replaceAll("\\", "/");
        readFileSync(file, "utf8")
            .split("\n")
            .forEach((line, index) => {
                if (/hifishifter\.debug/.test(line)) offenders.push(rel + ":" + (index + 1));
            });
    }
    expect(offenders).toEqual([]);
});
