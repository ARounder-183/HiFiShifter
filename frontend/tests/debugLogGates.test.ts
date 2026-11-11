/**
 * 生产源码里的诊断输出必须走 debugLog.ts 的单一通道。
 */
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { expect, test } from "vitest";

const SRC = join(process.cwd(), "src");
const ALLOWED = new Set(["src/utils/debugLog.ts", "src/utils/waveformDebug.ts"]);

function sourceFiles(dir = SRC): string[] {
    const out: string[] = [];
    for (const entry of readdirSync(dir)) {
        const full = join(dir, entry);
        if (statSync(full).isDirectory()) out.push(...sourceFiles(full));
        else if (/.(ts|tsx)$/.test(entry) && !/.(test|bench)./.test(entry)) out.push(full);
    }
    return out;
}

test("生产源码不直接调用 console.log", () => {
    const offenders: string[] = [];
    for (const file of sourceFiles()) {
        const rel = relative(process.cwd(), file).replaceAll("\\", "/");
        if (ALLOWED.has(rel)) continue;
        const lines = readFileSync(file, "utf8").split("\n");
        lines.forEach((line, index) => {
            if (/console\.log\(/.test(line)) offenders.push(rel + ":" + (index + 1));
        });
    }
    expect(offenders).toEqual([]);
});
