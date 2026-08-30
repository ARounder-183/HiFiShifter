// 一次性 codemod：把仓库历史遗留的"脚本式断言"测试文件
// （顶层直接执行检查 + console.log，没有 describe/it/test 套件，
// 导致 vitest 报 "No test suite found" 而整批 FAIL）包装成真正的
// vitest 套件。
//
// 变换规则：
//   1. 全文件收集所有**顶层 import 声明**（多行花括号感知；允许分散在
//      文件多处，ESM 允许 import 出现任意顶层位置），统一上提到顶部；
//   2. 其余全部语句原样包进单个 test("<相对路径> scripted checks",
//      async () => {...}); —— async 以兼容历史脚本的顶层 await；
//   3. 注入 `import { test } from "vitest";`。
//
// 用法：node scripts/wrap-scripted-tests.mjs [--dry]
// 幂等性：已包含 `from "vitest"` 的文件自动跳过。

import { readFileSync, writeFileSync, readdirSync, statSync } from "node:fs";
import { join, relative, dirname, sep } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "src");
const dry = process.argv.includes("--dry");

function listTestFiles(dir) {
    const out = [];
    for (const name of readdirSync(dir)) {
        const full = join(dir, name);
        if (statSync(full).isDirectory()) {
            out.push(...listTestFiles(full));
        } else if (name.endsWith(".test.ts")) {
            out.push(full);
        }
    }
    return out;
}

const balance = (line) =>
    (line.match(/\(/g)?.length ?? 0) +
    (line.match(/\{/g)?.length ?? 0) -
    (line.match(/\)/g)?.length ?? 0) -
    (line.match(/\}/g)?.length ?? 0);

let wrapped = 0;
let skipped = 0;
const failures = [];

for (const file of listTestFiles(root)) {
    const content = readFileSync(file, "utf8");
    if (content.includes(`"vitest"`)) {
        skipped += 1;
        continue;
    }

    const lines = content.split(/\r?\n/);
    const importLines = [];
    const bodyLines = [];
    let inImport = false;
    let depth = 0;
    let sawImport = false;

    for (const raw of lines) {
        const trimmed = raw.trim();
        if (!inImport) {
            // 动态导入 `await import(...)` 已确认不存在于本仓库测试中；
            // 顶层 import 声明一律上提。
            if (/^import\b/.test(trimmed)) {
                sawImport = true;
                inImport = true;
                depth = balance(raw);
                importLines.push(raw);
                if (depth <= 0 && /;\s*$/.test(trimmed)) inImport = false;
                continue;
            }
            bodyLines.push(raw);
            continue;
        }
        importLines.push(raw);
        depth += balance(raw);
        if (depth <= 0 && /;\s*$/.test(trimmed)) inImport = false;
    }

    if (!sawImport) {
        failures.push(`${file}: no import statement found`);
        continue;
    }
    if (inImport) {
        failures.push(`${file}: unbalanced import block`);
        continue;
    }

    const body = bodyLines.join("\n").trim();
    if (body.length === 0) {
        failures.push(`${file}: empty body after imports`);
        continue;
    }

    const rel = relative(root, file).split(sep).join("/");

    const next = [
        `import { test } from "vitest";`,
        "",
        importLines.join("\n"),
        "",
        `test("${rel} scripted checks", async () => {`,
        body,
        "});",
        "",
    ].join("\n");

    if (!dry) writeFileSync(file, next, "utf8");
    wrapped += 1;
}

console.log(JSON.stringify({ wrapped, skippedAlreadyWrapped: skipped, failures }, null, 2));
