/**
 * 从五语词表抽出安装器文案，生成 NSIS 的 `LangString` 片段。
 *
 * 【为什么是"抽取"而不是"手写五份 .nsh"】安装器原先硬编码简体中文，五语用户看到的
 * 是中文向导。手写第二份文案等于把同一批文字放进一个**没有测试**的地方 —— 词表那边
 * 有 `catalogIntegrity`（五语键齐全、格式合规）守着，这边没有，几个月后必然漂移。
 * 抽取让"新增一种语言"自动生效，也让漏翻译在构建安装器时**立刻失败**。
 *
 * 【为什么用 esbuild 而不是正则解析 .ts】词表里既有跨行字符串也有注释，正则迟早会被
 * 某一行注释或一次换行坑掉，而失败方式是"安静地漏掉一条文案"。用仓库里已经存在的
 * esbuild 把它编译成 JS 再求值，拿到的是真正的对象。
 *
 * 用法：
 *   node tools/build-installer-strings.mjs           # 生成
 *   node tools/build-installer-strings.mjs --check    # 只校验已生成的文件是否最新
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..");
const catalogDir = join(repoRoot, "frontend", "src", "i18n");
const outputPath = join(scriptDir, "installer", "installer_strings.nsh");

/** 词表语言 → NSIS 语言常量。五个 `.nlf` 都随 NSIS 分发，无需额外下载。 */
const LANGUAGES = [
    { locale: "en-US", nsis: "English", export: "enUS", suffix: "EN" },
    { locale: "zh-CN", nsis: "SimpChinese", export: "zhCN", suffix: "ZH_CN" },
    { locale: "zh-TW", nsis: "TradChinese", export: "zhTW", suffix: "ZH_TW" },
    { locale: "ja-JP", nsis: "Japanese", export: "jaJP", suffix: "JA" },
    { locale: "ko-KR", nsis: "Korean", export: "koKR", suffix: "KO" },
];

/** 默认语言：NSIS 在系统语言不匹配任何已声明语言时回退到第一个。 */
const DEFAULT_LANGUAGE = LANGUAGES[0];

/** 需要抽出的键前缀。 */
const KEY_PREFIX = "installer_";

/** NSIS 的 `LangString` 名：`installer_welcome_title` → `HFS_WELCOME_TITLE`。 */
function langStringName(key) {
    return "HFS_" + key.slice(KEY_PREFIX.length).toUpperCase();
}

/**
 * 转义为 NSIS 字符串字面量。
 *
 * 【逐字符而不是链式 replace】顺序敏感：先把换行变成 `$\r$\n`，再转义 `$` 会把刚
 * 插进去的 `$` 又转义一遍，结果是字面量 `$\r$\n` 原样显示给用户。逐字符一次成型
 * 就没有这个陷阱。
 */
function nsisEscape(text) {
    let out = "";
    for (const char of text) {
        if (char === "\r") continue;
        if (char === "\n") out += "$\\r$\\n";
        else if (char === "$") out += "$$";
        else if (char === '"') out += '$\\"';
        else out += char;
    }
    return out;
}

/** 把词表 `.ts` 编译成 JS 并求值，取出导出的词典对象。 */
async function loadCatalog(entry) {
    // esbuild 装在 frontend 下；`import()` 需要 file:// URL（Windows 盘符路径
    // 会被 ESM 加载器当成协议而拒绝）。
    const esbuildEntry = createRequire(import.meta.url).resolve("esbuild", {
        paths: [join(repoRoot, "frontend")],
    });
    const esbuild = await import(pathToFileURL(esbuildEntry).href);
    const source = readFileSync(join(catalogDir, `${entry.locale}.ts`), "utf8");
    const { code } = await esbuild.transform(source, { loader: "ts", format: "cjs" });
    const module = { exports: {} };
    // 词表是纯数据模块（只有 `export const x = {...} as const`），求值是安全的。
    new Function("module", "exports", code)(module, module.exports);
    const catalog = module.exports[entry.export];
    if (!catalog || typeof catalog !== "object") {
        throw new Error(`catalog ${entry.locale} did not export ${entry.export}`);
    }
    return catalog;
}

/** 收集全部语言的安装器文案；缺键即失败。 */
async function collectStrings() {
    const catalogs = [];
    for (const entry of LANGUAGES) {
        catalogs.push({ entry, catalog: await loadCatalog(entry) });
    }

    const keys = Object.keys(catalogs[0].catalog)
        .filter((key) => key.startsWith(KEY_PREFIX))
        .sort();
    if (keys.length === 0) {
        throw new Error(`no keys with the ${KEY_PREFIX} prefix found in ${LANGUAGES[0].locale}`);
    }

    const missing = [];
    for (const { entry, catalog } of catalogs) {
        for (const key of keys) {
            const value = catalog[key];
            if (typeof value !== "string" || value.trim() === "") {
                missing.push(`${entry.locale}: ${key}`);
            }
        }
    }
    if (missing.length > 0) {
        throw new Error(
            "installer strings are missing (add them to every catalog):\n  " + missing.join("\n  "),
        );
    }

    return { keys, catalogs };
}

function render({ keys, catalogs }) {
    const lines = [
        "; 本文件由 tools/build-installer-strings.mjs 生成 —— 不要手工编辑。",
        ";",
        "; 文案的唯一来源是 frontend/src/i18n/*.ts 里 installer_ 前缀的词条；",
        "; 直接改这里会在下一次生成时被覆盖，且 scripts/check-product-consistency.ps1",
        "; 会因与本文件不一致而失败。",
        "",
        `; 语言：${LANGUAGES.map((entry) => `${entry.locale}→${entry.nsis}`).join("、")}`,
        `; 词条：${keys.length} 条 × ${LANGUAGES.length} 种语言`,
        "",
    ];

    for (const key of keys) {
        const name = langStringName(key);
        lines.push(`; ${key}`);
        for (const { entry, catalog } of catalogs) {
            lines.push(
                `LangString ${name} \${LANG_${entry.nsis.toUpperCase()}} "${nsisEscape(catalog[key])}"`,
            );
        }
        lines.push("");
    }

    // 语言选择对话框之前的报错只能用字面量（此时还不知道用户选了哪种语言），
    // 因此把五种语言并列 —— 看不懂日文的用户至少能在同一段文字里认出自己那一行。
    lines.push(
        "; 「需要 64 位 Windows」是唯一在语言选择**之前**就要显示的提示：",
        "; 那时 $LANGUAGE 还没定，$(HFS_...) 取不到值，只能五种语言并列。",
        "!macro HFS_FATAL_NOT_X64",
    );
    for (const { entry, catalog } of catalogs) {
        lines.push(`  ; ${entry.locale}`);
        lines.push(
            `  StrCpy $0 "$0${nsisEscape(catalog.installer_err_not_x64)}$\\r$\\n"`,
        );
    }
    lines.push(
        '  MessageBox MB_OK|MB_ICONSTOP "$0"',
        "!macroend",
        "",
    );

    return lines.join("\n");
}

async function main() {
    const checkOnly = process.argv.includes("--check");
    const strings = await collectStrings();
    const rendered = render(strings);

    if (checkOnly) {
        const current = existsSync(outputPath) ? readFileSync(outputPath, "utf8") : null;
        // 生成物带 BOM；比较前去掉，避免把"有没有 BOM"当成内容差异。
        const normalized = current?.replace(/^\uFEFF/, "") ?? null;
        if (normalized !== rendered) {
            throw new Error(
                `${outputPath} is stale; run: node tools/build-installer-strings.mjs`,
            );
        }
        console.log("installer strings are up to date");
        return;
    }

    mkdirSync(dirname(outputPath), { recursive: true });
    // UTF-8 **带 BOM**：`makensis /INPUTCHARSET UTF8` 对 BOM 更宽容，且与仓库里
    // 其它含非 ASCII 的脚本保持一致（见 tools/*.ps1 的处理）。
    writeFileSync(outputPath, "\uFEFF" + rendered, "utf8");
    console.log(
        `wrote ${outputPath} (${strings.keys.length} keys × ${LANGUAGES.length} languages)`,
    );
}

main().catch((error) => {
    console.error(`build-installer-strings: ${error.message}`);
    process.exit(1);
});
