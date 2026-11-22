/**
 * Extract the installer copy from the five-language catalog and emit the NSIS
 * `LangString` fragment.
 *
 * [Why extract instead of hand-writing five .nsh files] The installer used to
 * hard-code Simplified Chinese, so users of the other four languages saw a
 * Chinese wizard. Hand-writing a second copy would put the same words into a
 * place **without any tests** — the catalog side is guarded by
 * `catalogIntegrity` (all keys present in all five languages, well-formed);
 * this side had nothing, and would drift within months. Extraction makes
 * "add a language" take effect automatically and makes a missing translation
 * **fail immediately** at installer build time.
 *
 * [Why esbuild instead of regex-parsing the .ts] The catalog contains
 * multi-line strings as well as comments; a regex would eventually be tripped
 * up by one comment line or one newline, and the failure mode is "quietly
 * dropping one string". Compiling it to JS with the esbuild already in the
 * repo and evaluating it yields the real object.
 *
 * Usage:
 *   node scripts/release/build-installer-strings.mjs           # generate
 *   node scripts/release/build-installer-strings.mjs --check    # only verify the generated file is current
 */

import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scriptDir, "..", "..");
const catalogDir = join(repoRoot, "frontend", "src", "i18n");
const outputPath = join(scriptDir, "installer", "installer_strings.nsh");

/** Catalog language → NSIS language constant. All five `.nlf` files ship with NSIS. */
const LANGUAGES = [
    { locale: "en-US", nsis: "English", export: "enUS", suffix: "EN" },
    { locale: "zh-CN", nsis: "SimpChinese", export: "zhCN", suffix: "ZH_CN" },
    { locale: "zh-TW", nsis: "TradChinese", export: "zhTW", suffix: "ZH_TW" },
    { locale: "ja-JP", nsis: "Japanese", export: "jaJP", suffix: "JA" },
    { locale: "ko-KR", nsis: "Korean", export: "koKR", suffix: "KO" },
];

/** Default language: NSIS falls back to the first one when the system language matches none. */
const DEFAULT_LANGUAGE = LANGUAGES[0];

/** Key prefix that gets extracted. */
const KEY_PREFIX = "installer_";

/** NSIS `LangString` name: `installer_welcome_title` → `HFS_WELCOME_TITLE`. */
function langStringName(key) {
    return "HFS_" + key.slice(KEY_PREFIX.length).toUpperCase();
}

/**
 * Escape into an NSIS string literal.
 *
 * [Per-character instead of chained replace] Order matters: turn newlines into
 * `$\r$\n` first, then escaping `$` would re-escape the `$` just inserted and
 * the literal `$\r$\n` would be shown to the user verbatim. Building the
 * output per character avoids that trap in one pass.
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

/** Compile the catalog `.ts` into JS and evaluate it, taking the exported dictionary. */
async function loadCatalog(entry) {
    // esbuild is installed under frontend; `import()` needs a file:// URL (a
    // Windows drive-letter path is rejected by the ESM loader as a protocol).
    const esbuildEntry = createRequire(import.meta.url).resolve("esbuild", {
        paths: [join(repoRoot, "frontend")],
    });
    const esbuild = await import(pathToFileURL(esbuildEntry).href);
    const source = readFileSync(join(catalogDir, `${entry.locale}.ts`), "utf8");
    const { code } = await esbuild.transform(source, { loader: "ts", format: "cjs" });
    const module = { exports: {} };
    // The catalog is a pure data module (only `export const x = {...} as const`); evaluating it is safe.
    new Function("module", "exports", code)(module, module.exports);
    const catalog = module.exports[entry.export];
    if (!catalog || typeof catalog !== "object") {
        throw new Error(`catalog ${entry.locale} did not export ${entry.export}`);
    }
    return catalog;
}

/** Collect the installer copy for every language; a missing key fails. */
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
        "; Generated by scripts/release/build-installer-strings.mjs — do not edit by hand.",
        ";",
        "; The single source of the copy is the installer_-prefixed entries in frontend/src/i18n/*.ts;",
        "; editing this file directly gets overwritten on the next generation, and",
        "; scripts/checks/check-product-consistency.ps1 fails on the mismatch.",
        "",
        `; Languages: ${LANGUAGES.map((entry) => `${entry.locale}→${entry.nsis}`).join(", ")}`,
        `; Entries: ${keys.length} × ${LANGUAGES.length} languages`,
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

    // An error box shown before the language-selection dialog can only use
    // literals ($LANGUAGE is not decided yet, so $(HFS_...) has no value);
    // therefore all five languages are concatenated — a user who cannot read
    // Japanese still recognizes their own line in the same text.
    lines.push(
        '; "64-bit Windows required" is the only message shown **before** language selection:',
        "; at that point $LANGUAGE is unset and $(HFS_...) resolves to nothing, so all five languages are concatenated.",
        "!macro HFS_FATAL_NOT_X64",
    );
    for (const { entry, catalog } of catalogs) {
        lines.push(`  ; ${entry.locale}`);
        lines.push(`  StrCpy $0 "$0${nsisEscape(catalog.installer_err_not_x64)}$\\r$\\n"`);
    }
    lines.push('  MessageBox MB_OK|MB_ICONSTOP "$0"', "!macroend", "");

    return lines.join("\n");
}

async function main() {
    const checkOnly = process.argv.includes("--check");
    const strings = await collectStrings();
    const rendered = render(strings);

    if (checkOnly) {
        const current = existsSync(outputPath) ? readFileSync(outputPath, "utf8") : null;
        // The generated file has a BOM; strip it before comparing so the
        // BOM's presence is not treated as a content difference.
        const normalized = current?.replace(/^\uFEFF/, "") ?? null;
        if (normalized !== rendered) {
            throw new Error(
                `${outputPath} is stale; run: node scripts/release/build-installer-strings.mjs`,
            );
        }
        console.log("installer strings are up to date");
        return;
    }

    mkdirSync(dirname(outputPath), { recursive: true });
    // UTF-8 **with BOM**: `makensis /INPUTCHARSET UTF8` is more forgiving with a
    // BOM, and this matches the other non-ASCII scripts in the repo (see
    // scripts/**/*.ps1).
    writeFileSync(outputPath, "\uFEFF" + rendered, "utf8");
    console.log(
        `wrote ${outputPath} (${strings.keys.length} keys × ${LANGUAGES.length} languages)`,
    );
}

main().catch((error) => {
    console.error(`build-installer-strings: ${error.message}`);
    process.exit(1);
});
