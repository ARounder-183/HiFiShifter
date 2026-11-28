/**
 * The five-language catalog, loaded as plain data.
 *
 * [Why this module exists] Two generators read the same catalog:
 * `build-installer-strings.mjs` (the Windows NSIS `LangString` fragment) and
 * `build-macos-pkg.mjs` (the macOS Distribution XML plus its `.lproj` resources). Each used
 * to carry its own copy of "resolve esbuild, transform the .ts, evaluate it" — and a loader
 * whose failure mode is "quietly dropped one string" is exactly what must not exist twice.
 *
 * [Why esbuild instead of regex-parsing the .ts] The catalog contains multi-line strings as
 * well as comments; a regex would eventually be tripped up by one comment line or one
 * newline, and the failure mode is "quietly dropping one string". Compiling it to JS with the
 * esbuild already in the repo and evaluating it yields the real object.
 */

import { readFileSync } from "node:fs";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { createRequire } from "node:module";

const scriptDir = dirname(fileURLToPath(import.meta.url));
export const repoRoot = resolve(scriptDir, "..", "..");
export const catalogDir = join(repoRoot, "frontend", "src", "i18n");

/** Every locale the catalog ships. Adding one here reaches both installers at once. */
export const LOCALE_IDS = ["en-US", "zh-CN", "zh-TW", "ja-JP", "ko-KR"];

/** Key prefix that belongs to the installers rather than to the in-app UI. */
export const KEY_PREFIX = "installer_";

/**
 * Copy that only the macOS pkg uses.
 *
 * The NSIS generator skips this prefix, and the macOS generator reads it: the two installers
 * need different words for the same thing ("the VST3 folder you choose next" is a Windows
 * question; macOS always installs into /Library/Audio/Plug-Ins/VST3). Naming the platform in
 * the key keeps both generators honest without a second word list.
 */
export const MACOS_KEY_PREFIX = "installer_macos_";

/** `en-US` -> `enUS`: the name each catalog module exports. */
function exportName(locale) {
    const [language, region] = locale.split("-");
    return language + (region ? region.toUpperCase() : "");
}

let esbuildPromise = null;

/** esbuild is installed under frontend; load it once even when both generators run. */
function loadEsbuild() {
    if (!esbuildPromise) {
        // `import()` needs a file:// URL: the ESM loader rejects a Windows drive-letter path
        // as a protocol.
        const entry = createRequire(import.meta.url).resolve("esbuild", {
            paths: [join(repoRoot, "frontend")],
        });
        esbuildPromise = import(pathToFileURL(entry).href);
    }
    return esbuildPromise;
}

/** Compile one catalog `.ts` into JS and evaluate it, taking the exported dictionary. */
export async function loadCatalog(locale) {
    const esbuild = await loadEsbuild();
    const source = readFileSync(join(catalogDir, `${locale}.ts`), "utf8");
    const { code } = await esbuild.transform(source, { loader: "ts", format: "cjs" });
    const module = { exports: {} };
    // The catalog is a pure data module (only `export const x = {...} as const`);
    // evaluating it is safe.
    new Function("module", "exports", code)(module, module.exports);
    const name = exportName(locale);
    const catalog = module.exports[name];
    if (!catalog || typeof catalog !== "object") {
        throw new Error(`catalog ${locale} did not export ${name}`);
    }
    return catalog;
}

/**
 * Every `installer_`-prefixed entry, in all five languages, sorted by key.
 *
 * A key present in the reference catalog but missing (or blank) in another one throws: a
 * half-translated installer is worse than a build that stops.
 */
export async function collectInstallerStrings() {
    const catalogs = [];
    for (const locale of LOCALE_IDS) {
        catalogs.push({ locale, catalog: await loadCatalog(locale) });
    }

    const keys = Object.keys(catalogs[0].catalog)
        .filter((key) => key.startsWith(KEY_PREFIX))
        .sort();
    if (keys.length === 0) {
        throw new Error(`no keys with the ${KEY_PREFIX} prefix found in ${LOCALE_IDS[0]}`);
    }

    const missing = [];
    for (const { locale, catalog } of catalogs) {
        for (const key of keys) {
            const value = catalog[key];
            if (typeof value !== "string" || value.trim() === "") {
                missing.push(`${locale}: ${key}`);
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
