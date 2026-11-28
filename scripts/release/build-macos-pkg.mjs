/**
 * Build the macOS pkg layout — the Distribution XML and its localized resources — from the
 * same five-language catalog the Windows NSIS installer reads.
 *
 * [Why a product archive instead of a bare component package] `pkgbuild --root` produces a
 * component package, and Installer shows it as a single "Install" button: no welcome text, no
 * readme, no licence page, no choice of what to install, no conclusion. The Windows installer
 * has had all of that for a while (components page, licence page, uninstall options), so the
 * macOS download was the one place where a user could not say "I only want the app" — and
 * could not find out what was about to be written where. `productbuild --distribution` is
 * what turns the same payload into that wizard.
 *
 * [Why the copy is generated and not written here] Installer localizes in exactly two ways:
 * `.lproj` folders for welcome/readme/conclusion/licence, and `Localizable.strings` for
 * the choice `title`/`description` attributes (the schema calls them "localization keys").
 * Both are generated from the `installer_` entries in frontend/src/i18n/*.ts, which are
 * already guarded by `catalogIntegrity` (all keys in all five languages, no stray
 * whitespace, no U+2026). A hand-written second copy of five translations would be the one
 * place in the repo without a test, and it would drift within one release.
 *
 * Usage:
 *   node scripts/release/build-macos-pkg.mjs --kind=install   --out=DIR --version=0.2.0 \
 *        --formats=vst3,clap
 *   node scripts/release/build-macos-pkg.mjs --kind=uninstall --out=DIR --version=0.2.0
 *
 * Writes into DIR:
 *   distribution.xml                  the Installer wizard definition
 *   welcome.html readme.html conclusion.html   English (the fallback for every language)
 *   license.txt                       only for --kind=install (a byte copy of LICENSE)
 *   Localizable.strings               English choice titles and descriptions
 *   <lproj>/{welcome,readme,conclusion}.html, Localizable.strings
 *
 * The `.lproj` names are the ones Installer itself ships (zh_CN / zh_TW / ja / ko): the
 * legacy Apple spelling for Chinese, the ISO code for the other two. English lives at the
 * root on purpose — a non-localized resource is the documented fallback for every language
 * that has no folder, so French users get English instead of an empty page.
 */

import { copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";

import { collectInstallerStrings, repoRoot } from "./catalog.mjs";

/** Catalog locale -> `.lproj` folder name (null = the unlocalized root). */
const LPROJ = {
    "en-US": null,
    "zh-CN": "zh_CN",
    "zh-TW": "zh_TW",
    "ja-JP": "ja",
    "ko-KR": "ko",
};

/** Catalog locale -> the HTML `lang` attribute. */
const HTML_LANG = {
    "en-US": "en",
    "zh-CN": "zh-CN",
    "zh-TW": "zh-TW",
    "ja-JP": "ja",
    "ko-KR": "ko",
};

/** Reference locale: its text becomes the key in every `Localizable.strings`. */
const REFERENCE = "en-US";

/**
 * Component package identifiers.
 *
 * [Why these are not `com.hifishifter.unified`] A product archive is a set of component
 * packages, and Installer records one receipt per component. Splitting the payload therefore
 * means new identifiers; the old ones are still forgotten by the uninstaller so an upgrade
 * from a previous release does not leave a receipt claiming a package that no longer exists.
 */
const PKG_IDS = {
    app: "com.arounder.hifishifter.app",
    models: "com.arounder.hifishifter.models",
    vst3: "com.arounder.hifishifter.vst3",
    clap: "com.arounder.hifishifter.clap",
    uninstaller: "com.arounder.hifishifter.uninstaller",
};

/**
 * Minimum macOS. Apple Silicon starts at 11.0, the universal package ships arm64, and both
 * slices are built by a toolchain that targets 11.0 — claiming support for an older Intel
 * release would be a promise nothing in CI can keep.
 */
const MIN_MACOS = "11.0";

function fail(message) {
    console.error("build-macos-pkg: " + message);
    process.exit(1);
}

// ── Arguments ───────────────────────────────────────────────────────────────
function parseArgs(argv) {
    const options = { kind: "", out: "", version: "", formats: "vst3,clap" };
    const known = {
        "--kind": "kind",
        "--out": "out",
        "--version": "version",
        "--formats": "formats",
    };
    for (let index = 0; index < argv.length; index += 1) {
        const raw = argv[index];
        const eq = raw.indexOf("=");
        const name = eq < 0 ? raw : raw.slice(0, eq);
        const key = known[name];
        if (!key) fail("unknown option: " + raw + " (see the header of this file)");
        if (eq >= 0) {
            options[key] = raw.slice(eq + 1);
        } else {
            index += 1;
            if (index >= argv.length) fail(name + " requires a value");
            options[key] = argv[index];
        }
    }
    if (options.kind !== "install" && options.kind !== "uninstall") {
        fail("--kind must be install or uninstall");
    }
    if (!options.out) fail("--out is required");
    if (!options.version) fail("--version is required (the numeric version in <pkg-ref>)");
    options.formats = options.formats
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean);
    for (const format of options.formats) {
        if (format !== "vst3" && format !== "clap") fail("unknown format: " + format);
    }
    return options;
}

// ── Choices ─────────────────────────────────────────────────────────────────
/**
 * One entry per choice the wizard shows.
 *
 * `required` becomes `enabled="false"`: Installer greys the checkbox out instead of
 * hiding it, which is the honest way to say "this is part of the product" — the same thing the
 * Windows installer does for the standalone app.
 *
 * [Why the model library is required] There is no in-app model download, and the unified
 * package strips the weights out of every bundle (they exist once, in the shared store). An
 * installation without them is an app that starts and then cannot synthesise anything. Windows
 * can offer the choice because its app publishes the library from its own bundle on first run;
 * macOS cannot.
 */
function installChoices(copy, formats) {
    const choices = [
        {
            id: "app",
            title: copy.installer_component_app,
            description: copy.installer_desc_app,
            required: true,
        },
        {
            id: "models",
            title: copy.installer_component_models,
            // Not installer_desc_models: that sentence ends with "Stored once", and gluing the
            // two together reads as one run-on line in every language. The macOS entry already
            // says "the app and every plug-in read this one copy", which is the same fact.
            description: copy.installer_macos_desc_models_required,
            required: true,
        },
    ];
    if (formats.includes("vst3")) {
        choices.push({
            id: "vst3",
            title: copy.installer_component_vst3,
            description: copy.installer_macos_desc_vst3,
            required: false,
        });
    }
    if (formats.includes("clap")) {
        choices.push({
            id: "clap",
            title: copy.installer_component_clap,
            description: copy.installer_macos_desc_clap,
            required: false,
        });
    }
    return choices;
}

/**
 * The uninstaller asks nothing: it removes everything HiFiShifter ever wrote, in one step.
 *
 * [Why there is no "keep the models" option here] On Windows the option exists because the
 * shared library is shared with the plug-ins of *other* installation forms and because a
 * reinstall can reuse it. On macOS there is one installation; keeping a model store with no
 * app and no plug-in left to read it is not a choice, it is a leftover.
 */
function uninstallChoices(copy) {
    return [
        {
            id: "uninstaller",
            title: copy.installer_name,
            description: copy.installer_macos_uninstall_welcome_title,
            required: true,
        },
    ];
}

// ── XML / HTML / strings rendering ──────────────────────────────────────────
function xmlEscape(text) {
    return String(text)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
}

function htmlEscape(text) {
    return xmlEscape(text);
}

/** `"` / `\` / newline escaping for a .strings value or key. */
function stringsEscape(text) {
    let out = "";
    for (const char of String(text)) {
        if (char === "\\") out += "\\\\";
        else if (char === '"') out += '\\"';
        else if (char === "\n") out += "\\n";
        else if (char === "\r") continue;
        else out += char;
    }
    return out;
}

/**
 * One Installer page.
 *
 * [Why `list` instead of HTML in the catalog] The catalog is shared with a Win32 wizard
 * that cannot render HTML at all. Keeping the copy as plain lines — one paragraph per line for
 * welcome/conclusion, one bullet per line for the readme — means a translator never edits
 * markup, and a stray `<` in a path cannot break the page.
 */
function page(lang, heading, body, list) {
    const lines = String(body)
        .split("\n")
        .map((line) => line.trim())
        .filter(Boolean);
    const rendered = list
        ? [
              "    <ul>",
              ...lines.map((line) => "        <li>" + htmlEscape(line) + "</li>"),
              "    </ul>",
          ]
        : lines.map((line) => "    <p>" + htmlEscape(line) + "</p>");
    return [
        "<!DOCTYPE html>",
        '<html lang="' + lang + '">',
        "<head>",
        '    <meta charset="utf-8">',
        "    <title>" + htmlEscape(heading) + "</title>",
        "</head>",
        "<body>",
        "    <h2>" + htmlEscape(heading) + "</h2>",
        ...rendered,
        "</body>",
        "</html>",
        "",
    ].join("\n");
}

function stringsFile(entries) {
    const lines = [
        "/* Generated by scripts/release/build-macos-pkg.mjs — do not edit by hand. */",
        "/* Keys are the English text from frontend/src/i18n/*.ts; values are this language. */",
        "",
    ];
    for (const [key, value] of entries) {
        lines.push('"' + stringsEscape(key) + '" = "' + stringsEscape(value) + '";');
    }
    return lines.join("\n") + "\n";
}

function distributionXml({ kind, copy, choices, version }) {
    const lines = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<!--",
        "  Generated by scripts/release/build-macos-pkg.mjs — do not edit by hand.",
        "",
        "  [Why <domains> and not rootVolumeOnly] rootVolumeOnly is deprecated, and its default",
        "  leaves enable_anywhere true: the payload uses absolute paths (/Applications,",
        "  /Library), so a second volume would receive a half-working copy. enable_anywhere",
        "  false restricts the wizard to the boot volume and removes the destination page.",
        "",
        "  [Why no hostArchitectures] The attribute is a promise about which Macs may install, and",
        "  the payload's slices differ between the universal release build and a single-",
        "  architecture one. Omitting it installs nothing that cannot run (the workflow asserts",
        "  every Mach-O's slices), while a stale list here is the one mistake that would lock the",
        "  package out of the very Macs it was built for.",
        "-->",
        '<installer-gui-script minSpecVersion="2">',
        "    <title>" + xmlEscape(copy.installer_name) + "</title>",
        "    <organization>com.arounder</organization>",
        '    <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>',
        '    <options customize="' + (kind === "install" ? "always" : "never") + '"/>',
        "    <volume-check>",
        "        <allowed-os-versions>",
        '            <os-version min="' + MIN_MACOS + '"/>',
        "        </allowed-os-versions>",
        "    </volume-check>",
        "    <!-- Installer offers to quit a running app instead of failing halfway through. -->",
        "    <must-close>",
        '        <app id="com.arounder.hifishifter"/>',
        "    </must-close>",
        '    <welcome file="welcome.html"/>',
        '    <readme file="readme.html"/>',
    ];
    if (kind === "install") {
        lines.push('    <license file="license.txt"/>');
    }
    lines.push('    <conclusion file="conclusion.html"/>', "    <choices-outline>");
    for (const choice of choices) {
        lines.push('        <line choice="' + choice.id + '"/>');
    }
    lines.push("    </choices-outline>");
    for (const choice of choices) {
        lines.push(
            '    <choice id="' +
                choice.id +
                '" title="' +
                xmlEscape(choice.title) +
                '" description="' +
                xmlEscape(choice.description) +
                '" enabled="' +
                (choice.required ? "false" : "true") +
                '" start_selected="true">',
            '        <pkg-ref id="' + PKG_IDS[choice.id] + '"/>',
            "    </choice>",
        );
    }
    for (const choice of choices) {
        lines.push(
            '    <pkg-ref id="' +
                PKG_IDS[choice.id] +
                '" version="' +
                xmlEscape(version) +
                '">' +
                choice.id +
                ".pkg</pkg-ref>",
        );
    }
    lines.push("</installer-gui-script>", "");
    return lines.join("\n");
}

// ── Entry point ─────────────────────────────────────────────────────────────
async function main() {
    const options = parseArgs(process.argv.slice(2));
    const { catalogs } = await collectInstallerStrings();

    const reference = catalogs.find((entry) => entry.locale === REFERENCE);
    if (!reference) fail("the reference locale " + REFERENCE + " is missing from the catalog");

    const build = options.kind === "install" ? installChoices : uninstallChoices;
    const englishChoices = build(reference.catalog, options.formats);

    const welcomeKey =
        options.kind === "install"
            ? "installer_macos_welcome"
            : "installer_macos_uninstall_welcome";
    const readmeKey =
        options.kind === "install" ? "installer_macos_readme" : "installer_macos_uninstall_readme";
    const conclusionKey =
        options.kind === "install"
            ? "installer_macos_conclusion"
            : "installer_macos_uninstall_conclusion";

    const distribution = distributionXml({
        kind: options.kind,
        copy: reference.catalog,
        choices: englishChoices,
        version: options.version,
    });
    // The caller (package.sh) hands us a directory it just created; creating it here as well
    // keeps the script usable on its own, which is what the static gate does.
    mkdirSync(options.out, { recursive: true });
    writeFileSync(join(options.out, "distribution.xml"), distribution, "utf8");

    for (const { locale, catalog } of catalogs) {
        const folder = LPROJ[locale];
        if (folder === undefined) fail("no .lproj folder is declared for " + locale);
        // Installer looks for "<language>.lproj" folders; a bare "ja" directory is
        // never consulted and the page silently falls back to English.
        const target = folder ? join(options.out, folder + ".lproj") : options.out;
        mkdirSync(target, { recursive: true });

        const lang = HTML_LANG[locale];
        writeFileSync(
            join(target, "welcome.html"),
            page(lang, catalog[welcomeKey + "_title"], catalog[welcomeKey + "_body"], false),
            "utf8",
        );
        writeFileSync(
            join(target, "readme.html"),
            page(lang, catalog[readmeKey + "_title"], catalog[readmeKey + "_body"], true),
            "utf8",
        );
        writeFileSync(
            join(target, "conclusion.html"),
            page(lang, catalog[conclusionKey + "_title"], catalog[conclusionKey + "_body"], false),
            "utf8",
        );

        // Choice titles and descriptions are looked up by the English text as the key.
        const choices = build(catalog, options.formats);
        if (choices.length !== englishChoices.length) {
            fail(
                locale +
                    " produced " +
                    choices.length +
                    " choices, expected " +
                    englishChoices.length,
            );
        }
        const entries = [];
        for (let index = 0; index < choices.length; index += 1) {
            entries.push([englishChoices[index].title, choices[index].title]);
            entries.push([englishChoices[index].description, choices[index].description]);
        }
        writeFileSync(join(target, "Localizable.strings"), stringsFile(entries), "utf8");
    }

    if (options.kind === "install") {
        // The licence page shows the repository LICENSE verbatim: a translated licence is a
        // different legal document, and the byte copy is also what lands in the payload.
        copyFileSync(join(repoRoot, "LICENSE"), join(options.out, "license.txt"));
    }

    console.log(
        "build-macos-pkg: " +
            options.kind +
            " layout written to " +
            options.out +
            " (" +
            catalogs.length +
            " languages, " +
            englishChoices.length +
            " choices)",
    );
}

main().catch((error) => fail(error.message));
