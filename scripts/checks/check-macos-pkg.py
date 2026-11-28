#!/usr/bin/env python3
"""Mechanical gate for the macOS pkg wizard: choices, localization, and the uninstaller.

The macOS download used to be one 'pkgbuild --root' component package. That is a real pkg, it
installs the right files, and Installer shows it as a single "Install" button -- which is
exactly the problem: no welcome page, no read-me, no licence, no choice of what to install, and
nothing that removes it afterwards. The Windows installer has had all four for a while, so the
gap was invisible in code review and only visible on a Mac.

The fix is a product archive ('productbuild --distribution') built from one component package
per choice, plus a second archive that removes everything. Every piece of that can be wrong in
a way that still builds:

  * the Distribution XML can reference a pkg-ref filename that package.sh never writes;
  * a choice can be enabled for the model library, which produces an install with no weights
    and an app that starts and cannot synthesise;
  * the .lproj folders can be named with the ISO tags the frontend uses (zh-CN), which
    Installer never looks for, leaving a five-language installer that shows English;
  * the uninstaller can remove paths the app does not use, and keep the ones it does;
  * the artifact can be built and never uploaded.

Only Installer itself can prove the round trip; that is the workflow's "Assert macOS pkg
layout and round trip" step. This gate proves the wiring and the generated layout.

[Why the workflow also runs this file] `productbuild` re-serializes the Distribution it embeds:
attributes are reordered and `installKBytes` is added. A grep for 'id="app" title=...' therefore
proves nothing about the packaged artifact, and would break the day the order changes. The
workflow passes the extracted Distribution back to this file with --distribution, so the rules
above are applied to the real artifact by the same code that checks the generator.

  python3 scripts/checks/check-macos-pkg.py                          # the whole gate
  python3 scripts/checks/check-macos-pkg.py --distribution FILE --kind=install|uninstall

Exit code 0 when every rule holds.
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ElementTree

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "scripts/release/build-macos-pkg.mjs"
UNINSTALL_SH = ROOT / "scripts/release/macos-pkg/uninstall.sh"
PACKAGE_SH = ROOT / "scripts/build/unix/package.sh"
WORKFLOW = ROOT / ".github/workflows/build.yml"
TAURI_CONF = ROOT / "backend/src-tauri/tauri.conf.json"
APP_PATHS = ROOT / "backend/hifishifter-kernel/src/app_paths.rs"

# Installer looks for these folder names, not for the ISO tags the frontend uses. The two
# Chinese ones are the canonical identifiers (`zh-Hans`, `zh-Hant`), not the legacy
# `zh_CN`/`zh_TW` spelling: productbuild rewrites the legacy names to these on the way into
# the archive, so "zh_CN.lproj" in the generated layout is a name no packaged pkg carries.
LOCALIZED = ("ja", "ko", "zh-Hans", "zh-Hant")
PAGES = ("welcome.html", "readme.html", "conclusion.html")

# A product version is not a model size; a number followed by a storage unit is.
SIZE_CLAIM = re.compile(r"\b\d+(?:\.\d+)?\s*(?:MB|GB|MiB|GiB|TB)\b")


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SystemExit("check-macos-pkg: cannot read %s: %s" % (path, error))


def step_block(workflow: str, step_name: str) -> str | None:
    """The YAML lines belonging to one step (up to the next '- name:' at the same level)."""
    lines = workflow.splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "- name: " + step_name:
            indent = len(line) - len(line.lstrip())
            block = [line]
            for follow in lines[index + 1 :]:
                stripped = follow.strip()
                if stripped.startswith("- name:") and (len(follow) - len(follow.lstrip())) == indent:
                    break
                block.append(follow)
            return "\n".join(block)
    return None


def code_only(text: str) -> str:
    """The text with whole-line comments removed.

    [Why every needle check goes through this] A rule that a commented-out line can satisfy is
    not a rule: commenting out "export HFS_PLATFORM_ARCH=universal" once kept a gate green
    while the artifact was named after the host. The needles below are all things a build
    actually executes, so a comment mentioning them proves nothing.
    """
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def run_generator(kind: str, out: pathlib.Path, formats: str | None = None) -> None:
    node = shutil.which("node")
    if node is None:
        raise SystemExit(
            "check-macos-pkg: node is not on PATH; the installer pages are generated from the "
            "i18n catalog and this gate cannot read them without it"
        )
    command = [node, str(GENERATOR), "--kind=" + kind, "--out=" + str(out), "--version=1.2.3"]
    if formats is not None:
        command.append("--formats=" + formats)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(
            "check-macos-pkg: the layout generator failed for --kind=%s:\n%s%s"
            % (kind, result.stdout, result.stderr)
        )


def generated_files(out: pathlib.Path) -> list[pathlib.Path]:
    return sorted(path for path in out.rglob("*") if path.is_file())


def package_entries(root: ElementTree.Element) -> dict[str, list[tuple[str, str | None]]]:
    """id -> [(file text, version attribute)] for every top-level <pkg-ref>.

    [Why a list and not one string per id] productbuild writes **two** entries for each
    component package: one that names the file ("#app.pkg", carrying version and
    installKBytes) and a second that carries only <bundle-version>. Collapsing them into a
    single string keeps whichever came last -- the version-less one -- and the gate then
    reports "carries no version" and "installs ''" for a product archive that is correct.
    """
    entries: dict[str, list[tuple[str, str | None]]] = {}
    for ref in root.findall("pkg-ref"):
        entries.setdefault(ref.get("id") or "", []).append(
            ((ref.text or "").strip(), ref.get("version"))
        )
    return entries


def package_files(entries: list[tuple[str, str | None]]) -> list[str]:
    """The component package filenames the entries name, with productbuild's '#' removed."""
    names = []
    for text, _ in entries:
        name = text[1:] if text.startswith("#") else text
        if name:
            names.append(name)
    return names


def check_install(dist: pathlib.Path, bundle_id: str, findings: list[str]) -> None:
    try:
        root = ElementTree.parse(dist).getroot()
    except ElementTree.ParseError as error:
        findings.append("distribution.xml is not well-formed XML: %s" % error)
        return
    if root.tag != "installer-gui-script":
        findings.append("distribution.xml: root element is %r, not installer-gui-script" % root.tag)
        return
    if root.get("minSpecVersion") != "2":
        findings.append(
            "distribution.xml: minSpecVersion must be 2 -- <domains> and the boolean "
            "attributes are ignored by older readers"
        )

    # ── Choices ──
    outline = root.find("choices-outline")
    if outline is None:
        findings.append("distribution.xml: no <choices-outline>; Installer would show no choices")
        outline_ids: list[str] = []
    else:
        outline_ids = [line.get("choice") or "" for line in outline.findall("line")]

    choices = {choice.get("id"): choice for choice in root.findall("choice")}
    if outline_ids != ["app", "models", "vst3", "clap"]:
        findings.append(
            "distribution.xml: the choices page must offer app, models, vst3, clap in that "
            "order (found %s)" % (outline_ids or "none")
        )
    if sorted(choices) != sorted(outline_ids):
        findings.append(
            "distribution.xml: <choices-outline> and <choice> disagree (%s vs %s)"
            % (sorted(outline_ids), sorted(choices))
        )

    for choice_id, choice in choices.items():
        title = (choice.get("title") or "").strip()
        description = (choice.get("description") or "").strip()
        if not title or not description:
            findings.append(
                "distribution.xml: choice %r has an empty title or description; Installer shows "
                "a blank row" % choice_id
            )
        if choice.get("start_selected") != "true":
            findings.append(
                "distribution.xml: choice %r does not start selected; a default install would "
                "skip it" % choice_id
            )
        if len(choice.findall("pkg-ref")) != 1:
            findings.append(
                "distribution.xml: choice %r must reference exactly one component package"
                % choice_id
            )

    # The app and the model library are the two pieces nothing can run without. Making either
    # selectable is the mistake that produces "the app opens and never synthesises".
    for choice_id in ("app", "models"):
        if choice_id in choices and choices[choice_id].get("enabled") != "false":
            findings.append(
                "distribution.xml: choice %r must not be deselectable -- there is no in-app "
                "model download and no fallback payload" % choice_id
            )
    for choice_id in ("vst3", "clap"):
        if choice_id in choices and choices[choice_id].get("enabled") != "true":
            findings.append(
                "distribution.xml: choice %r must be selectable -- it is the whole point of the "
                "choices page" % choice_id
            )

    # ── Package references ──
    refs = package_entries(root)
    for choice_id, choice in choices.items():
        referenced = choice.find("pkg-ref")
        if referenced is None:
            continue
        target = referenced.get("id")
        if target not in refs:
            findings.append(
                "distribution.xml: choice %r points at pkg-ref %r, which is never defined"
                % (choice_id, target)
            )
            continue
        named = package_files(refs[target])
        if choice_id + ".pkg" not in named:
            findings.append(
                "distribution.xml: choice %r installs %r; package.sh writes %r"
                % (choice_id, named or "nothing", choice_id + ".pkg")
            )
    for target, entries in refs.items():
        if not any(version for _, version in entries):
            findings.append(
                "distribution.xml: pkg-ref %r carries no version; productbuild rejects a "
                "reference that does not match the component package" % target
            )

    # ── Wizard pages ──
    for tag, filename in (
        ("welcome", "welcome.html"),
        ("readme", "readme.html"),
        ("license", "license.txt"),
        ("conclusion", "conclusion.html"),
    ):
        element = root.find(tag)
        if element is None:
            findings.append(
                "distribution.xml: no <%s> page -- the wizard degrades to a bare Install button"
                % tag
            )
            continue
        if element.get("file") != filename:
            findings.append(
                "distribution.xml: <%s file=%r> should name %r"
                % (tag, element.get("file"), filename)
            )

    # ── Host requirements ──
    volume_check = root.find("volume-check")
    if volume_check is None or volume_check.find("allowed-os-versions/os-version") is None:
        findings.append(
            "distribution.xml: no <volume-check>/<allowed-os-versions>; the package would "
            "install on a macOS the binaries were never built for"
        )
    must_close = root.find("must-close")
    if must_close is None:
        findings.append(
            "distribution.xml: no <must-close>; overwriting a running app fails halfway through "
            "the install instead of offering to quit it"
        )
    else:
        apps = [app.get("id") for app in must_close.findall("app")]
        if apps != [bundle_id]:
            findings.append(
                "distribution.xml: <must-close> names %s, but the app's bundle identifier is %r"
                % (apps or "nothing", bundle_id)
            )

    domains = root.find("domains")
    if domains is None or domains.get("enable_anywhere") != "false":
        findings.append(
            'distribution.xml: <domains enable_anywhere="false"> is missing; the payload uses '
            "absolute paths, so a second volume would receive a half-working copy"
        )
    options = root.find("options")
    if options is None or options.get("customize") != "always":
        findings.append(
            'distribution.xml: <options customize="always"> is missing; Installer hides the '
            "choices page behind a Customize button the user may never press"
        )


def check_uninstall(dist: pathlib.Path, findings: list[str]) -> None:
    try:
        root = ElementTree.parse(dist).getroot()
    except ElementTree.ParseError as error:
        findings.append("uninstaller distribution.xml is not well-formed XML: %s" % error)
        return
    if root.tag != "installer-gui-script":
        findings.append("uninstaller distribution.xml: root element is %r" % root.tag)
        return
    options = root.find("options")
    if options is None or options.get("customize") != "never":
        findings.append(
            'uninstaller distribution.xml: <options customize="never"> is missing; the '
            "uninstaller asks nothing, so it must not present a choices page"
        )
    if root.find("license") is not None:
        findings.append(
            "uninstaller distribution.xml: it asks the user to accept a licence in order to "
            "remove the product"
        )
    for tag in ("welcome", "readme", "conclusion"):
        if root.find(tag) is None:
            findings.append(
                "uninstaller distribution.xml: no <%s> page; the user cannot see what is about "
                "to be deleted" % tag
            )
    refs = package_entries(root)
    named = [name for entries in refs.values() for name in package_files(entries)]
    if named.count("uninstaller.pkg") != 1:
        findings.append(
            "uninstaller distribution.xml: expected exactly one pkg-ref to uninstaller.pkg, "
            "found %s"
            % [(ref.get("id"), (ref.text or "").strip()) for ref in root.findall("pkg-ref")]
        )


def check_one_distribution(argv: list[str]) -> int:
    """Validate one Distribution XML, as extracted from a built product archive."""
    path = None
    kind = "install"
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--distribution":
            index += 1
            if index >= len(argv):
                raise SystemExit("check-macos-pkg: --distribution requires a path")
            path = pathlib.Path(argv[index])
        elif argument.startswith("--distribution="):
            path = pathlib.Path(argument.split("=", 1)[1])
        elif argument.startswith("--kind="):
            kind = argument.split("=", 1)[1]
        else:
            raise SystemExit("check-macos-pkg: unknown argument %r" % argument)
        index += 1
    if path is None:
        raise SystemExit("check-macos-pkg: --distribution requires a path")
    if kind not in ("install", "uninstall"):
        raise SystemExit("check-macos-pkg: --kind must be install or uninstall")
    if not path.is_file():
        raise SystemExit("check-macos-pkg: no such file: %s" % path)

    try:
        identifier = json.loads(read(TAURI_CONF))["identifier"]
    except (json.JSONDecodeError, KeyError) as error:
        raise SystemExit(
            "check-macos-pkg: cannot read the identifier from tauri.conf.json: %s" % error
        )

    findings: list[str] = []
    if kind == "install":
        check_install(path, identifier, findings)
    else:
        check_uninstall(path, findings)
    if findings:
        for finding in findings:
            print("check-macos-pkg: %s" % finding, file=sys.stderr)
        return 1
    print("check-macos-pkg: the packaged Distribution still says what it said before packaging")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if "--distribution" in argv:
        return check_one_distribution(argv)

    findings: list[str] = []
    package_raw = read(PACKAGE_SH)
    package_code = code_only(package_raw)
    workflow = read(WORKFLOW)
    uninstall_raw = read(UNINSTALL_SH)
    uninstall = code_only(uninstall_raw)
    app_paths = read(APP_PATHS)

    try:
        identifier = json.loads(read(TAURI_CONF))["identifier"]
    except (json.JSONDecodeError, KeyError) as error:
        raise SystemExit(
            "check-macos-pkg: cannot read the identifier from tauri.conf.json: %s" % error
        )

    # ── 1. The generated layout ──
    with tempfile.TemporaryDirectory(prefix="hfs-macpkg-") as temp:
        install = pathlib.Path(temp) / "install"
        uninstaller = pathlib.Path(temp) / "uninstaller"
        run_generator("install", install)
        run_generator("uninstall", uninstaller)
        run_generator("install", pathlib.Path(temp) / "vst3only", formats="vst3")

        check_install(install / "distribution.xml", identifier, findings)
        check_uninstall(uninstaller / "distribution.xml", findings)

        # --formats has to reach the choices page: a build without CLAP must not offer it.
        vst3_only = ElementTree.parse(
            pathlib.Path(temp) / "vst3only" / "distribution.xml"
        ).getroot()
        offered = [line.get("choice") for line in vst3_only.findall("choices-outline/line")]
        if offered != ["app", "models", "vst3"]:
            findings.append(
                "distribution.xml: --formats=vst3 must offer app, models, vst3 only (found %s); "
                "a choice with no component package makes productbuild fail" % offered
            )

        # English at the resource root is the fallback for every language without a folder.
        for page in PAGES:
            if not (install / page).is_file():
                findings.append(
                    "the installer layout has no root %s; a language without a .lproj folder "
                    "would show an empty page" % page
                )
        if not (install / "Localizable.strings").is_file():
            findings.append("the installer layout has no root Localizable.strings")
        for language in LOCALIZED:
            for page in PAGES:
                if not (install / (language + ".lproj") / page).is_file():
                    findings.append(
                        "the installer layout has no %s.lproj/%s; Installer would show English"
                        % (language, page)
                    )
            strings = install / (language + ".lproj") / "Localizable.strings"
            if not strings.is_file():
                findings.append(
                    "the installer layout has no %s.lproj/Localizable.strings; the choice titles "
                    "would stay English" % language
                )
            elif strings.read_text(encoding="utf-8").count(" = ") < 2:
                findings.append("%s.lproj/Localizable.strings has fewer than two entries" % language)

        # A .lproj holding the English page is not a translation; it is a copy nobody notices.
        english = (install / "welcome.html").read_text(encoding="utf-8")
        for language in LOCALIZED:
            page = install / (language + ".lproj") / "welcome.html"
            if page.is_file() and page.read_text(encoding="utf-8") == english:
                findings.append(
                    "%s.lproj/welcome.html is byte-identical to the English page; the copy was "
                    "never translated" % language
                )

        # The licence page must be the repository's LICENSE, byte for byte: a translated or
        # reformatted licence is a different legal document.
        license_copy = install / "license.txt"
        if not license_copy.is_file():
            findings.append("the installer layout has no license.txt")
        elif license_copy.read_bytes() != (ROOT / "LICENSE").read_bytes():
            findings.append("license.txt is not a byte copy of LICENSE")

        # No concrete model sizes anywhere in the generated copy: the numbers live in
        # models.json and nowhere else.
        for path in generated_files(install) + generated_files(uninstaller):
            match = SIZE_CLAIM.search(path.read_text(encoding="utf-8", errors="replace"))
            if match:
                findings.append(
                    "generated %s restates a model size (%r); the manifest owns those numbers"
                    % (path.name, match.group(0))
                )

    # ── 2. package.sh must build what the distribution references ──
    if package_code.count("productbuild --distribution") != 2:
        findings.append(
            "scripts/build/unix/package.sh: expected two productbuild --distribution calls (the "
            "installer and the uninstaller), found %d"
            % package_code.count("productbuild --distribution")
        )
    for needle, why in (
        ("--kind=install", "the installer layout is never generated"),
        ("--kind=uninstall", "the uninstaller layout is never generated"),
        ("--package-path", "productbuild is not told where the component packages are"),
        ("--resources", "productbuild is not told where the localized pages are"),
        ("--nopayload", "the uninstaller component carries a payload it does not need"),
        ("--scripts", "the uninstaller's removal script is never attached to the package"),
        ("ensure_macos_pkg_node_deps", "the layout generator's esbuild dependency is never ensured"),
        (
            "npm ci",
            "the esbuild dependency is never ensured; --reuse-frontend installs nothing, so a "
            "fresh checkout cannot generate the layout",
        ),
        ("com.arounder.hifishifter.$choice", "component identifiers are not derived from the choice"),
        ("Uninstall HiFiShifter.pkg", "the uninstaller is not copied into the payload"),
        ("macos_component_pkg app", "the app component package is never built"),
        ("macos_component_pkg models", "the model-library component package is never built"),
        # A relocatable bundle is the bug this gate exists to prevent: Installer finds the
        # bundle elsewhere on disk, upgrades that copy, writes the receipt, and leaves
        # /Applications empty -- which looks exactly like a working install until it is run.
        (
            "BundleIsRelocatable",
            "component bundles stay relocatable; Installer would install over a copy it finds "
            "elsewhere on disk instead of the payload path",
        ),
        ("--component-plist", "the component plist that pins the bundles is never passed"),
        (
            '"Applications/HiFiShifter.app"',
            "the app component is not pinned to /Applications",
        ),
    ):
        if needle not in package_code:
            findings.append("scripts/build/unix/package.sh: %s (%r missing)" % (why, needle))
    for choice in ("vst3", "clap"):
        if "macos_component_pkg " + choice not in package_code:
            findings.append(
                "scripts/build/unix/package.sh: the %r component package is never built" % choice
            )
    # The order that matters is the one in the entry point, not the order of the definitions:
    # the unified package embeds a copy of the uninstaller, so it has to exist first.
    entry_point = package_raw.find("# ── Entry point")
    if entry_point < 0:
        findings.append("scripts/build/unix/package.sh: the entry point marker is gone")
    else:
        calls = code_only(package_raw[entry_point:])
        uninstaller_call = calls.find("package_macos_uninstaller")
        unified_call = calls.find("package_macos_unified")
        if uninstaller_call < 0 or unified_call < 0:
            findings.append(
                "scripts/build/unix/package.sh: the entry point no longer packages both the "
                "uninstaller and the unified package"
            )
        elif uninstaller_call > unified_call:
            findings.append(
                "scripts/build/unix/package.sh: the uninstaller must be packaged before the "
                "unified package, which embeds a copy of it"
            )

    # ── 3. The uninstaller must remove what the app actually writes ──
    product_dir_match = re.search(r'PRODUCT_DIR: &str = "([^"]+)"', app_paths)
    if product_dir_match is None:
        findings.append(
            "backend/hifishifter-kernel/src/app_paths.rs: PRODUCT_DIR is no longer a plain string "
            "literal, so this gate cannot derive the directory names"
        )
    else:
        product_dir = product_dir_match.group(1)
        # Derived data goes, user data stays. That distinction is what makes the uninstaller
        # safe to run: "models-user" holds weights the installer never wrote, and the plug-in
        # notepad is text somebody typed.
        for relative in (
            "Library/Caches/" + product_dir,
            "Library/Logs/" + product_dir,
        ):
            if relative not in uninstall:
                findings.append(
                    "scripts/release/macos-pkg/uninstall.sh: %r is never removed, but it is "
                    "derived data the app writes (see app_paths.rs)" % relative
                )
        support = "Library/Application Support/" + product_dir
        for line in uninstall.splitlines():
            if line.strip().startswith("remove ") and 'remove "$home/' + support in line:
                findings.append(
                    "scripts/release/macos-pkg/uninstall.sh: %r is deleted, but that directory "
                    "holds the user's settings, plug-in notes and their own model files "
                    "(models-user); an uninstaller must not remove data the installer never "
                    "wrote" % support
                )
        if not any("kept" in line and support in line for line in uninstall.splitlines()):
            findings.append(
                "scripts/release/macos-pkg/uninstall.sh: nothing records that %r is kept, so "
                "the user cannot find their settings or their own models afterwards" % support
            )
    for path in (
        "/Applications/HiFiShifter.app",
        "/Library/Audio/Plug-Ins/VST3/HiFiShifter.vst3",
        "/Library/Audio/Plug-Ins/CLAP/HiFiShifter.clap",
    ):
        if path not in uninstall:
            findings.append("scripts/release/macos-pkg/uninstall.sh: %r is never removed" % path)
    if "pkgutil --forget" not in uninstall:
        findings.append(
            "scripts/release/macos-pkg/uninstall.sh: the receipts are never forgotten, so the "
            "next installer treats the machine as an upgrade"
        )
    if not uninstall_raw.lstrip().startswith("#!/bin/bash"):
        findings.append("scripts/release/macos-pkg/uninstall.sh: no #!/bin/bash shebang")
    if "\nset -e" in uninstall or "\nset -eu" in uninstall or "\nset -euo" in uninstall:
        findings.append(
            "scripts/release/macos-pkg/uninstall.sh: set -e must not be used -- a path that does "
            "not exist is the normal case, and aborting halfway leaves a machine that is neither "
            "installed nor clean"
        )
    if not uninstall_raw.rstrip().endswith("exit 0"):
        findings.append(
            "scripts/release/macos-pkg/uninstall.sh: it must end with exit 0; a non-zero exit "
            "makes Installer report a failed removal after a successful one"
        )

    # ── 4. The workflow must prove the round trip and ship both artifacts ──
    assert_step = step_block(workflow, "Assert macOS pkg layout and round trip")
    if assert_step is None:
        findings.append(
            "build.yml: step 'Assert macOS pkg layout and round trip' is missing -- nothing "
            "would install the package before a user does"
        )
    else:
        assert_code = code_only(assert_step)
        if "continue-on-error" in assert_code:
            findings.append(
                "build.yml: 'Assert macOS pkg layout and round trip' must not be "
                "continue-on-error; an assertion that cannot fail is documentation"
            )
        for needle, why in (
            ("pkgutil --expand-full", "the archive's Distribution and resources are never read"),
            (".lproj", "the localized pages are never checked"),
            ("-uninstaller.pkg", "the uninstaller is never exercised"),
            ("installer -allowUntrusted", "nothing actually installs the package"),
            ("codesign --verify", "the installed app's signature is never verified"),
            ("pkgutil --pkgs", "the receipts are never checked"),
            (
                "check-macos-pkg.py --distribution",
                "the packaged Distribution is not checked by the same rules as the generator",
            ),
        ):
            if needle not in assert_code:
                findings.append(
                    "build.yml: 'Assert macOS pkg layout and round trip': %s (%r missing)"
                    % (why, needle)
                )

    upload = code_only(step_block(workflow, "Upload unified package (Unix)") or "")
    if "-uninstaller" not in upload or "matrix.name" not in upload:
        findings.append(
            "build.yml: the Unix upload glob ignores *-uninstaller.*; the artifact would be "
            "built and thrown away"
        )

    if findings:
        for finding in findings:
            print("check-macos-pkg: %s" % finding, file=sys.stderr)
        return 1
    print(
        "check-macos-pkg: the product archive offers real choices, is localized in four "
        "languages, and the uninstaller removes what the installer wrote"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
