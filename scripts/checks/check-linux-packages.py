#!/usr/bin/env python3
"""Static gate for the Linux distribution: native packages first, a plain portable bundle second.

[Why this exists] The Linux download is now four deb/rpm packages plus a portable tarball, and
every part of it fails as "green but wrong":

  * a package whose contents write somewhere else than the FHS paths the app resolves --
    the app then starts and cannot find models.json, libwebgpu_dawn.so or its plug-in resources;
  * a deb that depends on the Debian library names and an rpm that inherited them, so
    "dnf install" refuses the package on Fedora;
  * a portable installer that quietly grew a dialog backend again (the old wizard's whiptail
    branch is what made it disappear on the page after "components");
  * an upload glob that still looks for '*-unified.run' and matches nothing, which costs a
    whole release run -- the same failure mode the artifact-name gate was written for.

Only a real distribution can prove that a package installs; the workflow does that on Ubuntu and
Fedora. This gate proves the wiring, and it runs on every platform.

Run: python scripts/checks/check-linux-packages.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_SH = ROOT / "scripts" / "build" / "unix" / "package.sh"
PACKAGE_DIR = ROOT / "scripts" / "release" / "linux-packages"
PORTABLE_DIR = ROOT / "scripts" / "release" / "linux-portable"
WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"

PACKAGES = ["hifishifter", "hifishifter-models", "hifishifter-vst3", "hifishifter-clap"]

findings: list[str] = []


def read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise SystemExit("check-linux-packages: cannot read %s: %s" % (path, error))


def report(message: str) -> None:
    findings.append(message)


def require(text: str, needle: str, message: str) -> None:
    if needle not in text:
        report(message)


def forbid(text: str, needle: str, message: str) -> None:
    if needle in text:
        report(message)


def function_body(source: str, name: str) -> str:
    """The text of 'name() { ... }' up to the closing brace in column 0."""
    start = source.find(name + "() {")
    if start < 0:
        return ""
    end = source.find("\n}\n", start)
    return source[start : end if end > 0 else len(source)]


def code_only(text: str) -> str:
    """Drop comment lines: a note that explains why a backend is gone is not a use of it."""
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


# -- 1. package.sh builds packages, not a self-extracting wizard ---------------
package_sh = read(PACKAGE_SH)

require(package_sh, "package_linux_packages() {", "package.sh has no package_linux_packages")
require(
    package_sh,
    "scripts/release/linux-packages/build.sh",
    "package.sh does not call scripts/release/linux-packages/build.sh",
)
for stale in ("make_self_extracting_run", "stage_linux_installer", "linux-installer/", "-unified.run"):
    forbid(package_sh, stale, "package.sh still references %r (the wizard is retired)" % stale)
require(
    package_sh,
    '-packages"',
    "package.sh no longer names the Linux artifact directory '...-packages'",
)
require(
    package_sh,
    '-unified"',
    "package.sh no longer names the macOS artifact '...-unified'",
)
require(
    package_sh,
    "scripts/release/linux-portable",
    "package.sh no longer stages the portable bundle from scripts/release/linux-portable",
)
require(
    package_sh,
    "patchelf --set-rpath",
    "package.sh does not rewrite the installed binary's rpath; /usr/lib/HiFiShifter would be "
    "invisible to the dynamic loader",
)

# -- 2. the packager maps the staged tree onto the FHS layout -----------------
build_sh = read(PACKAGE_DIR / "build.sh")

# [Why the function body and not the file] The header comment names the same paths to explain the
# layout; a check over the whole file would still pass after the real mapping was changed.
for function, path, why in [
    ("contents_app", "/usr/bin/hifishifter", "the app binary must be installed on PATH"),
    ("contents_app", "/usr/lib/HiFiShifter", "the private libraries live in /usr/lib/HiFiShifter"),
    ("contents_app", "/usr/share/applications/hifishifter.desktop", "the desktop entry"),
    ("contents_app", "/usr/share/mime/packages/hifishifter.xml", "the MIME definition"),
    ("contents_models", "/usr/share/HiFiShifter/models/", "the shared model store path"),
    ("contents_vst3", "/usr/lib/vst3/HiFiShifter.vst3", "the system VST3 folder"),
    ("contents_clap", "/usr/lib/clap/HiFiShifter.clap", "the system CLAP folder"),
    ("contents_clap", "/usr/lib/clap/HiFiShifter-resources", "the sibling CLAP resources"),
]:
    body = function_body(build_sh, function)
    if not body:
        report("build.sh is missing %s()" % function)
    elif path not in body:
        report("%s() never installs %s (%s)" % (function, path, why))

require(
    build_sh,
    'readonly NFPM_VERSION="v',
    "build.sh no longer pins the nfpm version",
)
require(
    build_sh,
    "verify_models_tree",
    "build.sh no longer validates the model store against models.json before packaging it",
)
require(
    build_sh,
    'pkg_arch="all"',
    "the models package lost its architecture-independent build",
)

# -- 3. the four package definitions -----------------------------------------
for name in PACKAGES:
    config = PACKAGE_DIR / "nfpm" / (name + ".yaml")
    if not config.is_file():
        report("missing package definition: nfpm/%s.yaml" % name)
        continue
    text = read(config)
    require(text, "name: " + name, "nfpm/%s.yaml declares a different package name" % name)
    require(text, "platform: linux", "nfpm/%s.yaml is not a Linux package" % name)
    require(text, "version: ${HFS_PKG_VERSION}", "nfpm/%s.yaml lost the version placeholder" % name)
    if name != "hifishifter-models":
        require(
            text,
            "overrides:",
            "nfpm/%s.yaml has no rpm overrides; the rpm would require Debian library names" % name,
        )

app_yaml = read(PACKAGE_DIR / "nfpm" / "hifishifter.yaml")
require(
    app_yaml,
    "hifishifter-models (= ${HFS_MODEL_VERSION})",
    "the app package no longer depends on the exact model version",
)
require(app_yaml, "libwebkit2gtk-4.1-0", "the app package lost its Debian WebKitGTK dependency")
require(app_yaml, "libgtk-3-0", "the app package lost its Debian GTK dependency")
require(app_yaml, "libasound2t64", "the app package lost its Debian ALSA dependency")
require(app_yaml, "webkit2gtk4.1", "the app package lost its RPM WebKitGTK dependency")
require(app_yaml, "gtk3", "the app package lost its RPM GTK dependency")
require(app_yaml, "alsa-lib", "the app package lost its RPM ALSA dependency")
require(app_yaml, "recommends:", "the app package no longer recommends the plug-in packages")

for name in ("hifishifter-vst3", "hifishifter-clap"):
    text = read(PACKAGE_DIR / "nfpm" / (name + ".yaml"))
    require(
        text,
        "hifishifter-models (= ${HFS_MODEL_VERSION})",
        "%s no longer depends on the exact model version" % name,
    )
    forbid(
        text,
        "hifishifter (=",
        "%s depends on the app package; a DAW-only install must not need it" % name,
    )

# -- 4. the desktop entry and the MIME definition ----------------------------
desktop = read(PACKAGE_DIR / "assets" / "hifishifter.desktop")
for needle, why in [
    ("Exec=hifishifter %F", "the menu entry must start the installed binary with files"),
    ("TryExec=hifishifter", "the menu entry must advertise the executable it needs"),
    ("Icon=hifishifter", "the menu entry must use the installed icon name"),
    ("MimeType=application/x-hifishifter-project;", "the menu entry lost the project association"),
    ("Terminal=false", "the menu entry would open a terminal"),
]:
    require(desktop, needle, "hifishifter.desktop: " + why)
for line in desktop.splitlines():
    if line.startswith("Exec=") and "AppImage" in line:
        report("hifishifter.desktop still starts the portable AppImage: " + line)

mime = read(PACKAGE_DIR / "assets" / "hifishifter.xml")
require(mime, 'type="application/x-hifishifter-project"', "the MIME definition lost its type")
require(mime, 'pattern="*.hshp"', "the MIME definition lost .hshp")
require(mime, 'pattern="*.hsp"', "the MIME definition lost .hsp")

# -- 5. the portable bundle has no dialog backend and no blind removal -------
for script in sorted(PORTABLE_DIR.glob("*.sh")):
    text = read(script)
    for forbidden in ("zenity", "whiptail", "dialog ", "hfs_maybe_open_terminal"):
        forbid(
            code_only(text),
            forbidden,
            "%s references %r: the portable installer must stay plain text"
            % (script.name, forbidden.strip()),
        )

portable_lib = read(PORTABLE_DIR / "lib.sh")
require(portable_lib, "hfs_is_safe_remove() {", "portable lib.sh lost the removal predicate")
require(portable_lib, "hfs_layout() {", "portable lib.sh lost the single path calculation")
require(portable_lib, "install-manifest", "portable lib.sh no longer names the install manifest")

install_sh = read(PORTABLE_DIR / "install-user.sh")
for flag in ("--yes", "--prefix", "--components"):
    require(install_sh, flag, "install-user.sh no longer accepts " + flag)
require(install_sh, "hfs_manifest_read", "install-user.sh no longer reads the previous manifest")
require(install_sh, "libfuse", "install-user.sh lost the libfuse fallback in the launcher")

uninstall_sh = read(PORTABLE_DIR / "uninstall-user.sh")
require(uninstall_sh, "hfs_manifest_read", "uninstall-user.sh does not read the install manifest")
require(uninstall_sh, "hfs_is_safe_remove", "uninstall-user.sh removes without the safety predicate")
require(uninstall_sh, "HFS_MANIFEST", "uninstall-user.sh never reads the manifest path")
for owned in (".vst3", ".clap", "install-manifest"):
    require(portable_lib, owned, "portable lib.sh never mentions " + owned)

# -- 6. the workflow uploads and verifies the packages -----------------------
workflow = read(WORKFLOW)
require(workflow, "-packages/**", "build.yml does not upload the -packages directory")
forbid(workflow, "-unified.run", "build.yml still looks for a *-unified.run artifact")
require(
    workflow,
    "scripts/checks/check-linux-packages.py",
    "build.yml no longer runs this gate before installing the packages",
)
for step in (
    "Assert Linux package layout",
    "Install and uninstall Linux packages (Ubuntu)",
    "Install and uninstall Linux packages (Fedora)",
    "Run the portable installer round trip",
):
    require(workflow, step, "build.yml is missing the %r step" % step)

# -- 7. the documentation describes the packages, not the wizard -------------
docs = [ROOT / "README.md"] + sorted((ROOT / "docs" / "i18n").glob("README_*.md"))
for doc in docs:
    text = read(doc)
    if ".deb" not in text or ".rpm" not in text:
        report("%s does not mention .deb/.rpm for Linux" % doc.relative_to(ROOT))
    if re.search(r"\.run\b", text):
        report("%s still presents the .run wizard as the Linux download" % doc.relative_to(ROOT))

manuals = [ROOT / "docs" / "i18n" / "USERMANUAL.md"] + sorted(
    (ROOT / "docs" / "i18n").glob("USERMANUAL_*.md")
)
for manual in manuals:
    text = read(manual)
    if re.search(r"\.run\b", text):
        report("%s still documents the .run installer" % manual.relative_to(ROOT))

if findings:
    print("Linux packaging is inconsistent:")
    for finding in findings:
        print("  - " + finding)
    sys.exit(1)

print("Linux packaging is consistent (%d packages, deb + rpm + portable)" % len(PACKAGES))
