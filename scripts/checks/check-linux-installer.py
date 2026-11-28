#!/usr/bin/env python3
"""Static gate for the Linux installer: the wiring that no shell parser would catch.

[Why this exists] The Linux download is a self-extracting wizard, and every part of it fails as
"green but wrong":

  * a string key that exists in the catalog but that no page displays (dead copy in five
    languages), or a page that shows a hard-coded English sentence;
  * a removal path that does not go through the safety predicate, so a hand-edited manifest can
    delete something that is not ours;
  * a payload stub that lost its marker line (the .run then silently extracts garbage);
  * package.sh staging a script the wizard does not have, or missing one it needs.

Only the CI round trip can prove that the installer really installs and uninstalls; this gate
proves the parts that would otherwise only be discovered by a user.

Run: python scripts/checks/check-linux-installer.py
"""

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
INSTALLER = ROOT / "scripts" / "release" / "linux-installer"
PACKAGE = ROOT / "scripts" / "build" / "unix" / "package.sh"
CATALOG = ROOT / "frontend" / "src" / "i18n"
LOCALES = ["en-US", "zh-CN", "zh-TW", "ja-JP", "ko-KR"]

findings = []


def read(path):
    return path.read_text(encoding="utf-8")


def report(message):
    findings.append(message)


def function_body(source, name):
    """Return the text of 'name() { ... }' -- brace nesting is not used in these scripts."""
    start = source.find(name + "() {")
    if start < 0:
        return ""
    end = source.find("\n}\n", start)
    return source[start:end if end > 0 else len(source)]


run_sh = read(INSTALLER / "run.sh")
ui_sh = read(INSTALLER / "ui.sh")
paths_sh = read(INSTALLER / "paths.sh")
strings_sh = read(INSTALLER / "strings.sh")
stub_sh = read(INSTALLER / "self-extract.sh")
install_sh = read(INSTALLER / "install.sh")
package_sh = read(PACKAGE)

# -- the generated string table --------------------------------------------
keys = set(re.findall(r"^\s+\[([a-z0-9_]+)\]=", strings_sh, re.M))
if not keys:
    report("strings.sh has no entries; run scripts/release/build-linux-installer-strings.mjs")

for locale in ("en_US", "zh_CN", "zh_TW", "ja_JP", "ko_KR"):
    if f"declare -A HFS_I18N_{locale}=(" not in strings_sh:
        report(f"strings.sh is missing the {locale} table")

# Every key the wizard displays, and every failure it reports, comes from the catalog.
used = set(re.findall(r"\bt ([a-z0-9_]+)", run_sh + ui_sh))
used |= set(re.findall(r"\bdie ([a-z0-9_]+)", run_sh))
used = {key for key in used if not key.isdigit()}

for key in sorted(used - keys):
    report(f"run.sh/ui.sh references '{key}', which is not in strings.sh")
for key in sorted(keys - used):
    report(f"'{key}' is translated into five languages but no page displays it")

# The catalog is the single source of those strings: a key that only exists in strings.sh
# (hand-edited there) would be overwritten on the next generation, so it must not exist.
for locale in LOCALES:
    source = read(CATALOG / f"{locale}.ts")
    translated = set(re.findall(r"^    installer_linux_([a-z0-9_]+):", source, re.M))
    if translated != keys:
        missing = sorted(keys - translated)
        extra = sorted(translated - keys)
        if missing:
            report(f"{locale}.ts is missing installer_linux_ keys: {', '.join(missing)}")
        if extra:
            report(f"{locale}.ts has installer_linux_ keys strings.sh does not: {', '.join(extra)}")

# -- pages are localized ---------------------------------------------------
# A wizard page with a hard-coded sentence is how the NSIS installer shipped Chinese to every
# user for a while. Every dialog call has to pull its copy from the catalog.
for match in re.finditer(r"^\s*(ui_(?:msg|error|text|radio|check|confirm|input|dir))\s+(.*)$", run_sh, re.M):
    call, arguments = match.group(1), match.group(2)
    if "$(t " not in arguments:
        report(f"run.sh: {call} is called without a catalog string: {arguments.strip()[:70]}")

# -- removals go through one predicate -------------------------------------
if "hfs_is_safe_remove() {" not in paths_sh:
    report("paths.sh no longer defines hfs_is_safe_remove")

for name in ("hfs_do_copy_file", "hfs_do_copy_tree", "hfs_do_remove"):
    body = function_body(run_sh, name)
    if not body:
        report(f"run.sh is missing {name}")
    elif "hfs_is_safe_remove" not in body:
        report(f"{name} writes or removes without asking hfs_is_safe_remove")

# No bare "rm -rf" outside the two functions that validate their argument first.
current = None
for number, line in enumerate(run_sh.splitlines(), 1):
    header = re.match(r"^([a-z_]+)\(\) \{", line)
    if header:
        current = header.group(1)
    elif line.startswith("}"):
        current = None
    if line.lstrip().startswith("#"):
        continue
    if "rm -rf" in line and current not in ("hfs_do_remove", "hfs_do_copy_tree"):
        if 'rm -rf -- "$HFS_WORK"' not in line:
            report(f"run.sh:{number}: rm -rf outside the guarded helpers ({current})")

# -- the plan and the manifest ---------------------------------------------
if '"$gen/manifest"' not in run_sh:
    report("run.sh no longer writes the install manifest into the plan")
elif run_sh.find("total\\t%s") < run_sh.find('"$gen/manifest"'):
    report("the manifest is written after the plan total; a failed copy would still look installed")
if "path=" not in run_sh:
    report("the manifest no longer records path= lines; the uninstaller has nothing to read")

# -- the self-extracting container -----------------------------------------
stub_lines = stub_sh.rstrip("\n").splitlines()
if not stub_lines or stub_lines[-1] != "__HFS_PAYLOAD_BELOW__":
    report("self-extract.sh must end with the __HFS_PAYLOAD_BELOW__ marker line")
if "tail -n +" not in stub_sh:
    report("self-extract.sh no longer streams the payload with tail")

for needle in ("make_self_extracting_run", "linux-installer/self-extract.sh"):
    if needle not in package_sh:
        report(f"package.sh no longer uses {needle}")

for staged in ("install.sh", "run.sh", "ui.sh", "paths.sh", "strings.sh", "uninstall.sh"):
    if staged not in package_sh:
        report(f"package.sh does not stage {staged} into the installer payload")

# -- the entry points ------------------------------------------------------
if 'exec "$here/installer/run.sh" "$@"' not in install_sh:
    report("install.sh no longer forwards to installer/run.sh")
if "uninstall.sh" not in package_sh:
    report("uninstall.sh is not part of the payload")

if findings:
    print("linux installer wiring is inconsistent:")
    for finding in findings:
        print("  - " + finding)
    sys.exit(1)

print(f"linux installer wiring is consistent ({len(keys)} keys, 5 languages)")
