#!/usr/bin/env python3
"""依赖漏洞扫描门禁：把两份锁文件逐包送 OSV，只放行"已评估过"的通告。

【为什么要单独一个脚本】`npm audit` 只覆盖 npm，`cargo audit` 需要装工具链；
而本仓的依赖面是"两份锁文件"——`backend/Cargo.lock` 与
`frontend/package-lock.json`。OSV 同时收录 crates.io 与 npm，一次批量查询就能
把两边都扫完，且不需要在本机装任何东西。

【为什么要有白名单】本仓有 5 条"修不动"的通告（上游把版本钉死 / 不再维护）。
没有白名单，门禁永远是红的；有了白名单，**新增**的通告才是一次真正的告警。
每条都必须写理由（见 `advisory-allowlist.json`），并定期复审。

【退出码】
- 0：没有白名单之外的通告（网络失败也走这里，但会打印醒目的 WARN —— 定时任务
  不该因为 OSV 抖一下就变红；真正重要的是"扫到了新东西"）。
- 1：存在白名单之外的通告。
- 2：参数/文件错误。

用法：
    python scripts/checks/check-dependency-advisories.py
    python scripts/checks/check-dependency-advisories.py --verbose
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

OSV_BATCH = "https://api.osv.dev/v1/querybatch"
ROOT = pathlib.Path(__file__).resolve().parents[2]
CARGO_LOCK = ROOT / "backend" / "Cargo.lock"
NPM_LOCK = ROOT / "frontend" / "package-lock.json"
ALLOWLIST = pathlib.Path(__file__).resolve().parent / "advisory-allowlist.json"
CHUNK = 200


def cargo_packages() -> list[tuple[str, str]]:
    """从 Cargo.lock 里取 (name, version)。

    Cargo.lock 是 TOML，但每个 `[[package]]` 段的头两行固定是 name/version；
    为这点信息引一个 TOML 解析器不值得，也不该让门禁依赖额外包。
    """
    text = CARGO_LOCK.read_text(encoding="utf-8")
    packages: list[tuple[str, str]] = []
    for block in text.split("[[package]]")[1:]:
        name = version = None
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("name = ") and name is None:
                name = line[len("name = ") :].strip('"')
            elif line.startswith("version = ") and version is None:
                version = line[len("version = ") :].strip('"')
            elif name and version:
                break
        if name and version:
            packages.append((name, version))
    return packages


def npm_packages() -> list[tuple[str, str]]:
    """从 package-lock.json 的 `packages` 表里取 (name, version)。

    嵌套安装会出现 `node_modules/a/node_modules/b` 这类键：包名取最后一个
    `node_modules/` 之后的部分。
    """
    lock = json.loads(NPM_LOCK.read_text(encoding="utf-8"))
    packages: list[tuple[str, str]] = []
    for key, meta in lock.get("packages", {}).items():
        if not key or "node_modules/" not in key:
            continue
        name = key.rsplit("node_modules/", 1)[-1]
        version = meta.get("version")
        if name and version:
            packages.append((name, version))
    return packages


def query_osv(ecosystem: str, packages: list[tuple[str, str]]) -> dict[tuple[str, str], list[dict]]:
    """批量查询；返回 {(name, version): [vuln, ...]}，只保留有命中的条目。"""
    hits: dict[tuple[str, str], list[dict]] = {}
    for start in range(0, len(packages), CHUNK):
        chunk = packages[start : start + CHUNK]
        payload = json.dumps(
            {
                "queries": [
                    {"package": {"name": name, "ecosystem": ecosystem}, "version": version}
                    for name, version in chunk
                ]
            }
        ).encode("utf-8")
        body = _post_with_retry(payload)
        results = body.get("results", [])
        if len(results) != len(chunk):
            raise RuntimeError(
                f"OSV returned {len(results)} results for {len(chunk)} queries"
            )
        for (name, version), result in zip(chunk, results):
            vulns = result.get("vulns") or []
            if vulns:
                hits[(name, version)] = vulns
    return hits


def _post_with_retry(payload: bytes) -> dict:
    last_error: Exception | None = None
    for attempt in range(3):
        request = urllib.request.Request(
            OSV_BATCH,
            data=payload,
            headers={"content-type": "application/json", "user-agent": "hifishifter-advisory-check"},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(1 + attempt)
    raise RuntimeError(f"OSV query failed after 3 attempts: {last_error}")


def load_allowlist() -> list[dict]:
    entries = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    for entry in entries:
        for field in ("package", "id", "reason"):
            if not entry.get(field):
                raise SystemExit(f"advisory-allowlist.json: entry missing '{field}': {entry}")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="print accepted advisories too")
    args = parser.parse_args()

    for path in (CARGO_LOCK, NPM_LOCK, ALLOWLIST):
        if not path.is_file():
            print(f"ERROR: missing {path}", file=sys.stderr)
            return 2

    allowlist = load_allowlist()
    allowed = {(entry["package"], entry["id"]) for entry in allowlist}

    crates = cargo_packages()
    modules = npm_packages()
    print(f"scanning {len(crates)} crates + {len(modules)} npm packages against OSV ...")

    try:
        hits = query_osv("crates.io", crates)
        hits.update(query_osv("npm", modules))
    except RuntimeError as error:
        # 网络问题不该让定时任务变红（见文件头注释），但必须留下痕迹。
        print(f"WARN: {error}")
        print("WARN: advisory scan skipped -- re-run when the network is back")
        return 0

    accepted: list[tuple[str, str, list[str]]] = []
    unexpected: list[tuple[str, str, list[str]]] = []
    for (name, version), vulns in sorted(hits.items()):
        ids = [vuln.get("id", "?") for vuln in vulns]
        if any((name, vuln_id) in allowed for vuln_id in ids):
            accepted.append((name, version, ids))
        else:
            unexpected.append((name, version, ids))

    for name, version, ids in accepted:
        if args.verbose:
            print(f"  accepted  {name} {version}: {', '.join(ids)}")
    print(f"accepted (allow-listed): {len(accepted)}")
    print(f"unexpected: {len(unexpected)}")

    if unexpected:
        print("\nUnallowlisted advisories -- either upgrade the dependency or add a reviewed")
        print("entry to scripts/checks/advisory-allowlist.json with a reason:\n")
        for name, version, ids in unexpected:
            print(f"  {name} {version}: {', '.join(ids)}")
        return 1

    print("OK: no unallowlisted advisories")
    return 0


if __name__ == "__main__":
    sys.exit(main())
